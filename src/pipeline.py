"""
GuardedChatSession orchestrates one full turn end-to-end:

    raw input
      -> Layer 0: sanitize (+ collapsed / base64-decoded views) (src/guardrails/sanitizer.py)
      -> Layer 1: injection/jailbreak scan of every view        (src/guardrails/injection_detector.py)
      -> Layer 2: embedding topic relevance on the INPUT         (src/guardrails/topic_relevance.py)
                  (short follow-ups are also scored together with the previous user message)
      -> Layer 2b: LLM topic judge, only for the ambiguous score band (src/guardrails/topic_judge.py)
      -> Layer 3: hardened system prompt + chat model call       (src/prompt_builder.py, src/api_client.py)
      -> Layer 4: embedding topic relevance on the OUTPUT        (src/guardrails/topic_relevance.py)
      -> final response

Layers 1, 2 and 2b can each end the turn early with a refusal WITHOUT
calling the chat model for an answer -- this is both a safety property
(off-topic/malicious input never reaches the answering model) and a cost
property (most adversarial or off-topic probes cost one embedding call
or less, which shows up directly in the usage totals).

Layer 4 exists because layers 0-3 are not assumed to be perfect: it is
the safety net for cases where the model was still steered off-topic
despite everything upstream (e.g. slow multi-turn drift, or a jailbreak
pattern not covered by Layer 1's heuristics).

Failure policy: if the INPUT embedding check or the judge fails, the turn
fails closed (error message / refusal). If the OUTPUT embedding check
fails, the already-generated answer is shown (fail open) and the failure
is logged to errors.log and the audit record -- the input has already
passed Layers 1-2b, and hiding a vetted answer because of a transient
network error would hurt usability for little safety gain.
"""

from __future__ import annotations

import re
import time
import uuid

from src.api_client import ApiClient, ApiContentFilterError, ApiError
from src.audit_logger import AuditLogger, build_error_logger
from src.config import Settings
from src.context_manager import ContextManager
from src.guardrails import injection_detector, sanitizer, topic_judge
from src.guardrails.segmenter import split_chunks, split_segments, word_count
from src.guardrails.topic_relevance import TopicRelevanceChecker
from src.prompt_builder import build_system_prompt
from src.topic import Topic
from src.usage_monitor import UsageMonitor

GENERIC_ERROR_MESSAGE = (
    "I'm having trouble reaching the AI service right now. Please try again in a moment."
)

# Messages this short (in words) are often context-dependent follow-ups
# ("thanks!", "how often?") that can't be judged on their own.
SHORT_MESSAGE_WORDS = 8

# Creative-writing requests ("write a poem / story / song / speech about...")
# are sent to the judge whatever their embedding score: a single topic word
# in the request ("mention a carabiner once") lifts an off-topic poem just
# over the pass line, and only a reader of the whole sentence can tell a
# poem ABOUT the topic from a poem about something else wearing a topic word.
# Generic on purpose -- no topic words, no special case for any one form.
_CREATIVE_REQUEST = re.compile(
    r"\b(?:write|compose|create|draft|pen|make\s+up|come\s+up\s+with|give\s+me)\b.{0,60}?"
    r"\b(?:poem|sonnet|haiku|limerick|verse|rap|song|lyrics|story|tale|fable|essay|speech|toast|"
    r"script|screenplay|ode|ballad|joke|riddle|letter|monologue|dialogue)s?\b",
    re.IGNORECASE,
)
# Same idea for a switch of TASK TYPE: asking for a different kind of
# artefact (a recipe, code, an email, a translation, a business document)
# with some topic context around it. Found when the scope guidance made the
# model hand over a full pasta recipe framed as "carb loading" nutrition.
_TASK_SWITCH_REQUEST = re.compile(
    r"\b(?:write|compose|create|draft|give|send|make|need|want|show|build|code|translate|summari[sz]e)\b.{0,60}?"
    r"\b(?:recipes?|cover\s+letter|business\s+plan|e-?mails?|resume|cv|contract|invoice|spreadsheet|"
    r"python|javascript|sql|html|code|function|script|program|app|translation|summary\s+of\s+this)\b"
    r"|\btranslate\b",
    re.IGNORECASE,
)


class GuardedChatSession:
    def __init__(
        self,
        settings: Settings,
        topic: Topic,
        api_client: ApiClient,
        relevance_checker: TopicRelevanceChecker,
        usage_monitor: UsageMonitor,
        audit_logger: AuditLogger,
    ):
        self._settings = settings
        self._topic = topic
        self._api = api_client
        self._relevance = relevance_checker
        self._usage = usage_monitor
        self._audit = audit_logger
        self._errors = build_error_logger(settings.log_dir)
        self._context = ContextManager(settings.max_context_turns, settings.max_context_tokens)
        self._system_prompt = build_system_prompt(topic)
        self._session_id = str(uuid.uuid4())
        self._turn_index = 0

    @classmethod
    def create(cls, settings: Settings, topic: Topic) -> "GuardedChatSession":
        api_client = ApiClient(settings)
        threshold = settings.similarity_threshold_override or topic.similarity_threshold
        session_id = str(uuid.uuid4())
        usage_monitor = UsageMonitor(settings, session_id)
        audit_logger = AuditLogger(settings.log_dir)
        relevance_checker = TopicRelevanceChecker(
            api_client,
            topic,
            settings.topic_config_path.parent,
            threshold,
            on_embedding_call=usage_monitor.record_embedding,
            embedding_model=settings.embedding_deployment,
        )
        session = cls(settings, topic, api_client, relevance_checker, usage_monitor, audit_logger)
        session._session_id = session_id
        return session

    # ----------------------------------------------------------------
    def handle_message(self, raw_text: str) -> str:
        start = time.perf_counter()
        self._turn_index += 1
        layers: dict = {}

        # Layer 0 -------------------------------------------------------
        clean = sanitizer.sanitize(raw_text, self._settings.max_input_chars)
        layers["sanitizer"] = {
            "truncated": clean.was_truncated,
            "had_control_chars": clean.had_control_chars,
            "format_chars_removed": clean.format_chars_removed,
            "suspicious_encoding": clean.had_suspicious_encoding,
            "collapsed_scanned": clean.collapsed_text is not None,
            "base64_decoded": len(clean.decoded_segments),
            "base64_decision": "none",
        }

        # Layer 1: scan the cleaned text AND the extra views ------------
        injection, view = self._scan_all_views(clean, self._topic.persona_roles)
        if clean.decoded_segments:
            # Decoded text that is itself an injection -> refuse here.
            # Decoded text that is harmless -> only flagged; later layers decide.
            layers["sanitizer"]["base64_decision"] = (
                "refused" if view == "base64_decoded" else "flagged_only"
            )
        layers["injection_detector"] = {
            "matched": injection.matched, "rule": injection.matched_rule, "view": view,
            "soft_rules": list(injection.soft_rules),
        }
        if injection.matched:
            self._usage.record_skipped_by_guardrail(f"injection:{injection.matched_rule}")
            return self._finish(clean.text, "refused_input", layers, start,
                                self._topic.injection_refusal_message)

        # Layers 2 and 2b: is this on topic? -----------------------------
        try:
            on_topic = self._input_is_on_topic(clean.text, layers,
                                               force_judge=bool(injection.soft_rules))
        except ApiError as exc:
            self._errors.error("embedding call failed on input relevance check: %s", exc)
            return self._finish(clean.text, "error", layers, start, GENERIC_ERROR_MESSAGE,
                                error=str(exc))
        if not on_topic:
            reason = "off_topic_judge" if "topic_judge" in layers else "off_topic_input"
            self._usage.record_skipped_by_guardrail(reason)
            return self._finish(clean.text, "refused_input", layers, start,
                                self._topic.refusal_message)

        # Layer 3: hardened system prompt + model call -------------------
        try:
            chat_result = self._api.chat_complete(self._build_messages(clean.text))
        except ApiContentFilterError as exc:
            # The provider's own safety filter refused: an extra, external
            # guardrail. Show the normal refusal, not an "outage" message.
            self._errors.warning("chat call blocked by provider content filter: %s", exc)
            layers["provider_content_filter"] = {"blocked": True}
            return self._finish(clean.text, "refused_input", layers, start,
                                self._topic.refusal_message)
        except ApiError as exc:
            self._errors.error("chat completion failed: %s", exc)
            return self._finish(clean.text, "error", layers, start, GENERIC_ERROR_MESSAGE,
                                error=str(exc))

        self._usage.record_chat(chat_result.prompt_tokens, chat_result.completion_tokens)
        final_response = chat_result.content

        # Layer 4: re-check the model's own answer -----------------------
        if self._settings.check_output_topicality:
            try:
                final_response, refuse = self._check_output(final_response, layers)
            except ApiError as exc:
                # Fail OPEN, but never silently (see module docstring).
                self._errors.error(
                    "embedding call failed on output relevance check (failing open): %s", exc
                )
                layers["topic_relevance_output"] = {"error": str(exc), "decision": "fail_open"}
                refuse = False

            if refuse:
                self._errors.warning(
                    "output relevance check rejected a model response (%s) -- "
                    "possible upstream guardrail bypass",
                    layers["topic_relevance_output"].get("decision"),
                )
                # The refused user text is kept for conversational coherence,
                # but never in the recap (it did not pass the topic check).
                self._context.add("user", clean.text, summary_text="")
                self._context.add("assistant", self._topic.refusal_message)
                return self._finish(
                    clean.text, "refused_output", layers, start, self._topic.refusal_message
                )

        self._context.add("user", clean.text, summary_text=layers.get("_summary_text"))
        self._context.add("assistant", final_response)
        return self._finish(clean.text, "answered", layers, start, final_response)

    # ----------------------------------------------------------------
    def _check_output(self, reply: str, layers: dict) -> tuple[str, bool]:
        """Layer 4. Returns (reply to show, refuse?).

        The whole reply is scored (as before) AND, when the topic sets
        output_chunk_threshold, each paragraph-chunk is scored too -- all in
        ONE embedding call. Red-team finding: a one-vector check let a short
        off-topic tail ("bonus tip: brew kombucha...") ride on a long on-topic
        answer.

        Behaviour chosen for an off-topic chunk: REMOVE it and tell the user
        briefly, rather than refusing the whole reply -- the rest of the
        answer passed every layer and is what the user asked for, and a
        mixed request ("answer my climbing question, and add X") is exactly
        the case where the in-scope part should survive. If most of the reply
        (more than half its words) is off-topic, the whole reply is refused
        instead, because what would be left is not a real answer. Chunks
        shorter than output_chunk_min_words (sign-offs, "Happy climbing!")
        are never removed on their own: too little text to score reliably.
        """
        topic = self._topic
        chunks = split_chunks(reply) if topic.output_chunk_threshold is not None else []
        scored_chunks = chunks if len(chunks) > 1 else []
        results = self._relevance.score_many([reply, *scored_chunks])
        whole = results[0]
        record: dict = {"score": round(whole.score, 4), "passed": whole.passed}
        layers["topic_relevance_output"] = record
        if not whole.passed:
            record["decision"] = "refused_whole_reply"
            return reply, True
        if not scored_chunks or topic.output_chunk_threshold is None:
            return reply, False

        min_words = topic.output_chunk_min_words
        off = [
            i for i, (chunk, res) in enumerate(zip(scored_chunks, results[1:]))
            if word_count(chunk) >= min_words and res.score < topic.output_chunk_threshold
        ]
        record["chunks"] = len(scored_chunks)
        record["chunk_scores"] = [round(r.score, 4) for r in results[1:]]
        if not off:
            return reply, False
        removed_words = sum(word_count(scored_chunks[i]) for i in off)
        record["removed_chunks"] = off
        record["removed_word_share"] = round(removed_words / max(1, word_count(reply)), 3)
        if record["removed_word_share"] > 0.5:
            record["decision"] = "refused_mostly_off_topic"
            return reply, True
        record["decision"] = "removed_off_topic_chunks"
        kept = [c for i, c in enumerate(scored_chunks) if i not in off]
        return "\n\n".join(kept) + f"\n\n{topic.partial_removal_note}", False

    def _truncation_notice(self) -> str:
        return (f"(Note: your message was longer than {self._settings.max_input_chars} characters, "
                "so only the first part was read.)")

    # ----------------------------------------------------------------
    @staticmethod
    def _scan_all_views(clean: sanitizer.SanitizeResult, allowed_roles: tuple[str, ...] = ()):
        """Hard hit on any view -> refuse. Soft hits from every view are
        collected and returned so the caller can route the turn to the judge."""
        views = [("cleaned", clean.text)]
        if clean.collapsed_text:
            views.append(("collapsed", clean.collapsed_text))
        views.extend(("base64_decoded", segment) for segment in clean.decoded_segments)
        soft: list[str] = []
        for view, text in views:
            result = injection_detector.scan(text, allowed_roles)
            if result.matched:
                return result, view
            soft.extend(r for r in result.soft_rules if r not in soft)
        return injection_detector.InjectionResult(False, None, tuple(soft)), None

    def _input_is_on_topic(self, text: str, layers: dict, force_judge: bool = False) -> bool:
        """Layer 2 (embedding score, with contextual scoring for follow-ups),
        then Layer 2b (LLM judge) if the score lands in the ambiguous band."""
        topic = self._topic
        plain = self._relevance.score(text)
        score = plain.score
        record = {"score": round(plain.score, 4), "passed": plain.passed}
        if plain.centroid_score is not None:
            record["centroid_score"] = round(plain.centroid_score, 4)
            record["anchor_score"] = round(plain.anchor_score or 0.0, 4)

        # Contextual scoring (only reached by messages that passed Layer 1):
        # score the last two accepted user messages + this one together, so a
        # follow-up like "how often?" or "thanks!" isn't judged in isolation.
        # History can only make a message AMBIGUOUS (lift it into the judge
        # band, where the LLM judge decides with the recent turns as context)
        # -- it can never auto-pass one. Otherwise any short off-topic message
        # ("banana bread recipe?") would ride on an on-topic history.
        short = len(text.split()) <= SHORT_MESSAGE_WORDS
        previous = self._context.recent_user_messages(2)
        judge_low, judge_high = topic.judge_low, topic.judge_high  # None if the topic has no judge band
        if judge_high is not None and previous and (short or not plain.passed):
            context_score = self._relevance.score("\n".join([*previous, text])).score
            record["context_score"] = round(context_score, 4)
            score = max(score, min(context_score, judge_high - 1e-6))
            record["used_context"] = score > plain.score

        if judge_low is None or judge_high is None:
            record["passed"] = score >= self._relevance.threshold
            layers["topic_relevance_input"] = record
            return record["passed"]

        if score >= judge_high and force_judge:
            # Borderline Layer 1 wording (soft rule): never auto-pass; the
            # judge reads the whole sentence and decides.
            record["band"] = "judge"
            record["judge_reason"] = "soft_injection_wording"
        elif score >= judge_high:
            record["band"] = "pass"
            reason = self._needs_judge_despite_high_score(text, record)
            if reason:
                record["band"] = "judge"
                record["judge_reason"] = reason
        elif score < judge_low and short:
            # A very short message carries too little text for an embedding
            # to judge ("flappers, tape or not?"), so instead of refusing on
            # the score alone it goes to the judge, which still fails closed.
            record["band"] = "judge"
            record["judge_reason"] = "short_message"
        elif score < judge_low:
            record["band"] = "refuse"
        else:
            record["band"] = "judge"
        record["passed"] = record["band"] == "pass"
        if "_summary_text" in record:
            layers["_summary_text"] = record.pop("_summary_text")
        layers["topic_relevance_input"] = record
        if record["band"] != "judge":
            return record["passed"]

        verdict = topic_judge.judge(self._api, topic, text, self._context.recent_turns(2))
        if verdict.prompt_tokens or verdict.completion_tokens:
            self._usage.record_chat(verdict.prompt_tokens, verdict.completion_tokens,
                                    purpose="judge")
        if verdict.error:
            self._errors.error("topic judge failed closed: %s", verdict.error)
        layers["topic_judge"] = {
            "in_scope": verdict.in_scope, "reason": verdict.reason, "error": verdict.error,
        }
        return verdict.in_scope

    def _needs_judge_despite_high_score(self, text: str, record: dict) -> str | None:
        """Red-team round: a high WHOLE-message score can hide an off-topic
        part. Returns why the judge should decide, or None to auto-pass.

        * creative-writing and task-type-switch requests (see
          _CREATIVE_REQUEST, _TASK_SWITCH_REQUEST);
        * any sentence/clause scoring below judge_low on its own ("...plan,
          and at the very end add a bonus tip on brewing kombucha"). The
          segment score never REFUSES by itself -- a low-scoring clause can be
          harmless context ("I work night shifts") -- it only takes away the
          auto-pass. All segments are scored in one embedding call.
        The in-scope segments are also remembered as the text this turn may
        contribute to the recap (see ContextManager)."""
        if _CREATIVE_REQUEST.search(text):
            return "creative_writing"
        if _TASK_SWITCH_REQUEST.search(text):
            return "task_type_switch"
        segments = split_segments(text)
        if not segments:
            return None
        judge_low = self._topic.judge_low or 0.0
        results = self._relevance.score_many(segments)
        record["segment_scores"] = [round(r.score, 4) for r in results]
        kept = [seg for seg, r in zip(segments, results) if r.score >= judge_low]
        if len(kept) < len(segments):
            record["_summary_text"] = " ".join(kept)
            return "off_topic_segment"
        return None

    def _build_messages(self, user_text: str) -> list[dict]:
        history = self._context.get_messages()
        if history and history[-1]["role"] == "user":
            # Only happens if every kept turn was folded into the recap:
            # merge rather than send two user messages in a row.
            last = history.pop()
            user_text = f"{last['content']}\n\n{user_text}"
        return (
            [{"role": "system", "content": self._system_prompt}]
            + history
            + [{"role": "user", "content": user_text}]
        )

    def _finish(self, user_text, action, layers, start, response, error=None) -> str:
        latency_ms = (time.perf_counter() - start) * 1000
        layers.pop("_summary_text", None)  # internal hand-off, not audit data
        if layers.get("sanitizer", {}).get("truncated"):
            # Tell the user, on every outcome, that only part of a long
            # message was read (red-team round: a long message's real
            # question past the cut-off was silently dropped).
            response = f"{self._truncation_notice()}\n\n{response}"
        self._audit.log_turn(
            session_id=self._session_id,
            turn_index=self._turn_index,
            user_text=user_text,
            action=action,
            layers=layers,
            latency_ms=latency_ms,
            error=error,
        )
        return response

    def usage_summary(self) -> str:
        return self._usage.summary()
