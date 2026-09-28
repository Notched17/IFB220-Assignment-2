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

import time
import uuid

from src.api_client import ApiClient, ApiContentFilterError, ApiError
from src.audit_logger import AuditLogger, build_error_logger
from src.config import Settings
from src.context_manager import ContextManager
from src.guardrails import injection_detector, sanitizer, topic_judge
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
        injection, view = self._scan_all_views(clean)
        if clean.decoded_segments:
            # Decoded text that is itself an injection -> refuse here.
            # Decoded text that is harmless -> only flagged; later layers decide.
            layers["sanitizer"]["base64_decision"] = (
                "refused" if view == "base64_decoded" else "flagged_only"
            )
        layers["injection_detector"] = {
            "matched": injection.matched, "rule": injection.matched_rule, "view": view,
        }
        if injection.matched:
            self._usage.record_skipped_by_guardrail(f"injection:{injection.matched_rule}")
            return self._finish(clean.text, "refused_input", layers, start,
                                self._topic.injection_refusal_message)

        # Layers 2 and 2b: is this on topic? -----------------------------
        try:
            on_topic = self._input_is_on_topic(clean.text, layers)
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
                output_relevance = self._relevance.score(final_response)
            except ApiError as exc:
                # Fail OPEN, but never silently (see module docstring).
                self._errors.error(
                    "embedding call failed on output relevance check (failing open): %s", exc
                )
                layers["topic_relevance_output"] = {"error": str(exc), "decision": "fail_open"}
                output_relevance = None

            if output_relevance is not None:
                layers["topic_relevance_output"] = {
                    "score": round(output_relevance.score, 4),
                    "passed": output_relevance.passed,
                }
                if not output_relevance.passed:
                    self._errors.warning(
                        "output relevance check rejected a model response (score=%.4f) -- "
                        "possible upstream guardrail bypass",
                        output_relevance.score,
                    )
                    self._context.add("user", clean.text)
                    self._context.add("assistant", self._topic.refusal_message)
                    return self._finish(
                        clean.text, "refused_output", layers, start, self._topic.refusal_message
                    )

        self._context.add("user", clean.text)
        self._context.add("assistant", final_response)
        return self._finish(clean.text, "answered", layers, start, final_response)

    # ----------------------------------------------------------------
    @staticmethod
    def _scan_all_views(clean: sanitizer.SanitizeResult):
        views = [("cleaned", clean.text)]
        if clean.collapsed_text:
            views.append(("collapsed", clean.collapsed_text))
        views.extend(("base64_decoded", segment) for segment in clean.decoded_segments)
        for view, text in views:
            result = injection_detector.scan(text)
            if result.matched:
                return result, view
        return injection_detector.InjectionResult(matched=False, matched_rule=None), None

    def _input_is_on_topic(self, text: str, layers: dict) -> bool:
        """Layer 2 (embedding score, with contextual scoring for follow-ups),
        then Layer 2b (LLM judge) if the score lands in the ambiguous band."""
        topic = self._topic
        plain = self._relevance.score(text)
        score = plain.score
        record = {"score": round(plain.score, 4), "passed": plain.passed}

        # Contextual scoring (only reached by messages that passed Layer 1):
        # score the last two accepted user messages + this one together, so a
        # follow-up like "how often?" or "thanks!" isn't judged in isolation.
        # History can only make a message AMBIGUOUS (lift it into the judge
        # band, where the LLM judge decides with the recent turns as context)
        # -- it can never auto-pass one. Otherwise any short off-topic message
        # ("banana bread recipe?") would ride on an on-topic history.
        short = len(text.split()) <= SHORT_MESSAGE_WORDS
        previous = self._context.recent_user_messages(2)
        if topic.uses_judge and previous and (short or not plain.passed):
            context_score = self._relevance.score("\n".join([*previous, text])).score
            record["context_score"] = round(context_score, 4)
            score = max(score, min(context_score, topic.judge_high - 1e-6))
            record["used_context"] = score > plain.score

        if not topic.uses_judge:
            record["passed"] = score >= self._relevance.threshold
            layers["topic_relevance_input"] = record
            return record["passed"]

        if score >= topic.judge_high:
            record["band"] = "pass"
        elif score < topic.judge_low:
            record["band"] = "refuse"
        else:
            record["band"] = "judge"
        record["passed"] = record["band"] == "pass"
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
