"""
GuardedChatSession orchestrates one full turn end-to-end:

    raw input
      -> Layer 0: sanitize                              (src/guardrails/sanitizer.py)
      -> Layer 1: injection/jailbreak scan               (src/guardrails/injection_detector.py)
      -> Layer 2: topic relevance on the INPUT            (src/guardrails/topic_relevance.py)
      -> Layer 3: hardened system prompt + chat model call (src/prompt_builder.py, src/api_client.py)
      -> Layer 4: topic relevance on the OUTPUT           (src/guardrails/topic_relevance.py)
      -> final response

Layers 1 and 2 can each end the turn early with a refusal WITHOUT calling
the chat model at all -- this is both a safety property (off-topic/
malicious input never reaches the model) and a cost property (the
majority of adversarial or off-topic probes never generate a
chat-completion call, which shows up directly in the usage totals).

Layer 4 exists because layers 0-3 are not assumed to be perfect: it is
the safety net for cases where the model was still steered off-topic
despite everything upstream (e.g. slow multi-turn drift, or a jailbreak
pattern not covered by Layer 1's heuristics).
"""

from __future__ import annotations

import time
import uuid

from src.api_client import ApiClient, ApiError
from src.audit_logger import AuditLogger, build_error_logger
from src.config import Settings
from src.context_manager import ContextManager
from src.guardrails import injection_detector, sanitizer
from src.guardrails.topic_relevance import TopicRelevanceChecker
from src.prompt_builder import build_system_prompt
from src.topic import Topic
from src.usage_monitor import UsageMonitor

GENERIC_ERROR_MESSAGE = (
    "I'm having trouble reaching the AI service right now. Please try again in a moment."
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
        )
        session = cls(settings, topic, api_client, relevance_checker, usage_monitor, audit_logger)
        session._session_id = session_id
        return session

    # ----------------------------------------------------------------
    def handle_message(self, raw_text: str) -> str:
        start = time.perf_counter()
        self._turn_index += 1
        layers: dict = {}

        clean = sanitizer.sanitize(raw_text, self._settings.max_input_chars)
        layers["sanitizer"] = {
            "truncated": clean.was_truncated,
            "had_control_chars": clean.had_control_chars,
            "suspicious_encoding": clean.had_suspicious_encoding,
        }

        injection = injection_detector.scan(clean.text)
        layers["injection_detector"] = {"matched": injection.matched, "rule": injection.matched_rule}
        if injection.matched:
            self._usage.record_skipped_by_guardrail(f"injection:{injection.matched_rule}")
            return self._finish(clean.text, "refused_input", layers, start,
                                 self._topic.injection_refusal_message)

        try:
            # Token usage for this call is recorded automatically via the
            # on_embedding_call callback wired into the relevance checker
            # (see GuardedChatSession.create), so it stays accurate even
            # though centroid-building embeds happen inside that class.
            input_relevance = self._relevance.score(clean.text)
        except ApiError as exc:
            self._errors.error("embedding call failed on input relevance check: %s", exc)
            return self._finish(clean.text, "error", layers, start, GENERIC_ERROR_MESSAGE, error=str(exc))

        layers["topic_relevance_input"] = {
            "score": round(input_relevance.score, 4),
            "passed": input_relevance.passed,
        }
        if not input_relevance.passed:
            self._usage.record_skipped_by_guardrail("off_topic_input")
            return self._finish(clean.text, "refused_input", layers, start, self._topic.refusal_message)

        messages = (
            [{"role": "system", "content": self._system_prompt}]
            + self._context.get_messages()
            + [{"role": "user", "content": clean.text}]
        )

        try:
            chat_result = self._api.chat_complete(messages)
        except ApiError as exc:
            self._errors.error("chat completion failed: %s", exc)
            return self._finish(clean.text, "error", layers, start, GENERIC_ERROR_MESSAGE, error=str(exc))

        self._usage.record_chat(chat_result.prompt_tokens, chat_result.completion_tokens)

        final_response = chat_result.content
        if self._settings.check_output_topicality:
            try:
                output_relevance = self._relevance.score(final_response)
            except ApiError as exc:
                self._errors.error("embedding call failed on output relevance check: %s", exc)
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
