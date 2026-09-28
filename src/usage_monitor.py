"""
Tracks API usage across the session: number of calls, tokens in/out, and
an estimated cost. Every call is also appended to logs/usage.jsonl so
usage can be audited/analysed after the fact (operational oversight,
per the assignment's "monitor the usage of the AI service" requirement),
without needing to keep the process running.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Settings


@dataclass
class UsageTotals:
    chat_calls: int = 0
    judge_calls: int = 0  # subset of chat_calls made by the Layer 2b topic judge
    embedding_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    calls_skipped_by_guardrails: int = 0  # chat calls avoided by early refusal

    def estimated_cost(self, settings: Settings) -> float:
        return (
            self.prompt_tokens / 1000 * settings.chat_cost_per_1k_prompt
            + self.completion_tokens / 1000 * settings.chat_cost_per_1k_completion
            + self.embedding_tokens / 1000 * settings.embedding_cost_per_1k
        )


class UsageMonitor:
    def __init__(self, settings: Settings, session_id: str):
        self._settings = settings
        self._session_id = session_id
        self._totals = UsageTotals()
        self._log_path = settings.log_dir / "usage.jsonl"
        settings.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def totals(self) -> UsageTotals:
        return self._totals

    def record_chat(self, prompt_tokens: int, completion_tokens: int, purpose: str = "chat") -> None:
        """purpose is "chat" for the answer itself or "judge" for a Layer 2b
        topic-judge classification call; both are billed chat completions."""
        self._totals.chat_calls += 1
        if purpose == "judge":
            self._totals.judge_calls += 1
        self._totals.prompt_tokens += prompt_tokens
        self._totals.completion_tokens += completion_tokens
        self._write({
            "type": purpose,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def record_embedding(self, tokens: int) -> None:
        self._totals.embedding_calls += 1
        self._totals.embedding_tokens += tokens
        self._write({"type": "embedding", "tokens": tokens})

    def record_skipped_by_guardrail(self, reason: str) -> None:
        self._totals.calls_skipped_by_guardrails += 1
        self._write({"type": "chat_call_avoided", "reason": reason})

    def summary(self) -> str:
        t = self._totals
        cost = t.estimated_cost(self._settings)
        return (
            f"chat calls: {t.chat_calls} (of which topic-judge: {t.judge_calls}) | embedding calls: {t.embedding_calls} | "
            f"prompt tokens: {t.prompt_tokens} | completion tokens: {t.completion_tokens} | "
            f"embedding tokens: {t.embedding_tokens} | "
            f"chat calls avoided by guardrails: {t.calls_skipped_by_guardrails} | "
            f"estimated cost: ${cost:.5f} (illustrative rates, see app_config.json)"
        )

    def _write(self, payload: dict) -> None:
        record = {"timestamp": time.time(), "session_id": self._session_id, **payload}
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
