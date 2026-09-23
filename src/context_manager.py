"""
Manages conversation history so the context sent to the model:
  (a) stays under a token budget (prevents context-window overflow and
      unnecessary token spend), and
  (b) still lets the assistant hold a coherent multi-turn conversation.

Strategy: keep the most recent N turns verbatim (so recent back-and-forth
stays crisp), and fold anything older than that into a single rolling
summary line that is prepended to the context. The summariser is
pluggable: a fast, free, deterministic heuristic summariser is used by
default (good for predictable cost/testing), and an optional LLM-based
summariser can be switched on via SUMMARIZE_WITH_LLM=true for higher
quality summaries at the cost of an extra API call when the window
rolls over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


_tiktoken_encoder = None
_tiktoken_unavailable = False


def estimate_tokens(text: str) -> int:
    """Best-effort token estimate. Uses tiktoken if it's installed AND its
    encoding table is available, otherwise falls back to a ~4-chars-per-
    token heuristic (close enough to size a conversation window safely).

    NOTE: tiktoken's first call downloads its BPE merge table from a
    Microsoft-hosted URL rather than shipping it in the package. On a
    locked-down network (a uni lab, a CI runner, a sandboxed grading
    environment) that download can fail even though tiktoken itself
    imported fine -- this was caught during testing (see docs/TESTING.md)
    where the whole context manager crashed with a requests.HTTPError
    despite the ImportError guard. We now cache success/failure at module
    load and fall back to the heuristic on ANY failure, not just
    ImportError, so a blocked network degrades gracefully instead of
    crashing the assistant."""
    global _tiktoken_encoder, _tiktoken_unavailable

    if not _tiktoken_unavailable:
        if _tiktoken_encoder is None:
            try:
                import tiktoken

                _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                _tiktoken_unavailable = True
        if _tiktoken_encoder is not None:
            return len(_tiktoken_encoder.encode(text))

    return max(1, len(text) // 4)


class Summarizer(Protocol):
    def __call__(self, turns: list[Turn]) -> str: ...


def heuristic_summarizer(turns: list[Turn]) -> str:
    """Cheap, deterministic, no-API-call summary: one clause per user
    turn being folded away. Not as fluent as an LLM summary, but zero
    cost, zero latency, and fully deterministic (easy to unit test)."""
    user_points = [t.content.strip().rstrip(".!?") for t in turns if t.role == "user"]
    if not user_points:
        return ""
    joined = "; ".join(user_points[-5:])  # cap so the summary itself can't grow unbounded
    return f"Earlier in this conversation, the user also asked about: {joined}."


class ContextManager:
    def __init__(
        self,
        max_turns: int,
        max_tokens: int,
        summarizer: Summarizer = heuristic_summarizer,
    ):
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._summarizer = summarizer
        self._turns: list[Turn] = []
        self._rolling_summary: str = ""

    def add(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        self._enforce_limits()

    def _enforce_limits(self) -> None:
        # Turn-count based rollover.
        while len(self._turns) > self._max_turns:
            oldest = self._turns.pop(0)
            self._fold_into_summary([oldest])

        # Token-budget based rollover (handles unusually long individual
        # turns that turn-count alone wouldn't catch).
        while self._turns and self._total_tokens() > self._max_tokens:
            oldest = self._turns.pop(0)
            self._fold_into_summary([oldest])

    def _fold_into_summary(self, folded: list[Turn]) -> None:
        self._rolling_summary = self._summarizer(
            [Turn(role="user", content=self._rolling_summary)] + folded
            if self._rolling_summary
            else folded
        )

    def _total_tokens(self) -> int:
        return sum(estimate_tokens(t.content) for t in self._turns) + estimate_tokens(
            self._rolling_summary
        )

    def get_messages(self) -> list[dict]:
        """Returns the context portion (summary + recent turns) as
        chat-message dicts, WITHOUT the system prompt (the pipeline adds
        that separately, since it's topic-derived, not conversation
        state)."""
        messages: list[dict] = []
        if self._rolling_summary:
            messages.append({"role": "system", "content": self._rolling_summary})
        messages.extend({"role": t.role, "content": t.content} for t in self._turns)
        return messages

    def reset(self) -> None:
        self._turns.clear()
        self._rolling_summary = ""
