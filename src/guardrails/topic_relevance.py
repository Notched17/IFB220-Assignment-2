"""
Guardrail Layers 2 (input) & 4 (output) -- semantic topic relevance,
using embeddings (text-embedding-3-small on the IFB220 portal; the brief
names Ada-002, which the portal returns 404 for -- both are 1536-dim
OpenAI embedding models and the deployment is configurable).

Why embeddings instead of a keyword list: a keyword/regex allow-list is
trivially defeated by paraphrase ("what should I do about the thing on
my fingers after gripping small holds all day?" contains no climbing
keyword but is clearly in-scope; conversely a user can wrap an off-topic
request in climbing-flavoured words). Comparing the MEANING of the input
to a centroid built from representative on-topic sentences is far more
robust to paraphrase in both directions, and it is exactly the kind of
task embedding models are suited to -- which is also why the assignment
requires an embedding-based component in the first place.

This module is used TWICE in the pipeline:
  * on the user's input, BEFORE calling the chat model at all (cheapest
    possible rejection of off-topic requests -- saves a chat-completion
    call entirely, which matters for both cost and the "monitor usage"
    requirement)
  * on the model's own response, AFTER generation, as a second
    independent check. This catches slow multi-turn topic drift or a
    successful jailbreak that got past Layers 0-1 and the system prompt:
    even if the model was talked into answering something off-topic, the
    reply itself won't score as topically similar to the centroid, so it
    gets swapped for the refusal message instead of being shown to the
    user.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import re

from src.api_client import ApiClient
from src.config import ensure_writable_dir
from src.topic import Topic

_CACHE_DIRNAME = "cache"


@dataclass
class RelevanceResult:
    score: float
    passed: bool


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


class TopicRelevanceChecker:
    """Builds (and disk-caches) a topic centroid, then scores arbitrary
    text against it. One instance is created per Topic at startup."""

    def __init__(
        self,
        api_client: ApiClient,
        topic: Topic,
        topics_dir: Path,
        threshold: float,
        on_embedding_call: Callable[[int], None] | None = None,
        embedding_model: str = "",
    ):
        self._api_client = api_client
        self._topic = topic
        self._threshold = threshold
        self._embedding_model = embedding_model
        # The model name is part of the cache file name AND the stored hash,
        # so switching embedding model can never reuse a stale centroid.
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", embedding_model) or "default"
        self._cache_path = (
            topics_dir / _CACHE_DIRNAME / f"{topic.topic_id}__{safe_model}_centroid.json"
        )
        self._centroid: list[float] | None = None
        # Optional hook so the caller (the pipeline) can record accurate
        # embedding token usage for every embed() call this checker makes,
        # including the one-off centroid-building calls -- without this
        # class needing to know anything about UsageMonitor itself.
        self._on_embedding_call = on_embedding_call or (lambda tokens: None)

    def centroid(self) -> list[float]:
        if self._centroid is not None:
            return self._centroid

        current_hash = self._topic.anchor_phrases_hash(self._embedding_model)
        if self._cache_path.exists():
            try:
                cached = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = {}  # corrupt/unreadable cache: just rebuild it
            if cached.get("anchor_hash") == current_hash:
                self._centroid = cached["centroid"]
                return self._centroid

        # No valid cache: compute fresh from the topic's anchor phrases.
        # This costs len(anchor_phrases) embedding calls, but only happens
        # once per topic (result is cached to disk), which keeps ongoing
        # per-turn cost to a single embedding call.
        vectors = []
        for phrase in self._topic.anchor_phrases:
            result = self._api_client.embed(phrase)
            self._on_embedding_call(result.total_tokens)
            vectors.append(result.vector)
        centroid = _mean_vector(vectors)

        cache_dir = ensure_writable_dir(self._cache_path.parent, "topic_cache")
        self._cache_path = cache_dir / self._cache_path.name
        self._cache_path.write_text(
            json.dumps({
                "anchor_hash": current_hash,
                "embedding_model": self._embedding_model,
                "centroid": centroid,
            }),
            encoding="utf-8",
        )
        self._centroid = centroid
        return centroid

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(self, text: str) -> RelevanceResult:
        embedding = self._api_client.embed(text)
        self._on_embedding_call(embedding.total_tokens)
        similarity = _cosine(embedding.vector, self.centroid())
        return RelevanceResult(score=similarity, passed=similarity >= self._threshold)
