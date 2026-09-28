"""
Loads a topic definition from a JSON file under topics/.

Changing the assistant's subject matter (e.g. from climbing to motor
vehicles) is done ENTIRELY by pointing "topic_config" in app_config.json
at a different file here (or by editing the topic JSON itself) -- no
application code changes. See topics/climbing.json for the schema, and
topics/motor_vehicles.json / topics/cinematography.json for worked
examples of a full topic swap.

Optional fields judge_low / judge_high define the "ambiguous" band of
embedding scores that is sent to the Layer 2b LLM topic judge (see
src/guardrails/topic_judge.py). Without them, similarity_threshold alone
decides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Topic:
    topic_id: str
    display_name: str
    persona_description: str
    in_scope_summary: str
    out_of_scope_note: str
    refusal_message: str
    injection_refusal_message: str
    similarity_threshold: float
    anchor_phrases: tuple[str, ...]
    judge_low: float | None = None
    judge_high: float | None = None

    @property
    def uses_judge(self) -> bool:
        return self.judge_low is not None and self.judge_high is not None

    @staticmethod
    def load(path: Path) -> "Topic":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        required = {
            "topic_id",
            "display_name",
            "persona_description",
            "in_scope_summary",
            "out_of_scope_note",
            "refusal_message",
            "injection_refusal_message",
            "similarity_threshold",
            "anchor_phrases",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Topic config {path} is missing required field(s): {missing}")
        if not data["anchor_phrases"]:
            raise ValueError(f"Topic config {path} must include at least one anchor phrase")
        judge_low, judge_high = data.get("judge_low"), data.get("judge_high")
        if (judge_low is None) != (judge_high is None):
            raise ValueError(f"Topic config {path}: set both judge_low and judge_high, or neither")
        if judge_low is not None and not float(judge_low) <= float(judge_high):
            raise ValueError(f"Topic config {path}: judge_low must be <= judge_high")

        return Topic(
            topic_id=data["topic_id"],
            display_name=data["display_name"],
            persona_description=data["persona_description"],
            in_scope_summary=data["in_scope_summary"],
            out_of_scope_note=data["out_of_scope_note"],
            refusal_message=data["refusal_message"],
            injection_refusal_message=data["injection_refusal_message"],
            similarity_threshold=float(data["similarity_threshold"]),
            anchor_phrases=tuple(data["anchor_phrases"]),
            judge_low=None if judge_low is None else float(judge_low),
            judge_high=None if judge_high is None else float(judge_high),
        )

    def anchor_phrases_hash(self, embedding_model: str = "") -> str:
        """Used to invalidate a cached centroid if the anchor phrases OR the
        embedding model change (vectors from different models are not
        comparable)."""
        import hashlib

        joined = "\n".join((embedding_model, *self.anchor_phrases)).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()[:16]
