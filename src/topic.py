"""
Loads a topic definition from a JSON file under topics/.

Changing the assistant's subject matter (e.g. from climbing to motor
vehicles) is done ENTIRELY by pointing TOPIC_CONFIG at a different file
here -- no application code changes. See topics/climbing.json for the
schema, and topics/motor_vehicles.json / topics/cinematography.json for
worked examples of a full topic swap.
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
        )

    def anchor_phrases_hash(self) -> str:
        """Used to invalidate a cached centroid if the anchor phrases change."""
        import hashlib

        joined = "\n".join(self.anchor_phrases).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()[:16]
