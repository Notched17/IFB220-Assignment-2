#!/usr/bin/env python3
"""
LIVE calibration helper: embeds prompts with the real embedding deployment
and prints/saves their cosine similarity to a topic's centroid, so the
topic's similarity_threshold / judge_low / judge_high can be chosen from
the real score distribution rather than guessed.

Usage (from any folder; needs API_KEY):
    python tests/score_topic_prompts.py climbing
    python tests/score_topic_prompts.py motor_vehicles

For "climbing" it scores every prompt in tests/adversarial_prompts.json
(single-turn, every multi-turn turn, and each multi-turn turn combined
with the previous user turn, as contextual scoring would) plus the
calibration prompts in tests/calibration_prompts.json. Other topics use
only their calibration prompts. Output: docs/evidence/threshold_scores*.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api_client import ApiClient  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.guardrails.topic_relevance import TopicRelevanceChecker  # noqa: E402
from src.topic import Topic  # noqa: E402


def rows_for(topic_id: str) -> list[dict]:
    rows = []
    calib = json.loads((ROOT / "tests" / "calibration_prompts.json").read_text(encoding="utf-8"))
    for group in ("on_topic", "off_topic"):
        for i, prompt in enumerate(calib[topic_id][group], 1):
            rows.append({"id": f"cal_{group}_{i}", "group": f"calibration_{group}",
                         "expected": "answer" if group == "on_topic" else "refuse", "prompt": prompt})
    if topic_id == "climbing":
        adv = json.loads((ROOT / "tests" / "adversarial_prompts.json").read_text(encoding="utf-8"))
        for item in adv["single_turn"]:
            rows.append({"id": item["id"], "group": item["category"],
                         "expected": item["expected"], "prompt": item["prompt"]})
        for sc in adv["multi_turn_scenarios"]:
            for n, (turn, exp) in enumerate(zip(sc["turns"], sc["expected"]), 1):
                rows.append({"id": f"{sc['id']}_t{n}", "group": sc["category"],
                             "expected": exp, "prompt": turn,
                             "previous": sc["turns"][n - 2] if n > 1 else None})
    return rows


def main() -> None:
    topic_id = sys.argv[1] if len(sys.argv) > 1 else "climbing"
    settings = load_settings()
    topic = Topic.load(ROOT / "topics" / f"{topic_id}.json")
    checker = TopicRelevanceChecker(ApiClient(settings), topic, ROOT / "topics",
                                    topic.similarity_threshold,
                                    embedding_model=settings.embedding_deployment)
    rows = rows_for(topic_id)
    for row in rows:
        row["score"] = round(checker.score(row["prompt"]).score, 4)
        prev = row.pop("previous", None)
        row["context_score"] = (round(checker.score(f"{prev}\n{row['prompt']}").score, 4)
                                if prev else "")
        row["words"] = len(row["prompt"].split())

    suffix = "" if topic_id == "climbing" else f"_{topic_id}"
    out = ROOT / "docs" / "evidence" / f"threshold_scores{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "group", "expected", "score", "context_score", "words", "prompt"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["score"]):
            writer.writerow({**row, "prompt": row["prompt"][:200]})
    for row in sorted(rows, key=lambda r: r["score"]):
        print(f"{row['score']:.4f} {str(row['context_score']):>7} {row['expected']:6} "
              f"{row['id']:22} {row['prompt'][:70]!r}")
    print(f"\nembedding model: {settings.embedding_deployment} | saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
