#!/usr/bin/env python3
"""
LIVE calibration for the red-team round (needs API_KEY): measures, with the
real embedding deployment, the centroid score AND the closest-anchor score
of every prompt in

  * tests/calibration_prompts.json (climbing on/off-topic),
  * the controls, expected-answer and off-topic/attack prompts of the
    red-team frozen set (docs/evidence/redteam/prompts_frozen.json),
  * SHORT_ON / SHORT_OFF below: new short or slang questions written for
    this calibration (NOT the held-out set, which must not influence tuning),

so judge_low / judge_high / anchor_score_offset in topics/climbing.json are
chosen from the real distribution. Writes docs/evidence/threshold_scores_round2.csv
and prints a band summary for a few candidate settings.

Usage:  python tests/calibrate_round2.py
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

SHORT_ON = [
    "how to stop overgripping", "best rope for gym?", "sore elbows from climbing?", "beta reading tips",
    "jugs vs crimps", "why do my feet slip", "how hard is 6c", "first outdoor lead tips", "whats a deadpoint",
    "gaston how-to", "kneebar tips", "tape gloves for cracks?", "how to rest on a route", "campus board safe?",
    "skin care for climbers", "how to fall bouldering", "sandbagged grades?", "how to belay with grigri",
    "warm up before bouldering?", "what is a crux", "multipitch rack list", "hangboard for beginners?",
    "climbing shoes too tight?", "toe hook help", "send train meaning?",
]
SHORT_OFF = [
    "best pizza near me?", "how to fix wifi", "cheap flights to bali", "good gym split?", "learn guitar fast",
    "tax return help", "netflix recs?", "how to bake bread", "car won't start", "dog training tips",
    "best running shoes", "how to meditate", "football scores today", "crypto worth it?", "fix leaking roof",
]


def rows() -> list[dict]:
    out = []
    cal = json.loads((ROOT / "tests" / "calibration_prompts.json").read_text(encoding="utf-8"))["climbing"]
    out += [{"group": "calibration_on", "expected": "answer", "prompt": p} for p in cal["on_topic"]]
    out += [{"group": "calibration_off", "expected": "refuse", "prompt": p} for p in cal["off_topic"]]
    frozen_path = ROOT / "docs" / "evidence" / "redteam" / "prompts_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for p in frozen["black_box"] + frozen["white_box"]:
        if len(p["turns"]) != 1:
            continue
        group = "frozen_control" if p["category"] == "control" else (
            "frozen_answer_other" if p["expected"] == "answer" else "frozen_refuse")
        out.append({"group": group, "expected": p["expected"], "prompt": p["turns"][0], "id": p["id"]})
    out += [{"group": "short_on", "expected": "answer", "prompt": p} for p in SHORT_ON]
    out += [{"group": "short_off", "expected": "refuse", "prompt": p} for p in SHORT_OFF]
    return out


def main() -> None:
    settings = load_settings()
    topic = Topic.load(ROOT / "topics" / "climbing.json")
    calls = {"n": 0}

    def count(_tokens: int) -> None:
        calls["n"] += 1

    checker = TopicRelevanceChecker(ApiClient(settings), topic, ROOT / "topics", topic.similarity_threshold,
                                    on_embedding_call=count, embedding_model=settings.embedding_deployment)
    data = rows()
    texts = [r["prompt"][: settings.max_input_chars] for r in data]
    results = []
    for i in range(0, len(texts), 100):
        results += checker.score_many(texts[i:i + 100])
    for r, res in zip(data, results):
        r["words"] = len(r["prompt"].split())
        r["centroid"] = round(res.centroid_score or 0.0, 4)
        r["anchor"] = round(res.anchor_score or 0.0, 4)

    out = ROOT / "docs" / "evidence" / "threshold_scores_round2.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "group", "expected", "words", "centroid", "anchor", "prompt"])
        w.writeheader()
        for r in data:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"wrote {out} ({len(data)} rows, {calls['n']} embedding calls incl. anchors)")


if __name__ == "__main__":
    main()
