#!/usr/bin/env python3
"""
LIVE calibration for chunked Layer 4 (red-team round; needs API_KEY).

Scores every paragraph-chunk (src/guardrails/segmenter.split_chunks) of
  * the ON-TOPIC replies the red-team frozen run produced for expected-answer
    prompts (docs/evidence/redteam/replies_frozen_run1.json), and
  * OFF-TOPIC paragraphs: the ones that leaked in the frozen run plus the
    synthetic paragraphs below,
with the real embedding deployment, so the chunk threshold and minimum chunk
length are chosen from data. The held-out set is NOT used. Writes
docs/evidence/layer4_chunk_scores.csv.

Usage:  python tests/calibrate_layer4_chunks.py
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
from src.guardrails.segmenter import split_chunks, word_count  # noqa: E402
from src.guardrails.topic_relevance import TopicRelevanceChecker  # noqa: E402
from src.topic import Topic  # noqa: E402

SYNTHETIC_OFF = [
    "To knit a basic scarf, cast on 30 stitches with chunky wool and 8mm needles, then knit every row in garter stitch until it is about 150cm long. Cast off loosely and weave in the ends with a darning needle.",
    "Tomorrow's forecast for the city is a sunny morning with a high of 24 degrees, turning cloudy in the afternoon with a 40 percent chance of showers after 4pm and light westerly winds.",
    "For your tax return, you can usually claim work-related car expenses, home office running costs at a fixed rate per hour, and union fees. Keep receipts for anything over a few hundred dollars.",
    "Tomatoes grow best in full sun with well-drained soil. Water deeply twice a week, stake the plants early, and pinch out the side shoots so the plant puts its energy into fruit.",
    "The Sicilian Defence starts with 1.e4 c5. Black fights for the centre asymmetrically, and the Najdorf variation with 5...a6 is one of the most popular and deeply analysed lines at top level.",
    "Last night the home side won 3-1 after a second-half double from their striker, moving them up to third on the ladder with two rounds left before the finals.",
    "A good mortgage broker will compare loans from many lenders, explain fixed versus variable rates, and help you understand comparison rates, fees and offset accounts before you sign.",
    "Fun fact: the Great Wall of China is not visible from space with the naked eye, despite the popular myth; astronauts have said it is far too narrow to pick out from orbit.",
    "Bonus recipe: fry an onion and garlic, add a tin of tomatoes and a pinch of chilli, simmer for 15 minutes, then stir through cooked pasta and finish with parmesan and basil.",
    "To reset your home router, hold the reset button for 30 seconds with the power on, wait for the lights to settle, then reconnect using the default network name printed on the label.",
]


def main() -> None:
    settings = load_settings()
    topic = Topic.load(ROOT / "topics" / "climbing.json")
    checker = TopicRelevanceChecker(ApiClient(settings), topic, ROOT / "topics", topic.similarity_threshold,
                                    embedding_model=settings.embedding_deployment)
    ev = ROOT / "docs" / "evidence" / "redteam"
    replies = json.loads((ev / "replies_frozen_run1.json").read_text(encoding="utf-8"))
    expected = {p["id"]: p["expected"] for k in ("black_box", "white_box")
                for p in json.loads((ev / "prompts_frozen.json").read_text())[k]}
    refusal = topic.refusal_message
    rows = []
    for r in replies:
        if expected.get(r["id"]) == "answer" and r["reply"].strip() and refusal[:40] not in r["reply"]:
            rows += [{"source": f"{r['id']}_t{r['turn']}", "label": "on_topic", "chunk": c} for c in split_chunks(r["reply"])]
    leaked = {"wb7": "ESTA", "wb8": "kombucha", "wb36": "Rome"}
    for r in replies:
        if r["id"] in leaked:
            rows += [{"source": f"{r['id']}_t{r['turn']}", "label": "off_topic_leak", "chunk": c}
                     for c in split_chunks(r["reply"]) if leaked[r["id"]].lower() in c.lower()]
    rows += [{"source": f"synthetic_{i}", "label": "off_topic_synthetic", "chunk": c} for i, c in enumerate(SYNTHETIC_OFF, 1)]
    results = []
    for i in range(0, len(rows), 100):
        results += checker.score_many([r["chunk"] for r in rows[i:i + 100]])
    out = ROOT / "docs" / "evidence" / "layer4_chunk_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "label", "words", "score", "centroid", "anchor", "chunk"])
        w.writeheader()
        for r, res in zip(rows, results):
            w.writerow({"source": r["source"], "label": r["label"], "words": word_count(r["chunk"]),
                        "score": round(res.score, 4), "centroid": round(res.centroid_score or 0, 4),
                        "anchor": round(res.anchor_score or 0, 4), "chunk": r["chunk"][:300]})
    print(f"wrote {out} ({len(rows)} chunks)")


if __name__ == "__main__":
    main()
