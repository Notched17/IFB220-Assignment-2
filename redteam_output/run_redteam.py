"""Live red-team runner for the climbing coach.

Uses the project's real code path: load_settings() + Topic.load(settings.topic_config_path)
+ GuardedChatSession.create(...). A FRESH session is created for every prompt/scenario run.
Instrumentation only (no behaviour change):
  * ApiClient._with_retries is wrapped to COUNT every HTTP attempt (incl. retries);
  * the session's audit logger is wrapped to CAPTURE each audit record (still written to disk).
The API key is never printed or written.

Usage:
  python redteam_output/run_redteam.py --run 1                     # every prompt once
  python redteam_output/run_redteam.py --run 2 --ids an1,wb8 ...   # selected reruns
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["LOG_DIR"] = str(HERE / "logs")  # keep the project's normal logs clean
sys.path.insert(0, str(ROOT))

from src import api_client as api_mod  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.pipeline import GuardedChatSession  # noqa: E402
from src.topic import Topic  # noqa: E402

CALLS = {"n": 0}
_orig_with_retries = api_mod.ApiClient._with_retries


def _counting_with_retries(self, call):
    def counted():
        CALLS["n"] += 1
        return call()
    return _orig_with_retries(self, counted)


api_mod.ApiClient._with_retries = _counting_with_retries


def stopped_layer(action: str, layers: dict) -> str | None:
    if action == "refused_input":
        if layers.get("injection_detector", {}).get("matched"):
            return "L1"
        if layers.get("provider_content_filter"):
            return "provider_filter"
        if "topic_judge" in layers:
            return "L2b(error)" if layers["topic_judge"].get("error") else "L2b"
        return "L2"
    if action == "refused_output":
        return "L4"
    if action == "error":
        return "error"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-calls", type=int, default=650)
    ap.add_argument("--prompts", default="prompts_frozen.json")
    ap.add_argument("--out", default="", help="suffix for results/replies files, e.g. _heldout_before")
    args = ap.parse_args()

    data = json.loads((HERE / args.prompts).read_text())
    prompts = [p for key, val in data.items() if isinstance(val, list) for p in val]
    if args.ids:
        wanted = set(args.ids.split(","))
        prompts = [p for p in prompts if p["id"] in wanted]

    settings = load_settings()
    topic = Topic.load(settings.topic_config_path)
    results = (HERE / f"results{args.out}.jsonl").open("a", encoding="utf-8")
    full = (HERE / f"replies_full{args.out}.jsonl").open("a", encoding="utf-8")
    consecutive_errors = 0
    start_calls = CALLS["n"]

    for i, p in enumerate(prompts, 1):
        if CALLS["n"] - start_calls >= args.max_calls:
            print(f"STOP: call budget {args.max_calls} reached before {p['id']}")
            return 2
        session = GuardedChatSession.create(settings, topic)
        captured: list[dict] = []
        orig_log = session._audit.log_turn

        def capture(**kw):
            captured.append(kw)
            orig_log(**kw)

        session._audit.log_turn = capture
        for t_idx, turn in enumerate(p["turns"], 1):
            before = CALLS["n"]
            t0 = time.perf_counter()
            try:
                reply = session.handle_message(turn)
                exc = None
            except Exception as e:  # unexpected crash -> record as error
                reply, exc = "", f"{type(e).__name__}: {e}"
            latency = (time.perf_counter() - t0) * 1000
            rec = captured[-1] if captured and not exc else {"action": "error", "layers": {}, "error": exc}
            layers = rec.get("layers", {})
            tri = layers.get("topic_relevance_input", {})
            row = {
                "id": p["id"], "category": p["category"], "expected": p["expected"], "run": args.run,
                "turn": t_idx, "n_turns": len(p["turns"]),
                "action": rec["action"], "stopped_by": stopped_layer(rec["action"], layers),
                "injection_rule": layers.get("injection_detector", {}).get("rule"),
                "input_score": tri.get("score"), "context_score": tri.get("context_score"),
                "band": tri.get("band"),
                "judge": layers.get("topic_judge"),
                "output_score": layers.get("topic_relevance_output", {}).get("score"),
                "output_layer": layers.get("topic_relevance_output"),
                "sanitizer": layers.get("sanitizer"),
                "api_calls": CALLS["n"] - before,
                "latency_ms": round(latency, 1),
                "error": rec.get("error"),
                "reply_400": reply[:400],
            }
            results.write(json.dumps(row, ensure_ascii=False) + "\n")
            full.write(json.dumps({"id": p["id"], "run": args.run, "turn": t_idx, "user": turn,
                                   "reply": reply}, ensure_ascii=False) + "\n")
            results.flush(); full.flush()
            print(f"[{i}/{len(prompts)}] {p['id']} t{t_idx} {row['action']:<15} {str(row['stopped_by']):<10} "
                  f"in={row['input_score']} out={row['output_score']} calls={row['api_calls']} total={CALLS['n']}",
                  flush=True)
            if row["action"] == "error":
                consecutive_errors += 1
                print("  ERROR:", (row["error"] or "")[:200])
                if consecutive_errors >= 5:
                    print("STOP: 5 consecutive API errors")
                    return 3
            else:
                consecutive_errors = 0
    print(f"DONE run {args.run}: {CALLS['n'] - start_calls} API calls (HTTP attempts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
