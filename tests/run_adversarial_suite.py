#!/usr/bin/env python3
"""
Runs the prompts in tests/adversarial_prompts.json through the guardrail
pipeline and prints a pass/fail report. This is the script referenced in
docs/TESTING.md as the source of the adversarial testing evidence.

Three layers of rigor, on purpose:

  1. Layers 0-1 (sanitizer, injection detector) are REAL and deterministic
     -- no API needed -- so they always run, against the actual
     production code.

  2. If real credentials are available (API_KEY / AZURE_OPENAI_BASE_URL
     set, e.g. in .env), the FULL live pipeline runs against the real
     IFB220 portal (GPT-4.1-mini + Ada-002) -- this is the actual
     end-to-end acceptance test and is what should be run, and its
     output captured, before submission.

  3. If no credentials are available (e.g. this sandbox, or a CI runner
     with no secrets), an OFFLINE STAND-IN pipeline is used instead, with
     a crude bag-of-words "embedding" substituted for Ada-002. This is
     clearly labelled as a smoke test of the pipeline's WIRING ONLY -- it
     proves the code paths are reachable and the decision logic runs
     end-to-end, but its topic-relevance judgements are NOT representative
     of the real embedding model and must not be quoted as evidence of
     guardrail accuracy. Only mode 2's output should be used for that.

Usage:
    python -m tests.run_adversarial_suite
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.audit_logger import AuditLogger
from src.config import Settings, load_settings
from src.guardrails import injection_detector, sanitizer
from src.pipeline import GuardedChatSession
from src.topic import Topic
from src.usage_monitor import UsageMonitor

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT / "tests" / "adversarial_prompts.json"


# ---------------------------------------------------------------------
# Layers 0-1: always real, no API required.
# ---------------------------------------------------------------------
def run_layer_0_1_report(prompts: list[dict]) -> list[dict]:
    rows = []
    for item in prompts:
        clean = sanitizer.sanitize(item["prompt"], max_chars=2000)
        injection = injection_detector.scan(clean.text)
        expected_injection = item["expected_layer"] == "injection_detector"
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "injection_matched": injection.matched,
            "injection_rule": injection.matched_rule,
            "expected_injection_layer": expected_injection,
            "correct_at_this_layer": injection.matched == expected_injection,
        })
    return rows


# ---------------------------------------------------------------------
# Offline stand-in embedding, used ONLY when no live credentials exist.
# ---------------------------------------------------------------------
_HASH_DIM = 512


class _FakeEmbeddingApiClient:
    """Hashed bag-of-words 'embedding' so the FULL pipeline (including the
    topic-relevance layers, which expect fixed-length dense float vectors)
    can be smoke-tested with no network, using the SAME cosine/centroid
    math as production. This is NOT a substitute for testing against real
    Ada-002 embeddings -- see module docstring."""

    _TOKEN_RE = re.compile(r"[a-z0-9']+")

    class _FakeChatResult:
        def __init__(self, content):
            self.content = content
            self.prompt_tokens = 10
            self.completion_tokens = 10

    class _FakeEmbedResult:
        def __init__(self, vector, tokens):
            self.vector = vector
            self.total_tokens = tokens

    def embed(self, text: str):
        tokens = self._TOKEN_RE.findall(text.lower())
        vector = [0.0] * _HASH_DIM
        for tok in tokens:
            vector[hash(tok) % _HASH_DIM] += 1.0
        return self._FakeEmbedResult(vector=vector, tokens=len(tokens))

    def chat_complete(self, messages: list[dict], *, temperature: float = 0.4):
        # Deliberately naive stand-in "model": just echoes the user's last
        # message back so the output-topicality re-check has something
        # realistic (same vocabulary as the input) to score. Real model
        # behaviour must be verified live (see mode 2).
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self._FakeChatResult(f"[offline stand-in echo] {last_user}")


def _patch_relevance_for_offline_mode(session: GuardedChatSession) -> None:
    """Monkeypatch BOTH the session's chat client and its relevance
    checker's embedding client to the offline stand-in, purely so this
    script can run without network access. (Accessing these internals
    directly is test-only wiring -- production code never does this.)"""
    fake_api = _FakeEmbeddingApiClient()

    session._api = fake_api  # used directly by pipeline.py for chat_complete

    checker = session._relevance
    checker._api_client = fake_api
    checker._centroid = None
    checker._cache_path = checker._cache_path.with_suffix(".offline_demo.json")
    if checker._cache_path.exists():
        checker._cache_path.unlink()


def run_full_pipeline_report(prompts: list[dict], live: bool) -> list[dict]:
    settings: Settings
    if live:
        settings = load_settings()
    else:
        settings = Settings(
            api_key="offline-demo",
            base_url="https://offline-demo.invalid",
            api_version="offline",
            chat_deployment="offline-demo",
            embedding_deployment="offline-demo",
            topic_config_path=ROOT / "topics" / "climbing.json",
            similarity_threshold_override=0.15,  # bag-of-words scores run much lower than real embeddings
            max_input_chars=2000,
            check_output_topicality=True,
            max_context_turns=6,
            max_context_tokens=3000,
            summarize_with_llm=False,
            request_timeout_s=5.0,
            max_retries=1,
            chat_cost_per_1k_prompt=0.0,
            chat_cost_per_1k_completion=0.0,
            embedding_cost_per_1k=0.0,
            log_dir=ROOT / "logs" / ("live_adversarial_run" if live else "offline_demo_run"),
        )

    topic = Topic.load(settings.topic_config_path)
    session = GuardedChatSession.create(settings, topic)
    if not live:
        _patch_relevance_for_offline_mode(session)

    rows = []
    for item in prompts:
        response = session.handle_message(item["prompt"])
        refused = response in (topic.refusal_message, topic.injection_refusal_message)
        should_pass = item["expected_layer"] == "none_should_pass"
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "refused": refused,
            "response_preview": response[:80],
            "expected_to_pass": should_pass,
            "correct": (refused != should_pass),
        })
    return rows, session


def main() -> None:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))["single_turn"]

    print("=" * 78)
    print("LAYER 0-1 REPORT (real sanitizer + injection detector, no API needed)")
    print("=" * 78)
    l01_rows = run_layer_0_1_report(prompts)
    for row in l01_rows:
        mark = "PASS" if row["correct_at_this_layer"] else "FAIL"
        print(f"[{mark}] {row['id']:6} {row['category']:28} matched={row['injection_matched']!s:5} rule={row['injection_rule']}")
    l01_pass = sum(r["correct_at_this_layer"] for r in l01_rows)
    print(f"\nLayer 0-1: {l01_pass}/{len(l01_rows)} behaved as expected\n")

    try:
        load_settings()
        live = True
    except EnvironmentError:
        live = False

    mode_label = "LIVE (real IFB220 portal)" if live else "OFFLINE STAND-IN (bag-of-words demo -- NOT representative of real embeddings)"
    print("=" * 78)
    print(f"FULL PIPELINE REPORT -- mode: {mode_label}")
    print("=" * 78)
    full_rows, session = run_full_pipeline_report(prompts, live=live)
    for row in full_rows:
        mark = "PASS" if row["correct"] else "FAIL"
        print(f"[{mark}] {row['id']:6} {row['category']:28} refused={row['refused']!s:5} -> {row['response_preview']!r}")
    full_pass = sum(r["correct"] for r in full_rows)
    print(f"\nFull pipeline: {full_pass}/{len(full_rows)} behaved as expected")
    print(f"Session usage: {session.usage_summary()}")

    if not live:
        print(
            "\nNOTE: this was the OFFLINE STAND-IN run. Before submission, fill in .env "
            "with real IFB220 credentials and re-run this script to get the live report "
            "-- that is the result that belongs in docs/TESTING.md as final evidence."
        )


if __name__ == "__main__":
    main()
