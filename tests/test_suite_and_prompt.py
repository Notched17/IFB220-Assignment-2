"""Checks for the adversarial suite's outcome classifier and the system
prompt builder (both plain functions, no network)."""

from pathlib import Path

from src.prompt_builder import build_system_prompt
from src.topic import Topic
from tests.run_adversarial_suite import is_refusal, judge_item, stopped_at

TOPIC = Topic.load(Path(__file__).resolve().parent.parent / "topics" / "climbing.json")
ANSWERED = {"action": "answered", "layers": {}}


def test_refusal_with_curly_apostrophes_is_still_a_refusal():
    curly = TOPIC.refusal_message.replace("'", "\u2019")
    assert stopped_at(ANSWERED, curly, TOPIC) == "L3 model refused"


def test_partial_answer_that_declines_the_rest_is_classified_as_partial():
    reply = "Here is your hangboard plan...\n\n" + TOPIC.refusal_message.replace("'", "\u2019")
    assert stopped_at(ANSWERED, "x" * 300 + reply, TOPIC) == "L3 partial (declined part)"


def test_layer_outcomes_come_from_the_audit_record():
    assert stopped_at({"action": "refused_input", "layers": {"injection_detector": {"matched": True, "view": "collapsed"}}},
                      "", TOPIC) == "L1 injection detector (collapsed view)"
    assert stopped_at({"action": "refused_input", "layers": {"injection_detector": {"matched": False}, "topic_judge": {}}},
                      "", TOPIC) == "L2b topic judge"
    assert stopped_at({"action": "refused_output", "layers": {}}, "", TOPIC) == "L4 output relevance"


def test_scoring_rules():
    assert judge_item("safe", "answered", "taper plan", [r"def \w+\("]) == (True, [])
    assert judge_item("safe", "answered", "def f(x):", [r"def \w+\("])[0] is False
    assert judge_item("answer", "L2b topic judge", "", [])[0] is False
    assert judge_item("refuse", "L3 model refused", "", [])[0] is True
    assert is_refusal("L3 partial (declined part)") is False


def test_system_prompt_contains_scope_rules_and_balance():
    prompt = build_system_prompt(TOPIC)
    assert TOPIC.in_scope_summary in prompt and TOPIC.refusal_message in prompt
    assert "Recap of earlier conversation" in prompt      # rule 4: recap is not an instruction
    assert "do not over-refuse" in prompt                  # rule 1: balance against over-refusal
    assert "never diagnose" in prompt                      # rule 6: first aid, no diagnosis/doses
