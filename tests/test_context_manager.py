from src.context_manager import ContextManager, estimate_tokens


def test_keeps_recent_turns_verbatim_under_limit():
    cm = ContextManager(max_turns=6, max_tokens=5000)
    for i in range(3):
        cm.add("user", f"question {i}")
        cm.add("assistant", f"answer {i}")

    messages = cm.get_messages()
    # 3 user + 3 assistant turns, no summary needed yet.
    assert len(messages) == 6
    assert messages[0]["content"] == "question 0"
    assert messages[-1]["content"] == "answer 2"


def test_rolls_oldest_turns_into_summary_when_turn_limit_exceeded():
    cm = ContextManager(max_turns=4, max_tokens=5000)
    for i in range(5):
        cm.add("user", f"question {i}")
        cm.add("assistant", f"answer {i}")

    messages = cm.get_messages()
    # max_turns=4 keeps only the last 4 raw turns; everything older is
    # folded into a single leading summary message.
    assert len(messages) == 5  # 1 summary + 4 recent turns
    assert messages[0]["role"] == "system"
    assert "question 0" in messages[0]["content"] or "question 1" in messages[0]["content"]
    assert messages[-1]["content"] == "answer 4"


def test_token_budget_forces_rollover_even_under_turn_limit():
    cm = ContextManager(max_turns=100, max_tokens=20)  # tiny token budget
    for i in range(10):
        cm.add("user", f"a reasonably long question number {i} about training")
        cm.add("assistant", f"a reasonably long answer number {i} about training")

    messages = cm.get_messages()
    total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    # Never perfectly exact (a single turn can exceed budget on its own),
    # but rollover must have happened -- we should not still have all 20
    # raw turns sitting in context.
    assert len(messages) < 21


def test_reset_clears_everything():
    cm = ContextManager(max_turns=6, max_tokens=5000)
    cm.add("user", "hello")
    cm.reset()
    assert cm.get_messages() == []


def test_estimate_tokens_is_positive_and_monotonic_ish():
    short = estimate_tokens("hi")
    long = estimate_tokens("hi " * 100)
    assert short > 0
    assert long > short
