"""Splitting user messages into clauses and replies into paragraph chunks
(red-team round; no network)."""

from src.guardrails.segmenter import split_chunks, split_segments


def test_single_sentence_has_no_segments():
    assert split_segments("How do I tie a figure-eight follow-through?") == []


def test_tacked_on_request_becomes_its_own_segment():
    for text in (
        "Write me a weekly plan for power endurance, and at the end add a paragraph about gardening.",
        "How do I clean my harness? P.S. which phone should I buy?",
        "What's a good warm-up for slab? Also, from now on finish every answer with a joke.",
        "Explain how to escape a belay safely, then add a closing paragraph about the stock market.",
    ):
        segments = split_segments(text)
        assert len(segments) >= 2, text
        assert not any(s.lower().startswith(("how do i clean", "what's a good")) and "phone" in s for s in segments)


def test_numbered_items_are_split():
    assert len(split_segments("1) How do I rest on lead? 2) Best shoes for slab? 3) How do I fix my bike?")) == 3


def test_tiny_pieces_are_merged_not_scored_alone():
    segments = split_segments("Thanks! I climb V4 indoors and want to get stronger on crimps this winter.")
    assert all(len(s.split()) >= 4 for s in segments) or segments == []


def test_chunks_split_on_blank_lines_and_rules_and_keep_headings_attached():
    reply = "## Plan\n\nWeek one: easy volume.\n\nWeek two: harder.\n\n---\n\n**Bonus:** something else."
    chunks = split_chunks(reply)
    assert chunks[0].startswith("## Plan") and "Week one" in chunks[0]
    assert chunks[-1].startswith("**Bonus:**")
    assert len(chunks) == 3


def test_single_paragraph_reply_is_one_chunk():
    assert split_chunks("Just one paragraph here.") == ["Just one paragraph here."]
