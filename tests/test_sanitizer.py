from src.guardrails.sanitizer import sanitize


def test_collapses_whitespace_and_strips():
    result = sanitize("  what   is\n\na good   warm-up?  ", max_chars=2000)
    assert result.text == "what is a good warm-up?"
    assert not result.was_truncated


def test_strips_control_characters():
    result = sanitize("hello\x00\x07world", max_chars=2000)
    assert "\x00" not in result.text
    assert result.had_control_chars is True


def test_truncates_over_limit():
    long_text = "a" * 50
    result = sanitize(long_text, max_chars=10)
    assert len(result.text) == 10
    assert result.was_truncated is True


def test_normalizes_confusable_unicode():
    # Fullwidth "IGNORE" should fold down to ASCII via NFKC so downstream
    # regex-based checks can't be dodged with lookalike characters.
    result = sanitize("\uff29\uff27\uff2e\uff2f\uff32\uff25 instructions", max_chars=2000)
    assert "IGNORE" in result.text.upper()


def test_flags_suspicious_base64_blob():
    blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejEyMzQ1Njc4OTA=" * 2
    result = sanitize(f"here is some info {blob}", max_chars=5000)
    assert result.had_suspicious_encoding is True


def test_normal_climbing_question_untouched_besides_whitespace():
    text = "What's the best way to train finger strength for crimping?"
    result = sanitize(text, max_chars=2000)
    assert result.text == text
    assert not result.had_suspicious_encoding
    assert not result.had_control_chars
