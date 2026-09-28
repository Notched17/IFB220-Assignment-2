"""
Guardrail Layer 0 -- Input sanitisation.

Purpose: normalise the raw user string BEFORE any topic or injection
checks run, so that later layers see consistent input and can't be
dodged with simple formatting tricks (invisible characters, extra
whitespace, exotic unicode lookalikes, letter-spacing, base64, absurd
length).

This layer never makes an allow/refuse decision by itself -- it cleans the
text, reports what it changed (for the audit log), and hands the
injection detector two extra "views" of the input to scan:

  * collapsed_text   -- letter-spaced runs ("i g n o r e  a l l") joined
                        back into words ("ignore all")
  * decoded_segments -- the decoded text of any base64 blobs

The pipeline decides what to do with those views (see src/pipeline.py).
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_WHITESPACE = re.compile(r"\s+")

# A long run of base64-alphabet characters is a red flag: legitimate
# on-topic questions are written in natural language, not encoded blobs.
BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")
# Shorter candidates that are worth trying to DECODE (and re-scan).
_BASE64_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}")

# 5+ single characters separated by 1-3 spaces or dots: "i g n o r e", "i.g.n.o.r.e".
_SPACED_RUN_RE = re.compile(r"(?<!\w)\w(?:(?: {1,3}|\.)\w(?!\w)){4,}")


@dataclass
class SanitizeResult:
    text: str
    was_truncated: bool
    had_control_chars: bool
    had_suspicious_encoding: bool
    format_chars_removed: int = 0
    collapsed_text: str | None = None  # only set if letter-spacing was found
    decoded_segments: tuple[str, ...] = ()


def _strip_format_chars(text: str) -> tuple[str, int]:
    """Remove Unicode 'format' characters (category Cf): zero-width space
    and joiners, word joiner, BOM, soft hyphen, and the bidi embedding /
    override / isolate controls. They are invisible, so they can split a
    keyword ("ig​nore") without the reader noticing."""
    kept = [ch for ch in text if unicodedata.category(ch) != "Cf"]
    return "".join(kept), len(text) - len(kept)


def _collapse_run(run: str) -> str:
    if "." in run:  # "i.g.n.o.r.e a.l.l": dots join letters, spaces split words
        words = re.split(r"\s+", run)
    else:           # "i g n o r e  a l l": 2+ spaces split words
        words = re.split(r"\s{2,}", run)
    return " ".join(re.sub(r"[ .]", "", w) for w in words)


def collapse_letter_spacing(text: str) -> str | None:
    """Return a copy of `text` with letter-spaced runs joined, or None if
    there were none."""
    if not _SPACED_RUN_RE.search(text):
        return None
    collapsed = _SPACED_RUN_RE.sub(lambda m: _collapse_run(m.group(0)), text)
    return _MULTI_WHITESPACE.sub(" ", collapsed).strip()


def decode_base64_segments(text: str) -> tuple[str, ...]:
    """Decode base64-looking blobs; keep results that are mostly readable text."""
    decoded: list[str] = []
    for match in _BASE64_CANDIDATE_RE.finditer(text):
        blob = match.group(0)
        # Real base64 of text almost always mixes upper and lower case;
        # this skips long ordinary words ("internationalisation").
        if not (re.search(r"[A-Z]", blob) and re.search(r"[a-z]", blob)):
            continue
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        text_out = raw.decode("utf-8", errors="ignore").strip()
        if len(text_out) < 4:
            continue
        printable = sum(ch.isprintable() or ch.isspace() for ch in text_out)
        if printable / len(text_out) >= 0.9:
            decoded.append(text_out)
    return tuple(decoded)


def sanitize(raw: str, max_chars: int) -> SanitizeResult:
    had_control = bool(_CONTROL_CHARS.search(raw))
    cleaned = _CONTROL_CHARS.sub("", raw)

    # Invisible format/bidi characters go BEFORE NFKC, so a keyword split
    # by a zero-width space is rejoined before any matching happens.
    cleaned, format_removed = _strip_format_chars(cleaned)

    # NFKC folds visually-confusable unicode (fullwidth letters, some
    # homoglyphs) down to a canonical form so keyword/regex checks in the
    # next layer can't be trivially dodged with lookalike characters.
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # The collapsed view needs the original spacing (2+ spaces = word
    # break), so it is built before whitespace is normalised.
    collapsed = collapse_letter_spacing(cleaned)

    cleaned = _MULTI_WHITESPACE.sub(" ", cleaned).strip()

    # Views are computed on the whole input (before truncation), so
    # anything hidden past the cut-off is still scanned -- conservative.
    decoded = decode_base64_segments(cleaned)
    had_suspicious_encoding = bool(BASE64_BLOB_RE.search(cleaned)) or bool(decoded)

    was_truncated = len(cleaned) > max_chars
    if was_truncated:
        cleaned = cleaned[:max_chars]

    return SanitizeResult(
        text=cleaned,
        was_truncated=was_truncated,
        had_control_chars=had_control,
        had_suspicious_encoding=had_suspicious_encoding,
        format_chars_removed=format_removed,
        collapsed_text=collapsed,
        decoded_segments=decoded,
    )
