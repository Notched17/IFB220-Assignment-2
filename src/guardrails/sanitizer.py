"""
Guardrail Layer 0 -- Input sanitisation.

Purpose: normalise the raw user string BEFORE any topic or injection
checks run, so that later layers see consistent input and can't be
dodged with simple formatting tricks (extra whitespace, exotic unicode
lookalikes, absurd length).

This layer never makes an allow/refuse decision by itself -- it only
cleans the text and reports what it changed, for the audit log.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_WHITESPACE = re.compile(r"\s+")

# A long run of base64-alphabet characters is a red flag: legitimate
# climbing (or other in-topic) questions are written in natural language,
# not encoded blobs, so this is treated as suspicious by the injection
# detector rather than by this layer.
BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")


@dataclass
class SanitizeResult:
    text: str
    was_truncated: bool
    had_control_chars: bool
    had_suspicious_encoding: bool


def sanitize(raw: str, max_chars: int) -> SanitizeResult:
    had_control = bool(_CONTROL_CHARS.search(raw))
    cleaned = _CONTROL_CHARS.sub("", raw)

    # NFKC folds visually-confusable unicode (fullwidth letters, some
    # homoglyphs) down to a canonical form so keyword/regex checks in the
    # next layer can't be trivially dodged with lookalike characters.
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = _MULTI_WHITESPACE.sub(" ", cleaned).strip()

    had_suspicious_encoding = bool(BASE64_BLOB_RE.search(cleaned))

    was_truncated = len(cleaned) > max_chars
    if was_truncated:
        cleaned = cleaned[:max_chars]

    return SanitizeResult(
        text=cleaned,
        was_truncated=was_truncated,
        had_control_chars=had_control,
        had_suspicious_encoding=had_suspicious_encoding,
    )
