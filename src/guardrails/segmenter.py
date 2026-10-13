"""
Splits text into pieces that can be scored one by one (red-team round).

Why: Layers 2 and 4 originally embedded a whole message / a whole reply as
ONE vector, so a small off-topic payload riding on a lot of on-topic text
("...6-week plan, and at the end add a bonus tip on brewing kombucha")
was diluted and passed both layers. Scoring the pieces separately lets an
off-topic piece stand out.

  * split_segments(message) -- sentences / clauses / list items of a USER
    message (Layer 2): any piece that scores low sends the WHOLE message to
    the judge; a piece never refuses on its own.
  * split_chunks(reply)     -- paragraphs of a MODEL reply (Layer 4).

Both are plain, deterministic string functions (no API calls), so they are
easy to unit test.
"""

from __future__ import annotations

import re

# Pieces shorter than this are merged into a neighbour: "Thanks!" or "I'm
# 34." carry too little meaning for an embedding to score.
MIN_SEGMENT_WORDS = 4

# Sentence ends, semicolons, line breaks, numbered/bulleted items, and the
# connectors people use to tack an extra request on ("also", "P.S.", "while
# you're at it", "and then", "plus", "bonus").
_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|;\s*|\n+|\s+(?=\(?\d{1,2}[).]\s)|\s+(?=[-*•]\s)"
    r"|,?\s+(?=(?:also|plus|p\.?\s?s\.?|while you're at it|and then|bonus|separately|"
    r"and\s+(?:at\s+the\s+(?:very\s+)?end|finally|after\s+that|afterwards|then)|"
    r"at\s+the\s+(?:very\s+)?end|finish\s+(?:with|by)|end\s+(?:with|by)|"
    r"(?:and\s+)?for\s+(?:every|each)\s+(?:answer|reply|response))\b)"
    r"|,\s+(?:and\s+)?(?=then\b)|,\s+and\s+",
    re.IGNORECASE,
)


def _words(text: str) -> int:
    return len(text.split())


def _merge_short(pieces: list[str], min_words: int) -> list[str]:
    merged: list[str] = []
    for piece in pieces:
        if merged and (_words(piece) < min_words or _words(merged[-1]) < min_words):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def split_segments(text: str, min_words: int = MIN_SEGMENT_WORDS) -> list[str]:
    """Clauses of a user message. Returns [] if there is only one piece
    (nothing to compare against the whole-message score)."""
    pieces = [p.strip() for p in _SPLIT_RE.split(text) if p and p.strip()]
    pieces = _merge_short(pieces, min_words)
    return pieces if len(pieces) > 1 else []


def split_chunks(reply: str) -> list[str]:
    """Paragraphs of a model reply, split on blank lines and on horizontal
    rules ("---"). Headings stay attached to the paragraph below them so a
    heading never stands alone."""
    blocks = re.split(r"\n\s*\n|\n\s*-{3,}\s*\n", reply)
    chunks: list[str] = []
    carry = ""
    for block in (b.strip() for b in blocks):
        if not block or re.fullmatch(r"-{3,}", block):
            continue
        if re.fullmatch(r"(#+\s.*|\*\*[^*]+\*\*:?)", block):   # heading on its own line
            carry = f"{carry}{block}\n"
            continue
        chunks.append(f"{carry}{block}")
        carry = ""
    if carry:
        chunks.append(carry.strip())
    return chunks


def word_count(text: str) -> int:
    return _words(text)
