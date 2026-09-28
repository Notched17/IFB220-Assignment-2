"""
Guardrail Layer 1 -- Prompt-injection / jailbreak heuristic detector.

Purpose: catch the well-known family of "override the system prompt"
attacks BEFORE we spend a chat-completion call on them, and independently
of whether the model itself would have resisted them. This is a
programmatic, non-model layer on purpose: relying only on the model to
defend its own instructions means a single successful jailbreak defeats
the whole system. Even if this layer misses something and the model
*does* get manipulated, Layer 4 (output topicality re-check) is still
there to catch a response that drifted off-topic as a result.

This is intentionally a fast, cheap, explainable heuristic layer, not a
full classifier -- it is one layer among several, not the whole defence.

Design note (found during adversarial testing, see docs/TESTING.md):
an earlier version used single regexes with a fixed word ORDER, e.g.
requiring "ignore ... previous ... instructions" in that exact sequence.
That missed real attacks phrased the other way round, e.g. "disregard
the rules above" (rule-word before the "above"/scope word) or
"instructions were you given ... print them verbatim" (target noun
before the verb). The checks below are order-agnostic: they look for a
"trigger" word/phrase and a "target" word/phrase within a character
window of each other, regardless of which comes first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


def _finditer_all(patterns: list[str], text: str):
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            yield m.start()


def _near(text: str, group_a: list[str], group_b: list[str], window: int) -> bool:
    """True if any match from group_a and any match from group_b occur
    within `window` characters of each other, in either order."""
    positions_a = list(_finditer_all(group_a, text))
    if not positions_a:
        return False
    positions_b = list(_finditer_all(group_b, text))
    if not positions_b:
        return False
    return any(abs(a - b) <= window for a in positions_a for b in positions_b)


# --- Rule definitions ---------------------------------------------------
# Word lists are data: extending coverage means adding a pattern here, not
# writing a new rule. Checks stay order-agnostic via _near().

_OVERRIDE_VERBS = [r"\bignore\b", r"\bdisregard\b", r"\bforget\b", r"\boverride\b"]
_OVERRIDE_TARGETS = [
    r"\bprevious\b", r"\bprior\b", r"\babove\b",
    r"\binstructions?\b", r"\brules?\b(?!\s+of\s+thumb)", r"\bprompt\b", r"\brestrictions?\b",
]

# Same attack in Spanish / French / German ("Ignora todas las instrucciones
# anteriores", "Ignorez les instructions précédentes", "Ignoriere alle
# vorherigen Anweisungen").
_FOREIGN_OVERRIDE_VERBS = [
    r"\bignora\b", r"\bignorar\b", r"\bignoren\b", r"\bolvida\b", r"\bolvidar\b",
    r"\bignorez\b", r"\bignorer\b", r"\boubliez\b", r"\boublie\b",
    r"\bignoriere\b", r"\bignorieren\b", r"\bvergiss\b", r"\bvergessen\b",
]
_FOREIGN_OVERRIDE_TARGETS = [
    r"\binstrucciones\b", r"\breglas\b", r"\banteriores\b", r"\bindicaciones\b",
    r"\binstructions\b", r"\bconsignes\b", r"\br[èe]gles\b", r"\bpr[ée]c[ée]dentes?\b",
    r"\banweisungen\b", r"\bregeln\b", r"\bvorherigen\b",
]

_EXTRACTION_VERBS = [
    r"\breveal\b", r"\bshow\b", r"\bprint\b", r"\brepeat\b", r"\boutput\b", r"\bdisplay\b",
    r"\bsummari[sz]e\b", r"\blist\b", r"\bdescribe\b", r"\bexplain\b", r"\btranslate\b",
    r"\brecite\b", r"\bdump\b", r"\btell\s+me\b", r"\bwrite\s+out\b", r"\bspell\s+out\b",
]
# Targets are anchored to the ASSISTANT's own setup ("your rules", "the
# rules you were set up with"), so ordinary requests like "explain the
# rules of bouldering comps" or "show me the instructions for a
# figure-eight" are not flagged.
_SETUP_NOUNS = r"(instructions?|rules?|prompt|configuration|config|guidelines|directives|programming)"
_EXTRACTION_TARGETS = [
    r"\bsystem\s+(prompt|message|instructions?)\b",
    r"\bwords?\s+above\b", r"\btext\s+above\b", r"\beverything\s+above\b",
    r"\byou\s+were\s+given\b", r"\bwere\s+you\s+given\b", r"\byou\s+given\b",
    r"\byour\s+(\w+\s+)?" + _SETUP_NOUNS + r"\b",
    r"\b" + _SETUP_NOUNS + r"\s+(that\s+)?you(\s+were|\s+have\s+been|'ve\s+been|\s+are|'re)\b",
]

_TOLD_PATTERNS = [
    r"\bwhat\s+(were|have)\s+you\s+(been\s+)?(told|instructed|asked\s+to\s+do)\b",
    r"\bhow\s+(were|have)\s+you\s+(been\s+)?(set\s+up|configured|instructed|programmed)\b",
]

_PERSONA_VERBS = [r"\bact\b", r"\bpretend\b", r"\broleplay\b", r"\bimagine\b"]
_PERSONA_TARGETS = [r"\bas\s+(a|an)?\b", r"\byou\s+are\b", r"\byou're\b", r"\bto\s+be\b"]
_PERSONA_ALLOWED_NEARBY = re.compile(
    r"\b(climber|coach|instructor|guide|mentor|advisor)\b", re.IGNORECASE
)

_KNOWN_JAILBREAK_PERSONAS = [r"\bDAN\b", r"\bAIM\b", r"\bSTAN\b", r"\bDUDE\b"]
_UNRESTRICTED_HINTS = [
    r"\bno\s+restrictions?\b", r"\bwithout\s+restrictions?\b", r"\bunrestricted\b",
    r"\bno\s+rules\b", r"\bdoes\s+anything\b", r"\bcan\s+do\s+anything\b",
]


def _rule_override_instructions(text: str) -> bool:
    return _near(text, _OVERRIDE_VERBS, _OVERRIDE_TARGETS, window=40)


def _rule_foreign_override(text: str) -> bool:
    return _near(text, _FOREIGN_OVERRIDE_VERBS, _FOREIGN_OVERRIDE_TARGETS, window=60)


def _rule_what_were_you_told(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _TOLD_PATTERNS)


def _rule_new_instructions_claim(text: str) -> bool:
    return bool(re.search(r"\b(new|updated)\s+(system\s+)?instructions?\b", text, re.IGNORECASE))


def _rule_fake_system_role(text: str) -> bool:
    return bool(re.search(
        r"(\[\s*/?\s*system\s*\]|###\s*(system|admin|developer)\b|^\s*system\s*:)",
        text, re.IGNORECASE | re.MULTILINE,
    ))


def _rule_authority_claim(text: str) -> bool:
    return bool(re.search(
        r"\b(as\s+the|i\s+am\s+the)\s+(developer|administrator|system\s+admin|creator|owner)\s+of\s+this\b",
        text, re.IGNORECASE,
    ))


def _rule_persona_override(text: str) -> bool:
    if not _near(text, _PERSONA_VERBS, _PERSONA_TARGETS, window=25):
        return False
    # Allowed if the persona itself is climbing-coach-flavoured (e.g.
    # "act as a supportive coach") -- only flag genuine persona swaps.
    return not _PERSONA_ALLOWED_NEARBY.search(text)


def _rule_developer_mode(text: str) -> bool:
    return bool(re.search(
        r"\b(developer\s+mode|jailbreak|dan\s+mode|unrestricted\s+mode|no\s+restrictions?\s+mode)\b",
        text, re.IGNORECASE,
    ))


def _rule_known_jailbreak_persona(text: str) -> bool:
    # A bare name like "DAN" is too common to flag alone (false positives
    # on ordinary names), so this only fires when paired with an explicit
    # "no restrictions" style phrase nearby.
    return _near(text, _KNOWN_JAILBREAK_PERSONAS, _UNRESTRICTED_HINTS, window=60)


def _rule_prompt_extraction(text: str) -> bool:
    return _near(text, _EXTRACTION_VERBS, _EXTRACTION_TARGETS, window=75)


_RULES: list[tuple[str, Callable[[str], bool]]] = [
    ("override_instructions", _rule_override_instructions),
    ("foreign_language_override", _rule_foreign_override),
    ("new_instructions_claim", _rule_new_instructions_claim),
    ("fake_system_role", _rule_fake_system_role),
    ("authority_claim", _rule_authority_claim),
    ("persona_override", _rule_persona_override),
    ("developer_mode", _rule_developer_mode),
    ("known_jailbreak_persona", _rule_known_jailbreak_persona),
    ("prompt_extraction", _rule_prompt_extraction),
    ("what_were_you_told", _rule_what_were_you_told),
]


@dataclass
class InjectionResult:
    matched: bool
    matched_rule: str | None


def scan(text: str) -> InjectionResult:
    for name, check in _RULES:
        if check(text):
            return InjectionResult(matched=True, matched_rule=name)
    return InjectionResult(matched=False, matched_rule=None)
