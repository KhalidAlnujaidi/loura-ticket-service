"""Input-side prompt-injection heuristics (advisory layer).

Detection is ADVISORY: a flag never blocks or alters classification -- the
output schema gate remains the hard boundary, so the worst case a successful
injection can reach is still "wrong but valid-shaped" (see
tests/test_injection.py::test_follow_injection_mode_still_schema_valid).
On a flag the ticket classifies normally, a WARNING lands in the admin log,
and GET /admin/metrics reports the running count under guard.injection_flags.

Pattern family mirrors FakeLLM's _parse_injection_hints plus the classic
override / exfiltration / authority forms. All quantifiers are bounded (no
backtracking blowups). A trained classifier (Prompt-Guard-style, selected via
LOURA_INJECTION_GUARD) can slot into this same scan() contract later without
touching callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Flag:
    rule: str
    text: str  # matched excerpt, truncated for safe logging


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget)\b[^.!?]{0,60}"
            r"\b(?:instructions?|prompts?|rules?|guidelines?|directives?)\b",
            re.I,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(?:you are now|act as if|pretend (?:you are|to be)|from now on you)\b",
            re.I,
        ),
    ),
    (
        "mode_override",
        re.compile(
            r"\b(?:developer mode|jail\s?break|no restrictions|without restrictions|unrestricted mode)\b",
            re.I,
        ),
    ),
    (
        "system_exfil",
        re.compile(
            r"\b(?:reveal|print|repeat|output|leak)\b[^.!?]{0,40}"
            r"\b(?:system prompt|hidden instructions|initial instructions|developer (?:prompt|message))\b",
            re.I,
        ),
    ),
    (
        "classification_directive",
        re.compile(r"\b(?:classif\w*|categor\w*|label)\s+(?:it|this|the ticket)\s+as\b", re.I),
    ),
    (
        "summary_directive",
        re.compile(r"\bsummar\w*\s+(?:it|this)\s+as\s+[\"\'\u2018\u2019]", re.I),
    ),
    (
        "authority_claim",
        re.compile(
            r"\b(?:from the (?:ceo|chief executive|administrator)"
            r"|this is (?:the|our) (?:ceo|owner|administrator))\b",
            re.I,
        ),
    ),
    (
        "fake_role_marker",
        re.compile(r"(?:^|\n)\s*(?:\[?system\]?|assistant|developer)\s*:", re.I | re.M),
    ),
)


def scan(subject: str, body: str) -> list[Flag]:
    """Return heuristic flags for a ticket's content (order = PATTERNS order)."""
    text = f"{subject}\n{body}"
    flags: list[Flag] = []
    for name, pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            excerpt = " ".join(match.group(0).split())[:80]
            flags.append(Flag(rule=name, text=excerpt))
    return flags
