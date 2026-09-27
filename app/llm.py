from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError

from .schemas import Category, Classification, Priority

logger = logging.getLogger(__name__)

CATEGORIES: tuple[Category, ...] = ("billing", "technical", "account", "other")
PRIORITIES: tuple[Priority, ...] = ("low", "medium", "high")

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": (
        "charge", "charged", "refund", "invoice", "subscription",
        "overcharged", "billing", "payment",
    ),
    "technical": (
        "error", "500", "timeout", "api", "broken", "e_timeout",
        "not fixed", "export", "uploading", "stderr",
    ),
    "account": (
        "login", "log in", "password", "email", "account", "sign in",
        "invalid credentials",
    ),
    "other": (),
}

_SUMMARY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"download the invoices|where do i download",
     "Customer wants to know where to download their invoices."),
    (r"charged twice|two charges|duplicate charge",
     "Customer reports a duplicate charge on one subscription and asks for a refund."),
    (r"old company name|renamed|wrong company name",
     "Invoice PDF still shows the old company name after a rename."),
    (r"password reset|reset my password|invalid credentials",
     "Customer cannot log in after a password reset."),
    (r"change the email|email address on my account",
     "Customer wants to change the email address on their account."),
    (r"http 500|500s|returning 500",
     "Production API is returning HTTP 500 errors, blocking the customer's job."),
    (r"e_timeout|still broken|same error",
     "Previous fix did not work; uploads still fail with a timeout error."),
    (r"dark mode|feature request|nice to have",
     "Feature request: dark mode in the dashboard."),
    (r"data export|zip file|contains nothing",
     "Data export arrived empty: the zip file contains nothing."),
    (r"refund", "Customer requests a refund."),
    (r"invoice", "Customer has a question about an invoice."),
)


class ModelOutputError(Exception):
    """The model returned text that is not valid JSON or violates the schema."""


class LLM(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# JSON extraction + strict validation (the model-boundary choke point)
# ---------------------------------------------------------------------------

def _balanced_objects(text: str) -> list[str]:
    """Yield every top-level balanced {...} candidate, string-aware."""
    candidates: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    start = -1
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(text[start : i + 1])
                    start = -1
    return candidates


def extract_json(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise ModelOutputError("empty model output")
    stripped = text.strip()
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(stripped)
    candidates.extend(_balanced_objects(stripped))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ModelOutputError("no valid JSON object found in model output")


def validate_classification(raw: str) -> Classification:
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ModelOutputError("model output is not a JSON object")
    try:
        return Classification.model_validate(data)
    except ValidationError as exc:
        short = exc.errors()[:3]
        raise ModelOutputError(f"classification schema invalid: {short}") from exc


def reason_for(exc: BaseException) -> str:
    if isinstance(exc, ModelOutputError):
        reason = f"invalid model output: {exc}"
    else:
        reason = f"llm error: {type(exc).__name__}: {exc}"
    return reason[:500]


# ---------------------------------------------------------------------------
# FakeLLM: keyword classifier + scripted failure modes + injection demo
# ---------------------------------------------------------------------------

def _extract_ticket_text(user: str) -> str:
    match = re.search(r"<ticket>(.*?)</ticket>", user, re.S)
    inner = match.group(1) if match else user
    return html.unescape(inner)


def _keyword_category(text: str) -> str:
    scores = {
        cat: sum(1 for kw in kws if kw in text)
        for cat, kws in CATEGORY_KEYWORDS.items()
    }
    best = max(CATEGORIES, key=lambda cat: scores.get(cat, 0))
    if scores.get(best, 0) == 0:
        return "other"
    return best


def _keyword_priority(text: str) -> str:
    if any(tok in text for tok in ("nice to have", "not urgent", "no rush", "low priority")):
        return "low"
    if any(tok in text for tok in ("urgent", "production", "blocking", "asap", "critical", "outage")):
        return "high"
    return "medium"


def _keyword_summary(text: str, subject: str, category: str) -> str:
    for pattern, summary in _SUMMARY_PATTERNS:
        if re.search(pattern, text, re.I):
            return summary
    label = subject.strip()[:80] or "no subject"
    return f"Customer message with no clear keyword signal; filed as {category} (subject: {label})."


def _parse_injection_hints(text: str) -> dict[str, str]:
    hints: dict[str, str] = {}
    if match := re.search(
        r"classif\w*\s+(?:it\s+)?as\s+(billing|technical|account|other)\b",
        text,
        re.I,
    ):
        hints["category"] = match.group(1).lower()
    if match := re.search(r"priority\s+(low|medium|high)\b", text, re.I):
        hints["priority"] = match.group(1).lower()
    if match := re.search(
        r"summar\w*\s+(?:it\s+)?as\s*[\"']([^\"']+)[\"']", text, re.I
    ):
        hints["summary"] = match.group(1)
    return hints


class FakeLLM:
    """Keyword classifier standing in for a hosted LLM.

    scripted responses walk a list first (invalid JSON, exceptions, sleeps),
    then fall back to keyword mode. follow_injection=True obeys instructions
    embedded in the ticket, demonstrating the residual semantic-injection risk.
    """

    def __init__(
        self,
        *,
        scripted: list[Any] | None = None,
        follow_injection: bool = False,
    ) -> None:
        self.scripted: list[Any] = list(scripted or [])
        self.follow_injection = follow_injection
        self.calls = 0
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.scripted:
            return await self._next_scripted()
        return self._classify(user)

    async def _next_scripted(self) -> str:
        item = self.scripted.pop(0)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("scripted failure")
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, tuple) and item and item[0] == "sleep":
            await asyncio.sleep(item[1])
            if len(item) > 2 and item[2] is not None:
                return item[2]
            return ""  # fall through to validation failure
        if isinstance(item, str):
            return item
        raise TypeError(f"unsupported scripted item: {item!r}")

    def _classify(self, user: str) -> str:
        text = _extract_ticket_text(user)
        subject_match = re.search(r"<subject>(.*?)</subject>", text, re.S)
        subject = html.unescape(subject_match.group(1)).strip() if subject_match else ""
        lowered = text.lower()
        category = _keyword_category(lowered)
        priority = _keyword_priority(lowered)
        summary = _keyword_summary(lowered, subject, category)
        if self.follow_injection and (hints := _parse_injection_hints(text)):
            category = hints.get("category", category)
            priority = hints.get("priority", priority)
            summary = hints.get("summary", summary)
        payload = {"category": category, "priority": priority, "summary": summary}
        return json.dumps(payload, ensure_ascii=False)