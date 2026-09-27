"""Default backend: Laya (https://huggingface.co/convaiinnovations/laya).

Laya is a non-autoregressive typed-decision model: give it state + typed
questions, it returns calibrated choice answers in one forward pass and never
generates free text. We still marshal its answers through the same JSON text
contract the worker validates, so retry/validation/recovery behave identically
in every backend.

Limitation (documented in README): Laya cannot write the one-sentence summary,
so the summary field is templated here. FakeLLM replaces this backend in tests
where deterministic labels/scripted failures are required
(LOURA_LLM_BACKEND=fake).
"""

from __future__ import annotations

import asyncio
import json
import threading

from .prompts import parse_user_prompt

QUESTIONS: dict = {
    "category": {
        "type": "choice",
        "instructions": "Which category best describes this support ticket?",
        "criteria": {
            "billing": "invoices, charges, payments, refunds, subscriptions",
            "technical": "bugs, errors, outages, API failures, broken features",
            "account": "login, password, email changes, account access",
            "other": "everything else, feature requests, unclear messages",
        },
    },
    "priority": {
        "type": "choice",
        "instructions": "What priority should this ticket get?",
        "criteria": {
            "low": "nice to have, not urgent",
            "medium": "normal priority",
            "high": "urgent, production blocking",
        },
    },
}


class LayaLLM:
    def __init__(self, model_id: str = "convaiinnovations/laya") -> None:
        self.model_id = model_id
        self._agent = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _ensure_agent(self):
        if self._agent is None:
            with self._load_lock:
                if self._agent is None:
                    try:
                        import laya
                    except ImportError as exc:
                        raise RuntimeError(
                            "LOURA_LLM_BACKEND=laya but the 'laya' package is "
                            "not installed. Run `make install` (or "
                            "`pip install -r requirements-laya.txt`), or set "
                            "LOURA_LLM_BACKEND=fake."
                        ) from exc
                    self._agent = laya.load(self.model_id)
        return self._agent

    async def warmup(self) -> None:
        """Load the checkpoint during app startup (covers first-boot download)."""
        await asyncio.to_thread(self._ensure_agent)

    def _predict(self, subject: str, body: str) -> dict:
        agent = self._ensure_agent()
        # Serialize forwards: workers default to 2; a lock keeps timings and
        # CPU/GPU buffers predictable (Laya's card documents buffer races).
        with self._predict_lock:
            result = agent.predict({"subject": subject, "body": body}, QUESTIONS)
        answers = result["answers"]
        category = answers["category"]["choice"]
        priority = answers["priority"]["choice"]
        label = subject.strip()[:120] or "Untitled ticket"
        summary = f"{label}: classified as {category} with {priority} priority."
        return {"category": category, "priority": priority, "summary": summary}

    async def complete(self, system: str, user: str) -> str:
        subject, body = parse_user_prompt(user)
        payload = await asyncio.to_thread(self._predict, subject, body)
        return json.dumps(payload, ensure_ascii=False)