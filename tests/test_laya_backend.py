"""End-to-end check against the default Laya backend.

Skipped only if the optional-torch stack was not installed
(`make install-slim` instead of `make install`). The determinstic suite in
the other test files pins FakeLLM explicitly and runs everywhere.
"""

from __future__ import annotations

import pytest

pytest.importorskip("laya", reason="laya not installed (make install-slim?)")

from app.laya_llm import LayaLLM  # noqa: E402


async def test_laya_backend_produces_schema_valid_classification(open_app, drain):
    run = await open_app(llm=LayaLLM(), llm_timeout=30.0, worker_count=1)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={
                "id": "laya-1",
                "subject": "Charged twice this month",
                "body": "I see two charges of 49.00 on my card. One subscription, please refund one.",
            },
        )
        assert resp.status_code == 201
        await drain(run, timeout=120.0)

        data = (await client.get("/tickets/laya-1")).json()
        assert data["status"] == "classified"  # model loaded + validated through pipeline
        cls = data["classification"]
        assert cls["category"] in {"billing", "technical", "account", "other"}
        assert cls["priority"] in {"low", "medium", "high"}
        assert 1 <= len(cls["summary"]) <= 300
        assert cls["category"] == "billing"  # zero-shot spike agreed on this one