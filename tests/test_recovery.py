from __future__ import annotations

import asyncio

from app.db import Database
from app.llm import FakeLLM


async def test_pending_rows_reenqueued_on_startup(open_app, drain, settings_factory):
    # Simulate a previous run that died with work still pending:
    # rows exist in SQLite but nothing was ever enqueued.
    settings = settings_factory()
    db = Database(settings.db_path)
    db.init_schema()
    db.create_ticket(
        "t-r1", "Please help", "I cannot log in to my account after the password reset."
    )
    db.close()

    run = await open_app(settings=settings)
    async with run.client() as client:
        await drain(run)
        resp = await client.get("/tickets/t-r1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "classified"
        assert data["attempts"] == 1
        assert data["classification"]["category"] == "account"


async def test_inflight_cancel_keeps_pending_then_recovers(
    open_app, drain, settings_factory
):
    settings = settings_factory()
    blocked = FakeLLM()
    blocked.gate = asyncio.Event()  # model call never returns

    run1 = await open_app(llm=blocked, settings=settings)
    async with run1.client() as client:
        resp = await client.post(
            "/tickets",
            json={
                "id": "t-r2",
                "subject": "Broken export",
                "body": "Our production API returns errors and the nightly job is blocked.",
            },
        )
        assert resp.status_code == 201
        # wait until the worker is inside the model call
        await asyncio.wait_for(blocked.started.wait(), timeout=5)

    # hard stop with work in flight (restart simulation)
    await run1.stop()

    check = Database(settings.db_path)
    row = check.get_ticket("t-r2")
    assert row["status"] == "pending"  # nothing half-written
    assert row["attempts"] == 0
    check.close()

    # second app instance: startup recovery re-enqueues and finishes the job
    run2 = await open_app(settings=settings)
    async with run2.client() as client:
        await drain(run2)
        data = (await client.get("/tickets/t-r2")).json()
        assert data["status"] == "classified"
        assert data["attempts"] == 1
        assert data["classification"]["category"] in {
            "billing",
            "technical",
            "account",
            "other",
        }