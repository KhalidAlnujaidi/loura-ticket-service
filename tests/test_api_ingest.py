from __future__ import annotations

import asyncio

from app.db import Database
from app.llm import FakeLLM
from app.seed import load_samples


async def test_create_returns_201_pending_then_classifies(open_app, drain):
    llm = FakeLLM()
    llm.gate = asyncio.Event()  # hold classification until we release
    run = await open_app(llm=llm)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={
                "id": "t-1",
                "subject": "Charged twice this month",
                "body": "Two charges of 49.00 this month on one subscription. Refund please.",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert data["attempts"] == 0
        assert data["classification"] is None
        assert data["failure_reason"] is None

        llm.gate.set()
        await drain(run)

        resp = await client.get("/tickets/t-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "classified"
        assert data["attempts"] == 1
        assert data["failure_reason"] is None
        assert data["classification"]["category"] == "billing"
        assert data["classification"]["priority"] in {"low", "medium", "high"}
        assert 1 <= len(data["classification"]["summary"]) <= 300


async def test_duplicate_id_is_idempotent_no_rerun(open_app, drain):
    llm = FakeLLM()
    llm.gate = asyncio.Event()
    run = await open_app(llm=llm)
    payload = {
        "id": "t-dup",
        "subject": "Charged twice",
        "body": "Please refund the duplicate charge on my invoice.",
    }
    async with run.client() as client:
        first = await client.post("/tickets", json=payload)
        assert first.status_code == 201

        second = await client.post("/tickets", json=payload)
        assert second.status_code == 200
        assert second.json() == first.json()

        before = await client.get("/tickets/t-dup")
        assert before.json()["attempts"] == 0
        assert before.json()["status"] == "pending"

        listing = await client.get("/tickets")
        assert listing.json()["total"] == 1

        llm.gate.set()
        await drain(run)

        after = await client.get("/tickets/t-dup")
        assert after.json()["status"] == "classified"
        assert after.json()["attempts"] == 1
        assert llm.calls == 1


async def test_create_validation_422s(open_app):
    run = await open_app()
    async with run.client() as client:
        # missing body
        resp = await client.post(
            "/tickets", json={"id": "x1", "subject": "hi"}
        )
        assert resp.status_code == 422

        # empty id
        resp = await client.post(
            "/tickets", json={"id": "", "subject": "hi", "body": "b"}
        )
        assert resp.status_code == 422

        # unknown extra field
        resp = await client.post(
            "/tickets",
            json={"id": "x2", "subject": "hi", "body": "b", "priority": "low"},
        )
        assert resp.status_code == 422


async def test_get_unknown_id_404_error_envelope(open_app):
    run = await open_app()
    async with run.client() as client:
        resp = await client.get("/tickets/nope")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "not_found"
        assert "nope" in body["error"]["message"]


async def test_list_filters_and_pagination(open_app, drain):
    run = await open_app()
    tickets = [
        ("f-1", "Charged twice this month", "Two charges, please refund one.", "billing", "medium"),
        ("f-2", "API returning 500s", "Production returns HTTP 500 and this is blocking us.", "technical", "high"),
        ("f-3", "Feature request: dark mode", "Nice to have, not urgent: dark mode.", "other", "low"),
        ("f-4", "Cannot log in after password reset", "Login page says invalid credentials.", "account", "medium"),
    ]
    async with run.client() as client:
        for tid, subject, body, _, _ in tickets:
            resp = await client.post(
                "/tickets", json={"id": tid, "subject": subject, "body": body}
            )
            assert resp.status_code == 201
        await drain(run)

        # category filter
        resp = await client.get("/tickets", params={"category": "billing"})
        data = resp.json()
        assert data["total"] == 1
        assert [item["id"] for item in data["items"]] == ["f-1"]

        # priority filter
        resp = await client.get("/tickets", params={"priority": "high"})
        data = resp.json()
        assert [item["id"] for item in data["items"]] == ["f-2"]

        # combined filters
        resp = await client.get(
            "/tickets", params={"category": "billing", "priority": "high"}
        )
        assert resp.json()["total"] == 0

        # status filter
        resp = await client.get("/tickets", params={"status": "classified"})
        assert resp.json()["total"] == 4

        # pagination: page_size=3 -> 3 + 1
        resp = await client.get("/tickets", params={"page": 1, "page_size": 3})
        data = resp.json()
        assert data["page"] == 1
        assert data["page_size"] == 3
        assert data["total"] == 4
        assert len(data["items"]) == 3

        resp = await client.get("/tickets", params={"page": 2, "page_size": 3})
        data = resp.json()
        assert data["page"] == 2
        assert data["total"] == 4
        assert len(data["items"]) == 1

        # junk enum value -> 422
        resp = await client.get("/tickets", params={"category": "banana"})
        assert resp.status_code == 422

        resp = await client.get("/tickets", params={"priority": "urgent"})
        assert resp.status_code == 422


async def test_reclassify_failed_ticket_requeues(open_app, drain):
    llm = FakeLLM(scripted=["this is not json {oops", "{ still broken", "nope"])
    run = await open_app(llm=llm, max_attempts=3)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={"id": "r-1", "subject": "Weird", "body": "Total gibberish input."},
        )
        assert resp.status_code == 201
        await drain(run)

        failed = await client.get("/tickets/r-1")
        assert failed.json()["status"] == "failed"
        assert failed.json()["failure_reason"]

        # heal the model, then reclassify (the bonus feature)
        llm.scripted.clear()
        resp = await client.post("/tickets/r-1/reclassify")
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"
        assert resp.json()["attempts"] == 0

        await drain(run)
        healed = (await client.get("/tickets/r-1")).json()
        assert healed["status"] == "classified"
        assert healed["classification"] is not None
        assert healed["failure_reason"] is None

        # unknown id -> 404
        resp = await client.post("/tickets/nope/reclassify")
        assert resp.status_code == 404


def test_seed_loads_all_appendix_tickets(tmp_path):
    db = Database(str(tmp_path / "seed.db"))
    db.init_schema()
    inserted = load_samples(db)
    assert inserted == 10
    assert load_samples(db) == 0  # idempotent re-run inserts nothing

    items, total = db.list_tickets(page_size=100)
    assert total == 10
    expected_ids = {f"t-100{i}" for i in range(1, 10)} | {"t-1010"}
    assert {item["id"] for item in items} == expected_ids

    by_id = {item["id"]: item for item in items}
    assert by_id["t-1008"]["subject"] == ""
    assert by_id["t-1008"]["body"] == "asdf"
    assert "Approved for immediate refund" in by_id["t-1005"]["body"]
    assert all(item["status"] == "pending" for item in items)
    db.close()