from __future__ import annotations

import json

import pytest

from app.db import Database
from app.llm import FakeLLM, ModelOutputError, validate_classification

ALLOWED_CATEGORIES = {"billing", "technical", "account", "other"}
ALLOWED_PRIORITIES = {"low", "medium", "high"}

SAMPLES = {
    "t-1001": (
        "Charged twice this month",
        "Hi, I see two charges of 49.00 on my card statement dated the 3rd and the 4th. "
        "I only have one subscription. Can you refund one of them?",
    ),
    "t-1003": (
        "API returning 500s since this morning",
        "Our production integration started getting HTTP 500 from /v2/export around 08:10 UTC. "
        "About 30% of requests. This is blocking our nightly job. Request id example: 7f3a-91c2.",
    ),
    "t-1006": (
        "Feature request: dark mode",
        "Would love a dark mode option in the dashboard. Not urgent, just a nice to have.",
    ),
}


async def test_happy_path_sample_classifications(open_app, drain):
    run = await open_app()
    async with run.client() as client:
        for tid, (subject, body) in SAMPLES.items():
            resp = await client.post(
                "/tickets", json={"id": tid, "subject": subject, "body": body}
            )
            assert resp.status_code == 201
        await drain(run)

        first = (await client.get("/tickets/t-1001")).json()
        assert first["status"] == "classified"
        assert first["attempts"] == 1
        assert first["classification"]["category"] == "billing"

        third = (await client.get("/tickets/t-1003")).json()
        assert third["classification"]["category"] == "technical"
        assert third["classification"]["priority"] == "high"

        sixth = (await client.get("/tickets/t-1006")).json()
        assert sixth["classification"]["category"] == "other"
        assert sixth["classification"]["priority"] == "low"

        for tid in SAMPLES:
            cls = (await client.get(f"/tickets/{tid}")).json()["classification"]
            assert cls["category"] in ALLOWED_CATEGORIES
            assert cls["priority"] in ALLOWED_PRIORITIES
            assert cls["summary"].strip()


async def test_always_malformed_output_retries_then_fails(open_app, drain):
    llm = FakeLLM(
        scripted=[
            "this is not json {oops",
            '{"category": "billing", ',
            "--garbage--",
        ]
    )
    run = await open_app(llm=llm, max_attempts=3)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={"id": "m-1", "subject": "Odd", "body": "Totally unparseable instance."},
        )
        assert resp.status_code == 201
        await drain(run)

        data = (await client.get("/tickets/m-1")).json()
        assert data["status"] == "failed"
        assert data["attempts"] == 3
        assert data["failure_reason"]
        assert data["failure_reason"].startswith("invalid model output")
        assert data["classification"] is None

    # raw garbage was never written to the store as classification fields
    db = Database(run.settings.db_path)
    row = db.get_ticket("m-1")
    assert row["status"] == "failed"
    assert row["classification"] is None
    db.close()


async def test_out_of_range_enum_never_stored(open_app, drain):
    bad = '{"category": "urgent_bug", "priority": "critical", "summary": "looks fine"}'
    llm = FakeLLM(scripted=[bad, bad, bad])
    run = await open_app(llm=llm, max_attempts=3)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={"id": "m-2", "subject": "X", "body": "Text."},
        )
        assert resp.status_code == 201
        await drain(run)

        data = (await client.get("/tickets/m-2")).json()
        assert data["status"] == "failed"
        assert data["attempts"] == 3
        assert data["classification"] is None
        assert "invalid model output" in data["failure_reason"]

    db = Database(run.settings.db_path)
    row = db.get_ticket("m-2")
    assert row["status"] == "failed"
    assert row["classification"] is None
    db.close()


async def test_llm_timeout_counts_as_failed_attempt(open_app, drain):
    good = '{"category": "billing", "priority": "low", "summary": "refund asked"}'
    llm = FakeLLM(
        scripted=[("sleep", 0.5, good), ("sleep", 0.5, good), ("sleep", 0.5, good)]
    )
    run = await open_app(llm=llm, llm_timeout=0.05, max_attempts=3)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={"id": "m-3", "subject": "Slow", "body": "Please respond slowly."},
        )
        assert resp.status_code == 201
        await drain(run)

        data = (await client.get("/tickets/m-3")).json()
        assert data["status"] == "failed"
        assert data["attempts"] == 3
        assert "timeout" in data["failure_reason"]
        assert data["classification"] is None


async def test_markdown_fenced_json_is_extracted(open_app, drain):
    fenced = (
        "```json\n"
        '{"category": "account", "priority": "medium", "summary": "lost password after reset"}'
        "\n```"
    )
    llm = FakeLLM(scripted=[fenced])
    run = await open_app(llm=llm)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={
                "id": "m-4",
                "subject": "Cannot log in",
                "body": "Password reset did not work.",
            },
        )
        assert resp.status_code == 201
        await drain(run)

        data = (await client.get("/tickets/m-4")).json()
        assert data["status"] == "classified"
        assert data["classification"]["category"] == "account"
        assert data["attempts"] == 1


# ------------------------- pillar 4: schema-edge rejection (validation gate)


def test_summary_length_boundary_is_enforced():
    base = {"category": "billing", "priority": "low"}
    assert validate_classification(
        json.dumps({**base, "summary": "x" * 300})
    ).summary == "x" * 300  # the documented limit itself must pass
    with pytest.raises(ModelOutputError):
        validate_classification(json.dumps({**base, "summary": "x" * 301}))


def test_model_extra_keys_rejected():
    raw = json.dumps(
        {
            "category": "billing",
            "priority": "low",
            "summary": "refund asked",
            "confidence": 0.99,  # extra=forbid: no unmodeled keys stored
        }
    )
    with pytest.raises(ModelOutputError):
        validate_classification(raw)


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_or_whitespace_summary_rejected(bad):
    raw = json.dumps({"category": "billing", "priority": "low", "summary": bad})
    with pytest.raises(ModelOutputError):
        validate_classification(raw)