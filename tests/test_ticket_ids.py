"""Server-minted ticket ids: ``t-dd-mm-yy-hh-mm-XX`` (UTC), collision-retried.

These tests fail against the old contract (client-supplied id only, base36
UI-generated ids): they POST *without* an id and assert the server's format.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.main import generate_ticket_id

ID_RE = re.compile(r"^t-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-([a-z0-9]{2})$")


def _stamp_ts(match: re.Match) -> float:
    """UTC timestamp of the dd-mm-yy-hh-mm portion of a minted id."""
    dd, mm, yy, hh, minute = match.groups()[:5]
    dt = datetime.strptime(f"{dd}-{mm}-{yy} {hh}:{minute}", "%d-%m-%y %H:%M")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def test_generated_id_format_and_freshness():
    tid = generate_ticket_id()
    match = ID_RE.match(tid)
    assert match, tid
    age = datetime.now(timezone.utc).timestamp() - _stamp_ts(match)
    assert 0 <= age < 120, f"stamp not the current UTC minute: {tid}"


async def test_post_without_id_gets_minted_readable_id(open_app, drain):
    run = await open_app()
    async with run.client() as client:
        resp = await client.post(
            "/tickets", json={"subject": "Server should name me", "body": "refund my charge"}
        )
        assert resp.status_code == 201
        body = resp.json()
        match = ID_RE.match(body["id"])
        assert match, body["id"]
        # id stamp agrees with created_at (UTC; 90s tolerance covers a
        # minute-boundary crossing between mint and row insert)
        created = datetime.fromisoformat(body["created_at"]).timestamp()
        assert abs(created - _stamp_ts(match)) <= 90

        await drain(run)
        assert (await client.get(f"/tickets/{body['id']}")).json()["status"] == "classified"

        # explicit ids still pass through byte-for-byte (contract unchanged)
        resp = await client.post(
            "/tickets", json={"id": "custom-ID_01", "subject": "s", "body": "b"}
        )
        assert resp.status_code == 201
        assert resp.json()["id"] == "custom-ID_01"


async def test_post_with_null_id_also_mints(open_app):
    run = await open_app()
    async with run.client() as client:
        resp = await client.post(
            "/tickets", json={"id": None, "subject": "null id", "body": "b"}
        )
        assert resp.status_code == 201
        assert ID_RE.match(resp.json()["id"])


async def test_mint_collision_retries_instead_of_returning_foreign_row(
    open_app, monkeypatch
):
    import app.main as main_mod

    # Force the exact hazard: two mints in the same minute share a suffix.
    seq = iter(
        ["t-27-09-26-00-00-aa", "t-27-09-26-00-00-aa", "t-27-09-26-00-00-bb"]
    )
    monkeypatch.setattr(main_mod, "generate_ticket_id", lambda: next(seq))

    run = await open_app()
    async with run.client() as client:
        first = await client.post(
            "/tickets", json={"subject": "First caller", "body": "billing question"}
        )
        assert first.status_code == 201
        assert first.json()["id"] == "t-27-09-26-00-00-aa"

        second = await client.post(
            "/tickets", json={"subject": "Second caller", "body": "technical outage"}
        )
        # the clash must be retried away, never handed back as a 200 "duplicate"
        assert second.status_code == 201
        assert second.json()["id"] == "t-27-09-26-00-00-bb"
        assert second.json()["subject"] == "Second caller"

        assert (
            await client.get("/tickets/t-27-09-26-00-00-aa")
        ).json()["subject"] == "First caller"
        assert (
            await client.get("/tickets/t-27-09-26-00-00-bb")
        ).json()["subject"] == "Second caller"
