"""Tests for the UI pages, admin observability endpoints, and the optional
hosted (OpenAI-compatible) backend added on top of the take-home core."""

from __future__ import annotations

import logging
import os

import pytest

from app.config import Settings
from app.main import build_llm

# ---------------------------------------------------------------- UI pages


async def test_user_and_admin_pages_serve_html(open_app):
    run = await open_app()
    async with run.client() as client:
        resp = await client.get("/ui")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Loura" in resp.text and "Classify ticket" in resp.text

        resp = await client.get("/admin")
        assert resp.status_code == 200
        assert "Loura — admin" in resp.text
        assert "Live log" in resp.text


# ----------------------------------------------------------- admin endpoints


async def test_admin_logs_captures_records_with_cursor(open_app):
    run = await open_app()
    logging.getLogger("app.worker").warning("qa-log-marker-%s", run.settings.db_path[-12:])
    marker = f"qa-log-marker-{run.settings.db_path[-12:]}"
    async with run.client() as client:
        resp = await client.get("/admin/logs")
        assert resp.status_code == 200
        data = resp.json()
        texts = " ".join(item["text"] for item in data["items"])
        assert marker in texts
        assert data["next"] >= 1
        assert all(item["seq"] >= 1 for item in data["items"])

        # cursor semantics: asking after the head yields nothing new
        resp = await client.get("/admin/logs", params={"after": data["next"]})
        assert resp.json()["items"] == []


async def test_admin_metrics_reflect_workers_and_tickets(open_app, drain):
    run = await open_app(worker_count=3)
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={"id": "met-1", "subject": "Charged twice", "body": "two charges, refund one"},
        )
        assert resp.status_code == 201
        await drain(run)

        resp = await client.get("/admin/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workers"]["configured"] == 3
        assert data["workers"]["alive"] == 3
        assert data["workers"]["queue_depth"] == 0
        assert data["workers"]["in_flight"] == 0
        assert data["workers"]["idle"] is True
        assert data["tickets"]["classified"] == 1
        assert data["tickets"]["total"] == 1
        assert data["llm"]["backend"] == run.settings.llm_backend
        assert "server_time" in data


async def test_admin_token_gates_data_but_not_page(open_app):
    run = await open_app(admin_token="s3cret-token")
    async with run.client() as client:
        # page itself loads (it carries no data until the browser calls the API)
        assert (await client.get("/admin")).status_code == 200

        for path in ("/admin/logs", "/admin/metrics"):
            resp = await client.get(path)
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "unauthorized"

            resp = await client.get(path, headers={"X-Admin-Token": "wrong"})
            assert resp.status_code == 401

            resp = await client.get(path, headers={"X-Admin-Token": "s3cret-token"})
            assert resp.status_code == 200

        # public API untouched by the gate
        assert (await client.get("/tickets")).status_code == 200


def test_status_counts_helper(tmp_path):
    from app.db import Database

    db = Database(str(tmp_path / "counts.sqlite3"))
    db.init_schema()
    db.create_ticket("c-1", "s", "b")
    db.create_ticket("c-2", "s", "b")
    from app.llm import Classification  # local import keeps module top clean

    db.mark_classified("c-1", Classification(category="billing", priority="low", summary="refund asked"))
    db.mark_failed("c-2", "invalid model output: nope")
    counts = db.status_counts()
    assert counts == {"pending": 0, "classified": 1, "failed": 1, "total": 2}
    db.close()


# ------------------------------------------------------- hosted LLM backend


def test_build_llm_openai_requires_key():
    with pytest.raises(ValueError) as excinfo:
        build_llm(Settings(llm_backend="openai"))
    assert "LOURA_LLM_API_KEY" in str(excinfo.value)


def test_build_llm_openai_branch_and_defaults():
    llm = build_llm(
        Settings(
            llm_backend="openai",
            llm_api_key="sk-test",
            llm_base_url="https://api.example/v1/",
            llm_model="test-model",
        )
    )
    from app.hosted_llm import HostedLLM

    assert isinstance(llm, HostedLLM)
    assert llm.model == "test-model"
    assert llm.base_url == "https://api.example/v1"  # trailing slash stripped


def test_env_overrides_for_hosted_and_admin(monkeypatch):
    monkeypatch.setenv("LOURA_LLM_BACKEND", "openai")
    monkeypatch.setenv("LOURA_LLM_API_KEY", "sk-env")
    monkeypatch.setenv("LOURA_LLM_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("LOURA_LLM_MODEL", "some/model")
    monkeypatch.setenv("LOURA_ADMIN_TOKEN", "gate")
    s = Settings.from_env()
    assert (s.llm_backend, s.llm_api_key, s.llm_base_url, s.llm_model, s.admin_token) == (
        "openai",
        "sk-env",
        "https://router.example/v1",
        "some/model",
        "gate",
    )


async def test_hosted_complete_posts_chat_payload(monkeypatch):
    import httpx

    from app.hosted_llm import HostedLLM

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"category": "billing", "priority": "low", "summary": "refund asked"}'}}
                ]
            }

    async def fake_post(self, url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    llm = HostedLLM(api_key="sk-test", base_url="https://api.example/v1/", model="m1")
    out = await llm.complete("SYSTEM-PROMPT", "USER-PROMPT")
    assert '"category": "billing"' in out
    assert captured["url"] == "https://api.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "m1"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "SYSTEM-PROMPT"},
        {"role": "user", "content": "USER-PROMPT"},
    ]


async def test_hosted_errors_propagate_for_worker_retry(monkeypatch):
    import httpx

    from app.hosted_llm import HostedLLM

    async def fake_post(self, url, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    llm = HostedLLM(api_key="sk-test")
    with pytest.raises(httpx.ConnectError):
        await llm.complete("s", "u")


@pytest.mark.skipif(
    not os.getenv("LOURA_LLM_API_KEY"), reason="no LOURA_LLM_API_KEY set (optional live e2e)"
)
async def test_hosted_backend_end_to_end(open_app, drain):
    run = await open_app(
        llm=build_llm(Settings.from_env()), llm_timeout=30.0, worker_count=1
    )
    async with run.client() as client:
        resp = await client.post(
            "/tickets",
            json={
                "id": "hosted-1",
                "subject": "Charged twice this month",
                "body": "I see two charges of 49.00 on one subscription, please refund one.",
            },
        )
        assert resp.status_code == 201
        await drain(run, timeout=60.0)
        data = (await client.get("/tickets/hosted-1")).json()
        assert data["status"] == "classified"
        cls = data["classification"]
        assert cls["category"] == "billing"
        assert 1 <= len(cls["summary"]) <= 300