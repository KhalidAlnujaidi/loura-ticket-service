"""Regression tests for the QA findings (config validation + worker lifecycle).

Findings covered (see QA_REPORT.md):
  QA-1  LOURA_WORKERS <= 0 disabled all classification silently
  QA-2  LOURA_MAX_ATTEMPTS = 0 failed every ticket without calling the model
  QA-3  a non-positive LOURA_LLM_TIMEOUT failed every ticket before the call
  QA-4  ClassificationWorker.start() was not idempotent
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import FakeLLM
from app.worker import ClassificationWorker

REJECTED_INT = [
    ("LOURA_WORKERS", "0"),
    ("LOURA_WORKERS", "-1"),
    ("LOURA_MAX_ATTEMPTS", "0"),
    ("LOURA_MAX_ATTEMPTS", "-3"),
]

REJECTED_NUMERIC = [
    ("LOURA_WORKERS", "abc"),
    ("LOURA_MAX_ATTEMPTS", "1.5"),
    ("LOURA_LLM_TIMEOUT", "xyz"),
    ("LOURA_LLM_TIMEOUT", "0"),
    ("LOURA_LLM_TIMEOUT", "-5"),
]


@pytest.mark.parametrize("name,raw", REJECTED_INT + REJECTED_NUMERIC)
def test_invalid_env_overrides_are_rejected(monkeypatch, name, raw):
    """QA-1/2/3: bad numeric config must fail loudly at parse time."""
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError) as excinfo:
        Settings.from_env()
    # The message must name the offending variable so an operator can act on it.
    assert name in str(excinfo.value)


@pytest.mark.parametrize(
    "name,raw,attr,expected",
    [
        ("LOURA_WORKERS", "1", "worker_count", 1),
        ("LOURA_WORKERS", "8", "worker_count", 8),
        ("LOURA_MAX_ATTEMPTS", "1", "max_attempts", 1),
        ("LOURA_MAX_ATTEMPTS", "10", "max_attempts", 10),
        ("LOURA_LLM_TIMEOUT", "0.5", "llm_timeout", 0.5),
        ("LOURA_LLM_TIMEOUT", "30", "llm_timeout", 30.0),
    ],
)
def test_valid_env_overrides_still_parse(monkeypatch, name, raw, attr, expected):
    """The validation must not reject legitimate values."""
    monkeypatch.setenv(name, raw)
    assert getattr(Settings.from_env(), attr) == expected


def test_defaults_are_unchanged_without_env(monkeypatch):
    for var in (
        "LOURA_WORKERS",
        "LOURA_MAX_ATTEMPTS",
        "LOURA_LLM_TIMEOUT",
        "LOURA_DB_PATH",
        "LOURA_LLM_BACKEND",
        "LOURA_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings.from_env()
    assert (settings.worker_count, settings.max_attempts, settings.llm_timeout) == (
        2,
        3,
        15.0,
    )
    # The laya backend must keep serving the base checkpoint unless told otherwise.
    assert settings.laya_model == "convaiinnovations/laya"


def test_llm_model_env_selects_the_laya_checkpoint(monkeypatch):
    """LOURA_LLM_MODEL picks the served laya checkpoint (an HF id or a local dir)."""
    monkeypatch.setenv("LOURA_LLM_MODEL", "checkpoints/loura-tickets-v1")
    settings = Settings.from_env()
    assert settings.laya_model == "checkpoints/loura-tickets-v1"
    assert settings.llm_model == "checkpoints/loura-tickets-v1"


def test_build_llm_wires_the_laya_checkpoint(settings_factory):
    """build_llm must hand the checkpoint to LayaLLM (construction loads no weights)."""
    from app.main import build_llm

    llm = build_llm(settings_factory(llm_backend="laya", laya_model="checkpoints/x"))
    assert getattr(llm, "model_id") == "checkpoints/x"  # LayaLLM carries the checkpoint


def test_worker_start_rejects_zero_workers_directly(settings_factory):
    """QA-1 defence in depth: a directly-built Settings cannot yield 0 loops."""
    from app.db import Database

    settings = settings_factory(worker_count=0)
    db = Database(settings.db_path)
    db.init_schema()
    worker = ClassificationWorker(db, FakeLLM(), settings)
    with pytest.raises(RuntimeError) as excinfo:
        worker.start()
    assert "worker_count" in str(excinfo.value)
    assert worker._tasks == []
    db.close()


async def test_start_is_idempotent_and_stop_cancels_all(settings_factory):
    """QA-4: a second start() must not orphan the first set of loops."""
    from app.db import Database

    settings = settings_factory(worker_count=2)
    db = Database(settings.db_path)
    db.init_schema()
    worker = ClassificationWorker(db, FakeLLM(), settings)

    worker.start()
    first = list(worker._tasks)
    assert len(first) == 2

    worker.start()  # second call must be a no-op
    assert worker._tasks == first, "start() replaced the task list"
    assert all(not task.done() for task in worker._tasks)

    await worker.stop()
    assert worker._tasks == []
    assert all(task.cancelled() or task.done() for task in first)
    db.close()


async def test_start_restarts_after_stop(settings_factory):
    """A restart cycle (stop then start) must work again."""
    from app.db import Database

    settings = settings_factory(worker_count=2)
    db = Database(settings.db_path)
    db.init_schema()
    worker = ClassificationWorker(db, FakeLLM(), settings)

    worker.start()
    await worker.stop()
    worker.start()
    assert len(worker._tasks) == 2
    assert all(not task.done() for task in worker._tasks)
    await worker.stop()
    db.close()


async def test_all_tickets_classified_exactly_once_under_concurrency(open_app, drain):
    """Concurrency regression: N tickets across M workers, each done once."""
    total = 40
    run = await open_app(worker_count=8)
    async with run.client() as client:
        for i in range(total):
            resp = await client.post(
                "/tickets",
                json={
                    "id": f"cc-{i:03d}",
                    "subject": "Charged twice",
                    "body": "Two charges on one subscription, please refund the invoice.",
                },
            )
            assert resp.status_code == 201

        await drain(run, timeout=30.0)

        listing = (await client.get("/tickets", params={"page_size": 100})).json()
        assert listing["total"] == total

        items = listing["items"]
        assert all(item["status"] == "classified" for item in items), [
            (i["id"], i["status"]) for i in items if i["status"] != "classified"
        ]
        # exactly-once accounting: no double-processing, no lost work
        assert all(item["attempts"] == 1 for item in items), [
            (i["id"], i["attempts"]) for i in items if i["attempts"] != 1
        ]
        assert all(item["classification"] is not None for item in items)
        assert run.app.state.llm.calls == total

        # worker bookkeeping must be fully drained
        worker = run.app.state.worker
        assert worker._scheduled == set()
        assert worker._queue.qsize() == 0
        assert worker._in_flight == 0
        assert worker._idle.is_set()

@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_timeout_is_rejected(bad, monkeypatch):
    """nan/inf parse as floats and sail past a naive <= 0 check.

    nan then hangs asyncio.wait_for forever; inf fires instantly. Both must
    die at the env edge with the offending variable named.
    """
    monkeypatch.setenv("LOURA_LLM_TIMEOUT", bad)
    with pytest.raises(ValueError, match="LOURA_LLM_TIMEOUT"):
        Settings.from_env()

