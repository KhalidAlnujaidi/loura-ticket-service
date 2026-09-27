from __future__ import annotations

import itertools
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.llm import FakeLLM
from app.main import create_app


@pytest.fixture
def settings_factory(tmp_path):
    counter = itertools.count()

    def factory(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "db_path": str(tmp_path / f"db-{next(counter)}.sqlite3"),
            "retry_base_delay": 0.0,
            "max_attempts": 3,
            "worker_count": 2,
            "llm_timeout": 2.0,
        }
        base.update(overrides)
        return Settings(**base)

    return factory


@dataclass
class RunningApp:
    app: Any
    settings: Settings
    _cm: Any
    _open: bool = field(default=True)

    async def stop(self) -> None:
        if self._open:
            self._open = False
            await self._cm.__aexit__(None, None, None)

    @asynccontextmanager
    async def client(self) -> AsyncIterator[AsyncClient]:
        transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def open_app(settings_factory):
    opened: list[RunningApp] = []

    async def opener(
        llm: FakeLLM | None = None,
        settings: Settings | None = None,
        **overrides: Any,
    ) -> RunningApp:
        settings = settings or settings_factory(**overrides)
        # Tests default to the deterministic FakeLLM regardless of Settings'
        # backend (laya): scriptable failures, stable labels, zero downloads.
        app = create_app(settings, llm=llm or FakeLLM())
        cm = app.router.lifespan_context(app)
        await cm.__aenter__()
        run = RunningApp(app=app, settings=settings, _cm=cm)
        opened.append(run)
        return run

    yield opener

    for run in reversed(opened):
        await run.stop()


@pytest.fixture
def drain():
    async def _drain(run: RunningApp, timeout: float = 10.0) -> None:
        await run.app.state.worker.wait_idle(timeout)

    return _drain