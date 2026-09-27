from __future__ import annotations

import asyncio
import logging
import random

from .config import Settings
from .db import Database
from .llm import LLM, reason_for, validate_classification
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class ClassificationWorker:
    """In-process asyncio queue + N worker loops.

    The queue is disposable: startup recovery re-enqueues every row still
    'pending' in SQLite, so a restart loses nothing but in-flight progress.
    A mid-flight cancellation leaves the row pending (attempt counting only
    happens when an attempt finishes), so work is at-least-once.
    """

    def __init__(self, db: Database, llm: LLM, settings: Settings) -> None:
        self.db = db
        self.llm = llm
        self.settings = settings
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._scheduled: set[str] = set()
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    def start(self) -> None:
        # Idempotent: calling start() twice must not orphan the first set of
        # loops (they would keep running and be unreachable by stop()).
        if self._tasks:
            self._tasks = [task for task in self._tasks if not task.done()]
            if self._tasks:
                return
        if self.settings.worker_count < 1:
            # Settings.from_env() rejects this too; this guards the path where
            # a Settings is constructed directly. Zero loops means the service
            # accepts traffic and never classifies anything.
            raise RuntimeError(
                f"worker_count must be >= 1, got {self.settings.worker_count}"
            )
        self._stopping = False
        for ticket_id in self.db.pending_ids():
            self.enqueue(ticket_id)
        self._tasks = [
            asyncio.create_task(self._loop(name=f"w{i}"))
            for i in range(self.settings.worker_count)
        ]

    async def stop(self) -> None:
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def enqueue(self, ticket_id: str) -> bool:
        if ticket_id in self._scheduled:
            return False
        self._scheduled.add(ticket_id)
        self._queue.put_nowait(ticket_id)
        self._idle.clear()
        return True

    async def wait_idle(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self._idle.wait(), timeout)

    def snapshot(self) -> dict[str, object]:
        """Point-in-time worker stats for GET /admin/metrics.

        Runs on the event loop like everything else, so plain reads are safe.
        """
        return {
            "configured": self.settings.worker_count,
            "alive": sum(not task.done() for task in self._tasks),
            "queue_depth": self._queue.qsize(),
            "in_flight": self._in_flight,
            "scheduled": len(self._scheduled),
            "idle": self._idle.is_set(),
        }

    def _refresh_idle(self) -> None:
        if self._queue.empty() and self._in_flight == 0:
            self._idle.set()
        else:
            self._idle.clear()

    async def _loop(self, name: str) -> None:
        while True:
            ticket_id = await self._queue.get()
            self._in_flight += 1
            self._refresh_idle()
            try:
                await self._process(ticket_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[%s] unexpected error processing %s", name, ticket_id)
            finally:
                self._queue.task_done()
                self._scheduled.discard(ticket_id)
                self._in_flight -= 1
                self._refresh_idle()

    async def _process(self, ticket_id: str) -> None:
        while not self._stopping:
            row = self.db.get_ticket(ticket_id)
            if row is None or row["status"] != "pending":
                return
            if row["attempts"] >= self.settings.max_attempts:
                self.db.mark_failed(
                    ticket_id, row["failure_reason"] or "max attempts exhausted"
                )
                return
            prompt = build_user_prompt(row["subject"], row["body"])
            raw = ""
            try:
                raw = await asyncio.wait_for(
                    self.llm.complete(SYSTEM_PROMPT, prompt),
                    timeout=self.settings.llm_timeout,
                )
                classification = validate_classification(raw)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                reason = f"llm timeout after {self.settings.llm_timeout}s"
                logger.warning("ticket %s: %s", ticket_id, reason)
            except Exception as exc:
                reason = reason_for(exc)
                logger.warning(
                    "ticket %s attempt %d failed: %s | raw=%.200s",
                    ticket_id,
                    row["attempts"] + 1,
                    reason,
                    raw,
                )
            else:
                self.db.mark_classified(ticket_id, classification)
                return

            attempts = self.db.record_failed_attempt(ticket_id, reason)
            if attempts >= self.settings.max_attempts:
                self.db.mark_failed(ticket_id, reason)
                return
            await asyncio.sleep(self._backoff(attempts))

    def _backoff(self, attempts: int) -> float:
        if self.settings.retry_base_delay <= 0:
            return 0.0
        delay = self.settings.retry_base_delay * (2 ** (attempts - 1))
        delay = min(delay, self.settings.retry_max_delay)
        return delay * (0.5 + random.random())