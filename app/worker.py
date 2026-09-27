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
                # _process does not raise on purpose; last resort so a loop
                # can never die and take its queue slot with it.
                logger.exception("[%s] unexpected error processing %s", name, ticket_id)
            finally:
                self._queue.task_done()
                self._scheduled.discard(ticket_id)
                self._in_flight -= 1
                self._refresh_idle()

    async def _process(self, ticket_id: str) -> None:
        """Drive one ticket to a terminal state: classified, failed or gone.

        Every failure is booked against the same attempt budget — model
        failures and internal ones alike — so a ticket can never strand as
        `pending` with its queue slot consumed (the old failure mode: any
        error outside the model-retry path dropped the id on the floor until
        a restart). The one exception is a store too broken to book against;
        that leaves the row for the startup recovery scan instead of spinning.
        """
        while not self._stopping:
            try:
                terminal = await self._attempt(ticket_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                terminal = await self._book_internal_failure(ticket_id, exc)
            if terminal:
                return

    async def _attempt(self, ticket_id: str) -> bool:
        """One classification attempt with its bookkeeping. True when terminal."""
        row = self.db.get_ticket(ticket_id)
        if row is None or row["status"] != "pending":
            return True
        if row["attempts"] >= self.settings.max_attempts:
            self.db.mark_failed(
                ticket_id, row["failure_reason"] or "max attempts exhausted"
            )
            return True
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
            return True

        attempts = self.db.record_failed_attempt(ticket_id, reason)
        return await self._budget_or_backoff(ticket_id, attempts, reason)

    async def _book_internal_failure(
        self, ticket_id: str, exc: Exception
    ) -> bool:
        """Book an infrastructure failure (store blip, unexpected bug) like any
        other attempt. Never raises: if the store cannot even book the failure,
        leave the row pending for startup recovery."""
        reason = f"internal error: {type(exc).__name__}"
        logger.exception("ticket %s: %s", ticket_id, reason)
        try:
            attempts = self.db.record_failed_attempt(ticket_id, reason)
            return await self._budget_or_backoff(ticket_id, attempts, reason)
        except Exception:
            logger.exception("ticket %s: could not book %s", ticket_id, reason)
            return True

    async def _budget_or_backoff(
        self, ticket_id: str, attempts: int, reason: str
    ) -> bool:
        """Shared tail of a booked failure: exhaust -> failed, else back off."""
        if attempts >= self.settings.max_attempts:
            self.db.mark_failed(ticket_id, reason)
            return True
        await asyncio.sleep(self._backoff(attempts))
        return False

    def _backoff(self, attempts: int) -> float:
        if self.settings.retry_base_delay <= 0:
            return 0.0
        delay = self.settings.retry_base_delay * (2 ** (attempts - 1))
        delay = min(delay, self.settings.retry_max_delay)
        return delay * (0.5 + random.random())
