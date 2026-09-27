"""Log ring buffer + admin observability endpoints.

Captures everything this process logs (worker retries, uvicorn request lines)
into a bounded in-memory buffer so the admin UI can show recent activity with
zero logging infrastructure. The handler is attached for the app's lifespan
and detached on shutdown, so test suites that open many apps never accumulate
handlers.

Admin data endpoints are gated by LOURA_ADMIN_TOKEN when it is set (open when
unset, matching the take-home's no-auth scope). The /admin page itself is not
gated -- it ships no data until the browser calls the gated endpoints.
"""

from __future__ import annotations

import hmac
import logging
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

_BUFFER_SIZE = 500
_PAGE_MAX = 200


class LogBufferHandler(logging.Handler):
    """Bounded ring buffer of formatted log records with sequence numbers."""

    def __init__(self, maxlen: int = _BUFFER_SIZE) -> None:
        super().__init__()
        self._sync = threading.Lock()
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0
        self._last: logging.LogRecord | None = None
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        # The same handler instance sits on root AND the uvicorn loggers; under
        # some uvicorn configs a record can reach us twice back-to-back (direct
        # emit, then again via propagation). Identity check drops the echo.
        if record is self._last:
            return
        self._last = record
        try:
            text = self.format(record)
            if record.exc_info:
                text += "\n" + "".join(traceback.format_exception(*record.exc_info))
            with self._sync:
                self._seq += 1
                self._records.append(
                    {
                        "seq": self._seq,
                        "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                        .isoformat(timespec="milliseconds"),
                        "level": record.levelname,
                        "logger": record.name,
                        "text": text,
                    }
                )
        except Exception:  # logging must never raise into the app
            self.handleError(record)

    def since(self, after: int) -> tuple[list[dict[str, Any]], int]:
        """Records with seq > after (capped), plus the current head seq."""
        after = max(0, after)
        with self._sync:
            items = [r for r in self._records if r["seq"] > after]
            return items[-_PAGE_MAX:], self._seq


def _targets() -> list[logging.Logger]:
    # root catches app.*; uvicorn's loggers default to propagate=False, so
    # request lines need their own attachment to show up.
    return [
        logging.getLogger(),
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.access"),
    ]


def attach(handler: LogBufferHandler) -> None:
    for logger in _targets():
        if handler not in logger.handlers:
            logger.addHandler(handler)


def detach(handler: LogBufferHandler) -> None:
    for logger in _targets():
        logger.removeHandler(handler)


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": "unauthorized",
                "message": "missing or wrong X-Admin-Token (set LOURA_ADMIN_TOKEN to enable the gate)",
            }
        },
    )


def build_admin_router() -> APIRouter:
    router = APIRouter(prefix="/admin", include_in_schema=False)

    def authorized(request: Request) -> bool:
        token = request.app.state.settings.admin_token
        if not token:
            return True
        supplied = str(request.headers.get("x-admin-token", ""))
        return hmac.compare_digest(supplied, token)

    @router.get("/logs")
    async def admin_logs(request: Request, after: int = 0):
        """Newest log records after the ?after= cursor (poll-friendly)."""
        if not authorized(request):
            return _unauthorized()
        items, head = request.app.state.log_buffer.since(after)
        return {"items": items, "next": head, "count": len(items)}

    @router.get("/metrics")
    async def admin_metrics(request: Request):
        """Worker/queue snapshot + ticket counts + backend info."""
        if not authorized(request):
            return _unauthorized()
        state = request.app.state
        return {
            "workers": state.worker.snapshot(),
            "tickets": state.db.status_counts(),
            "llm": {
                "backend": state.settings.llm_backend,
                "max_attempts": state.settings.max_attempts,
            },
            "guard": {
                "detector": "heuristic",
                "injection_flags": state.injection_flags,
            },
            "server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    return router
