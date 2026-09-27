from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse

from .config import Settings
from .db import Database
from .llm import FakeLLM, LLM
from .schemas import Category, Priority, Status, TicketIn, TicketListOut, TicketOut
from .worker import ClassificationWorker


def build_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "fake":
        return FakeLLM()
    if settings.llm_backend == "laya":
        from .laya_llm import LayaLLM

        return LayaLLM()
    raise ValueError(f"unknown LOURA_LLM_BACKEND: {settings.llm_backend!r}")


def _not_found(ticket_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": "not_found",
                "message": f"ticket {ticket_id!r} not found",
            }
        },
    )


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.db_path)
    llm = llm or build_llm(settings)
    worker = ClassificationWorker(db, llm, settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        db.init_schema()
        warmup = getattr(llm, "warmup", None)
        if warmup is not None:
            # Load the checkpoint before accepting work: first boot may
            # download ~800MB, which must not eat into llm_timeout.
            await warmup()
        worker.start()
        try:
            yield
        finally:
            await worker.stop()
            db.close()

    app = FastAPI(
        title="Loura ticket classifier",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.llm = llm
    app.state.worker = worker

    @app.post("/tickets", response_model=TicketOut, status_code=201)
    async def create_ticket(payload: TicketIn, response: Response):
        row, created = db.create_ticket(payload.id, payload.subject, payload.body)
        if created:
            worker.enqueue(payload.id)
        else:
            # Idempotent by id: same object, no re-classify, 200 instead of 201.
            response.status_code = 200
        return row

    @app.get("/tickets/{ticket_id}", response_model=TicketOut)
    async def get_ticket(ticket_id: str):
        row = db.get_ticket(ticket_id)
        if row is None:
            return _not_found(ticket_id)
        return row

    @app.get("/tickets", response_model=TicketListOut)
    async def list_tickets(
        category: Category | None = Query(default=None),
        priority: Priority | None = Query(default=None),
        status: Status | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int | None = Query(default=None, ge=1, le=100),
    ):
        size = page_size or settings.default_page_size
        items, total = db.list_tickets(
            category=category,
            priority=priority,
            status=status,
            page=page,
            page_size=size,
        )
        return {
            "items": items,
            "page": page,
            "page_size": size,
            "total": total,
        }

    @app.post("/tickets/{ticket_id}/reclassify", response_model=TicketOut, status_code=202)
    async def reclassify_ticket(ticket_id: str):
        row = db.get_ticket(ticket_id)
        if row is None:
            return _not_found(ticket_id)
        db.reset_for_reclassify(ticket_id)
        worker.enqueue(ticket_id)
        return db.get_ticket(ticket_id)

    return app


app = create_app()