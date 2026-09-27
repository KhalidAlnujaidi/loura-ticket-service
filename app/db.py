from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .schemas import Classification

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'classified', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    failure_reason TEXT,
    category TEXT
        CHECK (category IS NULL OR category IN ('billing', 'technical', 'account', 'other')),
    priority TEXT
        CHECK (priority IS NULL OR priority IN ('low', 'medium', 'high')),
    summary TEXT,
    created_at TEXT NOT NULL,
    classified_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize(row: sqlite3.Row) -> dict[str, Any]:
    classification = None
    if row["status"] == "classified" and row["category"] is not None:
        classification = {
            "category": row["category"],
            "priority": row["priority"],
            "summary": row["summary"],
        }
    return {
        "id": row["id"],
        "subject": row["subject"],
        "body": row["body"],
        "status": row["status"],
        "attempts": row["attempts"],
        "failure_reason": row["failure_reason"],
        "classification": classification,
        "created_at": row["created_at"],
        "classified_at": row["classified_at"],
    }


class Database:
    """Thin synchronous SQLite wrapper, serialized behind a lock (WAL, file-backed)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=3000")
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    def init_schema(self) -> None:
        with self._lock:
            self._connect()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def create_ticket(self, ticket_id: str, subject: str, body: str) -> tuple[dict[str, Any], bool]:
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "INSERT INTO tickets (id, subject, body, status, attempts, created_at) "
                "VALUES (?, ?, ?, 'pending', 0, ?) ON CONFLICT(id) DO NOTHING",
                (ticket_id, subject, body, _now()),
            )
            conn.commit()
            created = cur.rowcount == 1
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return serialize(row), created

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return serialize(row) if row else None

    def list_tickets(
        self,
        *,
        category: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        where: list[str] = []
        params: list[object] = []
        for column, value in (("category", category), ("priority", priority), ("status", status)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            conn = self._connect()
            total = conn.execute(
                f"SELECT COUNT(*) FROM tickets {clause}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM tickets {clause} ORDER BY created_at ASC, id ASC LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
        return [serialize(row) for row in rows], total

    def status_counts(self) -> dict[str, int]:
        """Counts per status for GET /admin/metrics (always includes total)."""
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT status, COUNT(*) AS c FROM tickets GROUP BY status"
            ).fetchall()
        counts = {"pending": 0, "classified": 0, "failed": 0, "total": 0}
        for row in rows:
            counts[row["status"]] = row["c"]
            counts["total"] += row["c"]
        return counts

    def pending_ids(self) -> list[str]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT id FROM tickets WHERE status = 'pending' ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [row["id"] for row in rows]

    def record_failed_attempt(self, ticket_id: str, reason: str) -> int:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "UPDATE tickets SET attempts = attempts + 1, failure_reason = ? "
                "WHERE id = ? AND status = 'pending' RETURNING attempts",
                (reason, ticket_id),
            ).fetchone()
            conn.commit()
        return row["attempts"] if row else 0

    def mark_classified(self, ticket_id: str, classification: Classification) -> bool:
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "UPDATE tickets SET status = 'classified', attempts = attempts + 1, "
                "category = ?, priority = ?, summary = ?, failure_reason = NULL, classified_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (
                    classification.category,
                    classification.priority,
                    classification.summary,
                    _now(),
                    ticket_id,
                ),
            )
            conn.commit()
            return cur.rowcount == 1

    def mark_failed(self, ticket_id: str, reason: str) -> bool:
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "UPDATE tickets SET status = 'failed', failure_reason = ? "
                "WHERE id = ? AND status = 'pending'",
                (reason, ticket_id),
            )
            conn.commit()
            return cur.rowcount == 1

    def reset_for_reclassify(self, ticket_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "UPDATE tickets SET status = 'pending', attempts = 0, failure_reason = NULL, "
                "category = NULL, priority = NULL, summary = NULL, classified_at = NULL "
                "WHERE id = ?",
                (ticket_id,),
            )
            conn.commit()
            return cur.rowcount == 1