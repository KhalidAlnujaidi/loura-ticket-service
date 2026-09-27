from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import Settings
from .db import Database

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_tickets.json"


def load_samples(db: Database, path: Path = SAMPLE_PATH) -> int:
    tickets = json.loads(path.read_text(encoding="utf-8"))
    inserted = 0
    for ticket in tickets:
        _, created = db.create_ticket(ticket["id"], ticket["subject"], ticket["body"])
        inserted += int(created)
    return inserted


def main() -> None:
    settings = Settings.from_env()
    db = Database(settings.db_path)
    db.init_schema()
    inserted = load_samples(db)
    items, total = db.list_tickets(page_size=1)
    print(
        f"seeded {inserted} new ticket(s) ({total} total) into {settings.db_path}"
    )
    db.close()


if __name__ == "__main__":
    sys.exit(main())