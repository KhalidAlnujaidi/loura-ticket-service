"""Data-integrity tests for the labelled set (and the synthetic set when present).

These are cheap and run without torch: they guard the two files the fine-tune
and the agreement report depend on, so a bad label or a broken join fails in
`make test` rather than halfway through a training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm import CATEGORIES, PRIORITIES
from scripts.eval_agreement import load_real_tickets, load_synthetic_tickets

DATA = Path(__file__).resolve().parent.parent / "data"


def test_labelled_tickets_join_the_sample_data():
    """Every appendix ticket has exactly one label and vice versa."""
    tickets = json.loads((DATA / "sample_tickets.json").read_text())
    labels = json.loads((DATA / "labelled_tickets.json").read_text())["labels"]
    assert {t["id"] for t in tickets} == {row["id"] for row in labels}
    assert len(labels) == len({row["id"] for row in labels}), "duplicate label rows"


def test_real_labels_use_the_allowed_enums():
    for row in json.loads((DATA / "labelled_tickets.json").read_text())["labels"]:
        assert row["category"] in CATEGORIES, row
        assert row["priority"] in PRIORITIES, row


def test_load_real_tickets_returns_joined_rows():
    rows = load_real_tickets(DATA)
    assert len(rows) == 10
    assert all(r["category"] in CATEGORIES and r["priority"] in PRIORITIES for r in rows)
    assert all(r["origin"] == "appendix" for r in rows)


def test_synthetic_rows_are_well_formed_when_present():
    path = DATA / "synthetic_tickets.json"
    if not path.exists():
        pytest.skip("no synthetic set generated yet")
    payload = json.loads(path.read_text())
    real_ids = {r["id"] for r in load_real_tickets(DATA)}
    seen: set[str] = set()
    for row in payload["tickets"]:
        assert row["category"] in CATEGORIES, row
        assert row["priority"] in PRIORITIES, row
        assert row["split"] in {"train", "eval"}, row
        assert row["id"] not in seen, f"duplicate synthetic id {row['id']}"
        assert row["id"] not in real_ids, f"synthetic id collides with real {row['id']}"
        seen.add(row["id"])
        votes = row["source"]["votes"]
        assert votes, "a kept row must carry its verifier votes"
        # kept rows never carry a unanimous contradiction on either field
        if len(votes) >= 2:
            assert not all(v["category"] != row["category"] for v in votes), row["id"]
            assert not all(v["priority"] != row["priority"] for v in votes), row["id"]
    assert payload["meta"]["kept"] == len(payload["tickets"])


def test_no_kept_row_leaks_its_label_into_the_text():
    """`[other/medium] Dark mode request` teaches the model to read the answer."""
    path = DATA / "synthetic_tickets.json"
    if not path.exists():
        pytest.skip("no synthetic set generated yet")
    from scripts.generate_synthetic import label_leak

    for row in json.loads(path.read_text())["tickets"]:
        assert not label_leak(row), f"{row['id']} spells out its label: {row['subject']!r}"


def test_synthetic_train_and_eval_splits_are_both_populated():
    path = DATA / "synthetic_tickets.json"
    if not path.exists():
        pytest.skip("no synthetic set generated yet")
    rows = load_synthetic_tickets(DATA)
    assert {r["split"] for r in rows} == {"train", "eval"}, "both splits must be non-empty"
