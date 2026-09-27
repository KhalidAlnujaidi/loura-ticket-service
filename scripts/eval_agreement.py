"""Agreement report: run a backend over the labelled tickets and compare to gold.

The README's "future work" item, built as the harness the fine-tune needs:
a labelled set (data/labelled_tickets.json + data/sample_tickets.json, plus
data/synthetic_tickets.json when present) goes through the *serving path*
(app.laya_llm.LayaLLM -> the same validation gate the worker uses) and the
report records category/priority agreement per ticket.

Usage:
  python -m scripts.eval_agreement --backend laya
  python -m scripts.eval_agreement --backend laya --model checkpoints/loura-tickets-v1
  python -m scripts.eval_agreement --backend fake --out reports/fake.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_rev() -> str | None:
    """Short HEAD sha for the report, or None when not in a repo / git absent."""
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


sys.path.insert(0, str(ROOT))

from app.llm import FakeLLM, ModelOutputError, validate_classification  # noqa: E402
from app.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

DATA = ROOT / "data"


def load_real_tickets(data_dir: Path = DATA) -> list[dict]:
    """Join data/sample_tickets.json with data/labelled_tickets.json on id."""
    tickets = json.loads((data_dir / "sample_tickets.json").read_text())
    labels = json.loads((data_dir / "labelled_tickets.json").read_text())["labels"]
    by_id = {row["id"]: row for row in labels}
    out = []
    for t in tickets:
        label = by_id.get(t["id"])
        if label is None:
            raise SystemExit(f"labelled_tickets.json has no label for {t['id']}")
        out.append({**t, "category": label["category"], "priority": label["priority"],
                    "origin": "appendix", "split": "real"})
    missing = set(by_id) - {t["id"] for t in tickets}
    if missing:
        raise SystemExit(f"labels for unknown tickets: {sorted(missing)}")
    return out


def load_synthetic_tickets(data_dir: Path = DATA) -> list[dict]:
    path = data_dir / "synthetic_tickets.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [{**t, "origin": "synthetic", "split": t.get("split", "train")}
            for t in payload["tickets"]]


async def classify_all(llm, tickets: list[dict]) -> list[dict]:
    rows = []
    for t in tickets:
        started = time.perf_counter()
        pred, error = None, None
        try:
            raw = await llm.complete(SYSTEM_PROMPT, build_user_prompt(t["subject"], t["body"]))
            cls = validate_classification(raw)
            pred = {"category": cls.category, "priority": cls.priority, "summary": cls.summary}
        except ModelOutputError as exc:
            error = f"invalid model output: {exc}"
        except Exception as exc:  # noqa: BLE001 - a report must record any backend failure
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "id": t["id"], "origin": t["origin"], "split": t["split"],
            "gold": {"category": t["category"], "priority": t["priority"]},
            "pred": pred, "error": error,
            "category_ok": bool(pred and pred["category"] == t["category"]),
            "priority_ok": bool(pred and pred["priority"] == t["priority"]),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    errors = [r for r in rows if r["error"]]
    cat = sum(r["category_ok"] for r in rows)
    pri = sum(r["priority_ok"] for r in rows)
    both = sum(r["category_ok"] and r["priority_ok"] for r in rows)
    lat = sorted(r["latency_ms"] for r in rows)
    per_cat: dict[str, dict] = {}
    for r in rows:
        cell = per_cat.setdefault(r["gold"]["category"], {"n": 0, "category_ok": 0, "priority_ok": 0})
        cell["n"] += 1
        cell["category_ok"] += int(r["category_ok"])
        cell["priority_ok"] += int(r["priority_ok"])
    per_pri: dict[str, dict] = {}
    for r in rows:
        cell = per_pri.setdefault(r["gold"]["priority"], {"n": 0, "category_ok": 0, "priority_ok": 0})
        cell["n"] += 1
        cell["category_ok"] += int(r["category_ok"])
        cell["priority_ok"] += int(r["priority_ok"])
    return {
        "n": n, "errors": len(errors),
        "category_ok": cat, "priority_ok": pri, "both_ok": both,
        "category_agreement": round(cat / n, 4) if n else None,
        "priority_agreement": round(pri / n, 4) if n else None,
        "both_agreement": round(both / n, 4) if n else None,
        "latency_ms": {"median": lat[len(lat) // 2] if lat else None,
                       "max": lat[-1] if lat else None},
        "by_category": per_cat, "by_priority": per_pri,
    }


def markdown(rows: list[dict], totals: dict) -> str:
    lines = [
        "| id | origin | gold cat/pri | pred cat/pri | cat | pri | ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pred = r["pred"]
        lines.append(
            f"| {r['id']} | {r['origin']} | {r['gold']['category']}/{r['gold']['priority']} "
            f"| {(pred['category'] + '/' + pred['priority']) if pred else r['error'][:40]} "
            f"| {'OK' if r['category_ok'] else 'MISS'} | {'OK' if r['priority_ok'] else 'MISS'} "
            f"| {r['latency_ms']:.0f} |"
        )
    lines.append("")
    lines.append(
        f"**{totals['n']} tickets**: category {totals['category_ok']}/{totals['n']} "
        f"({totals['category_agreement']:.2f}), priority {totals['priority_ok']}/{totals['n']} "
        f"({totals['priority_agreement']:.2f}), both {totals['both_ok']}/{totals['n']}, "
        f"errors {totals['errors']}, median {totals['latency_ms']['median']:.0f} ms."
    )
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    tickets = load_real_tickets(data_dir) + load_synthetic_tickets(data_dir)
    if args.only == "real":
        tickets = [t for t in tickets if t["origin"] == "appendix"]
    elif args.only == "synthetic":
        tickets = [t for t in tickets if t["origin"] == "synthetic"]
    if args.synthetic_split:
        tickets = [t for t in tickets if t["origin"] == "appendix" or t["split"] == args.synthetic_split]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        tickets = [t for t in tickets if t["id"] in wanted]
    if not tickets:
        print("no tickets selected", file=sys.stderr)
        return 2

    if args.backend == "fake":
        llm = FakeLLM()
        model_label = "FakeLLM (keyword scorer)"
    elif args.backend == "laya":
        from app.laya_llm import LayaLLM

        llm = LayaLLM(model_id=args.model)
        model_label = args.model
        await llm.warmup()
    else:
        print(f"unknown backend {args.backend!r}", file=sys.stderr)
        return 2

    rows = await classify_all(llm, tickets)
    totals = summarize(rows)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": args.backend,
        "model": model_label,
        "git_rev": await asyncio.to_thread(git_rev),
        "selection": {"only": args.only, "synthetic_split": args.synthetic_split, "ids": args.ids},
        "totals": totals,
        "per_ticket": rows,
    }

    table = markdown(rows, totals)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        out.with_suffix(".md").write_text(table + "\n")
        print(f"\nreport -> {out} (+ {out.with_suffix('.md').name})")
    return 0 if totals["errors"] == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a backend over the labelled tickets and compare to gold labels.")
    ap.add_argument("--backend", default="laya", choices=["laya", "fake"])
    ap.add_argument("--model", default="convaiinnovations/laya",
                    help="model id or local checkpoint dir (backend=laya)")
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--only", choices=["real", "synthetic"], help="restrict to one origin")
    ap.add_argument("--synthetic-split", choices=["train", "eval"],
                    help="restrict synthetic rows to one split")
    ap.add_argument("--ids", help="comma-separated ticket ids")
    ap.add_argument("--out", help="write JSON report here (a .md lands next to it)")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
