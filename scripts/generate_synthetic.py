"""Generate synthetic support tickets with the free-tier pool, cross-verified.

Design (see the free-tier-subagent-routing skill):
  * Models come from the live status file written by the pool probe
    (`~/.cache/or-swarm-status.json`) or from `--models`; a model that is not
    verified alive is never routed to.
  * Generation is stratified over the (category, priority) grid and asks for a
    rotating mix of styles, including the quirks the real appendix tickets have
    (blank subject, nonsense text, forwarded quote chains, injection attempts,
    multi-issue bodies, explicit vs implicit urgency).
  * Every generated ticket is then labelled by a DIFFERENT model with the
    exact prompt the service uses (app.prompts.SYSTEM_PROMPT + <ticket> XML).
    Tickets where the independent label disagrees with the requested one are
    dropped to data/synthetic_dropped.json for review, not silently trained on.
  * Near-duplicates of the real exemplars (difflib ratio > 0.8) are dropped:
    a copied ticket would leak a real ticket into the synthetic set.

Usage:
  python -m scripts.generate_synthetic --per-cell 8 --k 3
  python -m scripts.generate_synthetic --models m1,m2 --per-cell 2 --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from scripts.eval_agreement import load_real_tickets  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
STATUS = Path.home() / ".cache" / "or-swarm-status.json"
PROMPT_VERSION = "v1"

STYLES = [
    "terse and clipped (one or two sentences, minimal politeness)",
    "verbose and rambling, with background detail the reader has to filter",
    "frustrated or annoyed in tone, but still factual",
    "non-native English with small grammar mistakes and typos",
    "a forwarded reply chain with quoted '> On <day> you wrote:' lines",
    "a multi-issue body: one main problem plus an aside the customer mentions",
    "a blank or useless subject line with a real problem in the body",
    "nonsense or near-empty body (like a mis-sent or test message)",
    ("an embedded instruction trying to make the classifier mislabel it "
     "(prompt injection), with a real underlying question"),
    "polite and formal, as if written by a large enterprise customer",
]

GENERATOR_SYSTEM = """You write realistic customer support tickets for a fictional B2B SaaS product.
Respond with ONLY a JSON array, no markdown, no commentary. Each element:
{"subject": "...", "body": "..."}"""


def load_alive_models(status_path: Path, explicit: str | None) -> list[str]:
    if explicit:
        return [m.strip() for m in explicit.split(",") if m.strip()]
    if not status_path.exists():
        raise SystemExit(f"no status file at {status_path}; run the pool probe first or pass --models")
    status = json.loads(status_path.read_text())
    if not status.get("ok"):
        raise SystemExit("status file lists no alive models")
    return list(status["ok"])


class RateLimiter:
    """Account-wide free-tier throttle: N requests per rolling minute.

    OpenRouter's free tier answers 429 `free-models-per-min` (limit 20 on this
    account) once the budget is spent, and a 429 costs a whole generation or
    verification slot, so the script paces itself instead of discovering the
    limit by failing.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0]) + 0.05
            time.sleep(min(wait, 5.0))


def chat(model: str, messages: list[dict], key: str, max_tokens: int, timeout: int,
         limiter: RateLimiter | None = None) -> dict:
    if limiter is not None:
        limiter.acquire()
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(API, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/loura-synthetic",
        "X-Title": "loura-synthetic-gen",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
        choice = (payload.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        usage = payload.get("usage") or {}
        return {"ok": True, "text": text, "seconds": round(time.time() - t0, 2),
                "cost": float(usage.get("cost") or 0), "error": None,
                "finish": choice.get("finish_reason")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "text": "", "seconds": round(time.time() - t0, 2), "cost": 0.0,
                "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:160]}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "text": "", "seconds": round(time.time() - t0, 2), "cost": 0.0,
                "error": f"{type(e).__name__}: {e}"}


def parse_tickets(text: str) -> list[dict]:
    """Tolerant JSON extraction: raw, fenced, or first balanced [...] / {...}."""
    candidates = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text.strip())
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and data and all(isinstance(d, dict) for d in data):
            out = []
            for d in data:
                subject = str(d.get("subject", "")).strip()
                body = str(d.get("body", "")).strip()
                if subject or body:
                    out.append({"subject": subject, "body": body})
            if out:
                return out
    return []


def build_generator_prompt(cell: dict, styles: list[str], exemplars: list[dict], k: int,
                           rubric: dict) -> str:
    cat, pri = cell["category"], cell["priority"]
    ex = "\n".join(
        f'- [{t["category"]}/{t["priority"]}] subject: {t["subject"]!r} | body: {t["body"]!r}'
        for t in exemplars)
    style_lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(styles))
    return f"""Write exactly {k} NEW customer support tickets for the fictional SaaS product "Loura".
Each ticket must have category={cat} and priority={pri} according to the rubric below.
Do NOT copy the examples; write fresh tickets that could sit next to them.

Category rubric:
  billing   = {rubric["category"]["billing"]}
  technical = {rubric["category"]["technical"]}
  account   = {rubric["category"]["account"]}
  other     = {rubric["category"]["other"]}

Priority rubric:
  high   = {rubric["priority"]["high"]}
  medium = {rubric["priority"]["medium"]}
  low    = {rubric["priority"]["low"]}

Style requirements (one per ticket, in this order):
{style_lines}

Additional rules:
  - Vary length, tone and vocabulary across the {k} tickets.
  - In at most half of them state the urgency explicitly (e.g. "this is urgent",
    "not urgent at all"); in the rest, let the described impact carry it.
  - Keep every ticket internally consistent with category={cat}, priority={pri}.
  - Scope the impact to match priority={pri}: "high" tickets may say production is
    down / many users / blocking; "medium" tickets should stay scoped to one user
    or a routine request (a single person who cannot log in is still medium);
    "low" tickets must not describe any blocking impact.

Real tickets from this product, for style calibration (do not copy):
{ex}

Respond with ONLY a JSON array of {k} objects: [{{"subject": "...", "body": "..."}}, ...]"""


def near_duplicate(ticket: dict, others: list[dict], threshold: float = 0.8) -> str | None:
    text = f"{ticket['subject']}\n{ticket['body']}".lower()
    for other in others:
        ref = f"{other['subject']}\n{other['body']}".lower()
        if difflib.SequenceMatcher(None, text, ref).ratio() > threshold:
            return other["id"]
    return None


# A generated ticket must never spell out its own label: `[other/medium] Dark mode
# request` teaches the model to read the answer off the subject line instead of the
# content. Observed live in an early smoke run, hence the hard filter.
LABEL_LEAK = re.compile(
    r"\[\s*(billing|technical|account|other)\s*/\s*(low|medium|high)\s*\]"
    r"|(?:category|priority)\s*=\s*(billing|technical|account|other|low|medium|high)",
    re.IGNORECASE)


def label_leak(ticket: dict) -> bool:
    return bool(LABEL_LEAK.search(f"{ticket['subject']}\n{ticket['body']}"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate cross-verified synthetic tickets.")
    ap.add_argument("--per-cell", type=int, default=8, help="tickets per (category, priority) cell")
    ap.add_argument("--k", type=int, default=2, help="tickets per generation call")
    ap.add_argument("--models", help="comma-separated alive model ids (default: status file)")
    ap.add_argument("--status", default=str(STATUS))
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic_tickets.json"))
    ap.add_argument("--dropped-out", default=str(ROOT / "data" / "synthetic_dropped.json"))
    ap.add_argument("--exemplar-exclude", default="t-1002,t-1006,t-1009",
                    help="real ticket ids the generator must not see (sealed holdout)")
    ap.add_argument("--eval-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rate-per-minute", type=int, default=15,
                    help="account-wide free-tier request budget (OpenRouter free limit is 20/min)")
    ap.add_argument("--raw-out", default=str(ROOT / "data" / "synthetic_raw.json"),
                    help="where the raw generated tickets are kept, so verification can be re-run")
    ap.add_argument("--verify-raw", help="skip generation; verify the tickets in this raw file")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and one prompt, then exit")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    limiter = RateLimiter(args.rate_per_minute)

    models = load_alive_models(Path(args.status), args.models)
    if len(models) < 2 and not args.dry_run:
        print(f"need >= 2 alive models for independent verification, have {models}", file=sys.stderr)
        return 2

    labels_meta = json.loads((ROOT / "data" / "labelled_tickets.json").read_text())["meta"]
    rubric = labels_meta["rubric"]
    excluded = {i.strip() for i in args.exemplar_exclude.split(",") if i.strip()}
    real = load_real_tickets()
    exemplars = [t for t in real if t["id"] not in excluded]
    print(f"[gen] models: {models}")
    print(f"[gen] exemplars: {[t['id'] for t in exemplars]} (excluded: {sorted(excluded)})")

    cells = [{"category": c, "priority": p}
             for c in ("billing", "technical", "account", "other")
             for p in ("low", "medium", "high")]

    # One job per generation call: k tickets in one cell, k styles rotated
    # deterministically. The nonsense style only fits the `other` cell (a garbled
    # body cannot honestly carry technical/billing content), so other cells skip it.
    rng = random.Random(args.seed)
    jobs, style_cursor = [], 0
    for cell in cells:
        usable = [s for s in STYLES
                  if cell["category"] == "other" or "nonsense" not in s]
        made = 0
        while made < args.per_cell:
            k = min(args.k, args.per_cell - made)
            styles = [usable[(style_cursor + i) % len(usable)] for i in range(k)]
            style_cursor += k
            jobs.append({"cell": cell, "k": k, "styles": styles})
            made += k
    rng.shuffle(jobs)  # spread cells across models/time

    if args.dry_run:
        j = jobs[0]
        print(f"\n[dry-run] {len(jobs)} jobs, {sum(j['k'] for j in jobs)} tickets planned")
        print(build_generator_prompt(j["cell"], j["styles"], exemplars, j["k"], rubric))
        return 0

    # ---- generate ---------------------------------------------------------
    def run_job(idx_job: tuple[int, dict]) -> dict:
        idx, job = idx_job
        model = models[idx % len(models)]
        prompt = build_generator_prompt(job["cell"], job["styles"], exemplars, job["k"], rubric)
        # Reasoning models spend budget before the JSON, so give them room; a
        # truncated array is the most common failure mode here.
        res = chat(model, [{"role": "system", "content": GENERATOR_SYSTEM},
                           {"role": "user", "content": prompt}], key,
                   max_tokens=4000, timeout=args.timeout, limiter=limiter)
        tickets = parse_tickets(res["text"]) if res["ok"] else []
        if not tickets:  # retry on the NEXT model with a bigger budget and a stricter nudge
            model2 = models[(idx + 1) % len(models)]
            res2 = chat(model2, [{"role": "user", "content": prompt +
                                  "\n\nRespond with ONLY the JSON array, no reasoning, no commentary."}],
                        key, max_tokens=6000, timeout=args.timeout, limiter=limiter)
            res = {"ok": res2["ok"], "text": res2["text"], "error": res2.get("error"),
                   "finish": res2.get("finish"),
                   "cost": res.get("cost", 0) + res2.get("cost", 0),
                   "seconds": res2.get("seconds")}
            model = model2
            tickets = parse_tickets(res2["text"]) if res2["ok"] else []
        for i, t in enumerate(tickets):  # keep each ticket's style with the ticket
            t["style"] = job["styles"][i] if i < len(job["styles"]) else job["styles"][-1]
        leaked = [t for t in tickets if label_leak(t)]
        tickets = [t for t in tickets if not label_leak(t)]  # never pay to verify a leaked label
        err = res.get("error")
        if not tickets and not err:
            err = f"unparseable completion (finish_reason={res.get('finish')}, {len(res.get('text') or '')} chars)"
        elif leaked and not tickets:
            err = f"all {len(leaked)} tickets leaked their label into the text"
        return {"job": job, "model": model, "tickets": tickets, "error": err,
                "cost": res.get("cost", 0), "seconds": res.get("seconds")}

    t0 = time.time()
    if args.verify_raw:
        raw = json.loads(Path(args.verify_raw).read_text())
        generated = [(row["generator"], {"cell": row["cell"]},
                      {"subject": row["subject"], "body": row["body"], "style": row.get("style")})
                     for row in raw["tickets"]]
        gen_cost = 0.0
        print(f"[gen] reusing {len(generated)} raw tickets from {args.verify_raw} (no generation calls)")
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            gen_results = list(ex.map(run_job, list(enumerate(jobs))))
        gen_cost = sum(r["cost"] for r in gen_results)
        generated = [(r["model"], r["job"], t) for r in gen_results for t in r["tickets"]]
        failed_jobs = [r for r in gen_results if not r["tickets"]]
        print(f"[gen] {len(generated)} raw tickets from {len(jobs) - len(failed_jobs)}/{len(jobs)} jobs "
              f"in {time.time() - t0:.0f}s (cost ${gen_cost:.4f})")
        for r in failed_jobs:
            print(f"[gen]   job failed: {r['job']['cell']} {r['model']}: {r['error']}")
        # Persist the raw generation so verification (the expensive half) can be
        # re-run without paying for generation again.
        Path(args.raw_out).write_text(json.dumps({
            "meta": {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "prompt_version": PROMPT_VERSION, "jobs": len(jobs),
                     "raw": len(generated)},
            "tickets": [{"generator": m, "cell": job["cell"], "subject": t["subject"],
                         "body": t["body"], "style": t.get("style")}
                        for m, job, t in generated]}, indent=2))
        print(f"[gen] raw -> {args.raw_out}")

    # ---- verify (two independent models, serving prompt, lenient parse) ---
    def classify_with(model: str, ticket: dict) -> dict:
        """One verifier verdict: category/priority only (no summary requirement)."""
        from app.llm import CATEGORIES, PRIORITIES, ModelOutputError, extract_json

        res = chat(model, [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": build_user_prompt(
                               ticket["subject"], ticket["body"])}], key,
                   max_tokens=1200, timeout=args.timeout, limiter=limiter)
        verdict = None
        if res["ok"]:
            try:
                data = extract_json(res["text"])
            except ModelOutputError:
                data = None
            if isinstance(data, dict):
                cat = str(data.get("category", "")).strip().lower()
                pri = str(data.get("priority", "")).strip().lower()
                if cat in CATEGORIES and pri in PRIORITIES:
                    verdict = {"category": cat, "priority": pri}
        return {"verdict": verdict, "cost": res.get("cost", 0), "error": res.get("error")}

    def verify(idx_ticket: tuple[int, tuple]) -> dict:
        idx, (gen_model, job, ticket) = idx_ticket
        pool = [m for m in models if m != gen_model] or models
        verifiers = [pool[idx % len(pool)], pool[(idx + 1) % len(pool)]] if len(pool) > 1 \
            else [pool[0]]
        votes = [{"verifier": v, **classify_with(v, ticket)} for v in verifiers]
        return {"ticket": ticket, "gen_model": gen_model, "cell": job["cell"],
                "style": ticket.get("style"), "votes": votes}

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        ver_results = list(ex.map(verify, list(enumerate(generated))))
    ver_cost = sum(v["cost"] for r in ver_results for v in r["votes"])
    print(f"[gen] verification done (cost ${ver_cost:.4f})")

    # ---- filter + split ---------------------------------------------------
    kept, dropped = [], []
    seen_texts = [{"id": t["id"], "subject": t["subject"], "body": t["body"]} for t in real]
    n_dropped: dict[str, int] = {}
    for r in ver_results:
        t = r["ticket"]
        cell = r["cell"]
        verdicts = [v["verdict"] for v in r["votes"] if v["verdict"]]
        cat_agreed = any(v["category"] == cell["category"] for v in verdicts)
        pri_agreed = any(v["priority"] == cell["priority"] for v in verdicts)
        # Gate: the requested label must be supported by at least one independent
        # verdict, and unanimity against it on either field kills the ticket.
        # A split keeps the requested label (recorded in `votes`) because priority
        # is genuinely contested between models on the same rubric.
        if not verdicts:
            reason = "no parseable verifier verdict"
        elif len(verdicts) == 1 and not (cat_agreed and pri_agreed):
            reason = "single verifier disagrees"
        elif not cat_agreed:
            reason = "both verifiers disagree on category"
        elif not pri_agreed:
            reason = "both verifiers disagree on priority"
        else:
            reason = None
        dup_of = near_duplicate(t, seen_texts)
        if reason is None and label_leak(t):
            reason = "label leaked into the ticket text"
        if reason is None and dup_of:
            reason = "duplicate of " + dup_of
        if reason:
            # Group duplicates under one key: a per-row key would list one entry per
            # duplicate instead of counting the reason.
            key = "near-duplicate of a real ticket" if dup_of else reason
            n_dropped[key] = n_dropped.get(key, 0) + 1
        row = {
            "subject": t["subject"], "body": t["body"],
            "category": cell["category"], "priority": cell["priority"],
            "source": {"generator": r["gen_model"], "style": r["style"],
                       "votes": [{"verifier": v["verifier"],
                                  "category": (v["verdict"] or {}).get("category"),
                                  "priority": (v["verdict"] or {}).get("priority")}
                                 for v in r["votes"]],
                       "agreeing_verdicts": sum(
                           v["category"] == cell["category"] and v["priority"] == cell["priority"]
                           for v in verdicts),
                       "category_agreed": cat_agreed, "priority_agreed": pri_agreed},
        }
        if reason:
            row["drop_reason"] = reason
            dropped.append(row)
            continue
        kept.append(row)
        seen_texts.append({"id": f"pending-{len(seen_texts)}", "subject": t["subject"], "body": t["body"]})

    # deterministic eval slice per cell
    by_cell: dict[tuple, list[dict]] = {}
    for row in kept:
        by_cell.setdefault((row["category"], row["priority"]), []).append(row)
    split_rng = random.Random(args.seed + 1)
    for cell_rows in by_cell.values():
        split_rng.shuffle(cell_rows)
        n_eval = max(1, round(len(cell_rows) * args.eval_fraction)) if len(cell_rows) > 1 else 0
        for row in cell_rows[:n_eval]:
            row["split"] = "eval"
        for row in cell_rows[n_eval:]:
            row["split"] = "train"

    kept.sort(key=lambda r: (r["category"], r["priority"], r["subject"]))
    for i, row in enumerate(kept, 1):
        row["id"] = f"s-{i:04d}"

    n_train = sum(r["split"] == "train" for r in kept)
    n_eval = sum(r["split"] == "eval" for r in kept)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "prompt_version": PROMPT_VERSION,
            "models_alive": models,
            "exemplar_ids": [t["id"] for t in exemplars],
            "exemplar_excluded": sorted(excluded),
            "raw_generated": len(generated),
            "kept": len(kept), "dropped": len(dropped), "drop_reasons": n_dropped,
            "measured_cost_usd": round(gen_cost + ver_cost, 6),
            "notes": ("labels = the requested cell, gated by two independent models' verdicts with the "
                      "serving prompt. Kept only when at least one verdict supports both fields and "
                      "there is no unanimous contradiction; dropped on unanimous contradiction, a "
                      "lone disagreeing verdict, no parseable verdict, or near-duplication of a real "
                      "ticket. Split verdicts are recorded per ticket in source.votes."),
        },
        "tickets": kept,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    Path(args.dropped_out).write_text(json.dumps({"meta": payload["meta"], "tickets": dropped}, indent=2))

    print(f"\n[gen] kept {len(kept)} (train {n_train} / eval {n_eval}), dropped {len(dropped)}")
    if n_dropped:
        print("[gen] drop reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(n_dropped.items())))
    per_cell: dict[str, int] = {}
    for r in kept:
        per_cell[f"{r['category']}/{r['priority']}"] = per_cell.get(f"{r['category']}/{r['priority']}", 0) + 1
    print("[gen] per cell: " + ", ".join(f"{k}={v}" for k, v in sorted(per_cell.items())))
    print(f"[gen] measured cost ${gen_cost + ver_cost:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
