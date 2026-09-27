# Loura Ticket Classification Service

Ingests support tickets, classifies each asynchronously through a model
treated as an unreliable dependency that returns text, and serves the results
back with filters and pagination. FastAPI + SQLite + a local typed-decision
model ([Laya](https://huggingface.co/convaiinnovations/laya)); a scriptable
`FakeLLM` stands in for tests. Python ≥ 3.10.

## Run from a clean clone

```bash
python3 -m venv .venv && make install   # core + dev + laya (torch stack)
make seed                                # loads the 10 appendix tickets (data/sample_tickets.json)
make run                                 # uvicorn: /ui for users, /admin for ops
make test                                # 67 passed, 2 skipped (skips need laya / LOURA_LLM_API_KEY)
```

No API keys, no broker, no Docker. The first `make install` pulls torch
(~1GB) and the first boot downloads the ~808MB checkpoint (one-time). Without
torch: `make install-slim`, then `LOURA_LLM_BACKEND=fake make run`.

## The open questions

### How and where tickets are stored

One SQLite file — `tickets.db` (`LOURA_DB_PATH` to move it), stdlib `sqlite3`,
WAL, writes serialized behind a lock. Single table `tickets`: `id TEXT
PRIMARY KEY, subject, body, status, attempts, failure_reason, category,
priority, summary, created_at, classified_at`, with `CHECK` constraints on
`status`/`category`/`priority` so an invalid value cannot land through any
code path, and indexes on the three filter columns. Why: ACID, zero
infrastructure, survives restarts, and the DB is the source of truth recovery
is built on. Known trade-off: the synchronous driver on the event loop is
wrong at volume — at real scale, Postgres + async driver behind the same
`Database` interface.

### How asynchronous work is executed

`POST /tickets` inserts the row and enqueues its id — it never calls the
model. N worker loops (`ClassificationWorker`: asyncio tasks started in app
lifespan, stopped on shutdown) pull ids off an in-process `asyncio.Queue`,
call `complete(system, user) -> str` under `asyncio.wait_for(…,
LOURA_LLM_TIMEOUT)`, then validate and store on the same loop. Laya's forward
pass runs in a worker thread (`asyncio.to_thread`, serialized behind a lock)
so the event loop stays responsive. Why an in-process queue: no Redis/RabbitMQ
for a take-home — the queue is deliberately disposable and rebuilt from DB
state on boot, so the recovery boundary lives in the DB, not in broker
acknowledgements.

### How many classifications may run at once

`LOURA_WORKERS` tickets at once (default 2, minimum 1 — enforced when config
parses and again in `start()`), one per worker loop. With the Laya backend the
model forward passes additionally run one at a time (one shared in-process
checkpoint behind a lock); the OpenAI-compatible backend makes up to
`LOURA_WORKERS` parallel provider calls. Why 2: demonstrates overlapping
classification and retries while staying gentle on a provider rate limit. A
test runs 40 tickets across 8 workers and asserts every ticket is classified
exactly once.

### What happens to in-flight work if the service is restarted

At-least-once, nothing lost. Shutdown cancels in-flight model calls; a
cancelled call writes nothing, and `attempts` increments only when an attempt
*finishes*, so the row stays `pending`. On startup every `pending` row is
re-enqueued and runs to `classified` or `failed`. The worst case is a
duplicate model call for one ticket — classification rewrites the same
columns, so the cost is inference, not correctness. Tested by killing the app
mid-model-call and asserting the row recovers. Trade-off: this re-enqueue
design instead of a graceful in-flight drain (see the pick in the next
answer).

### Retry policy for model failures, and how a ticket ends up failed

Everything that can go wrong is one retry path: malformed JSON, values
outside the enums, schema violations, provider exceptions and timeouts all
count as a failed attempt, and so do internal worker errors — a store blip or
a bug is booked as `internal error: <Type>` against the same budget, so a
ticket can never be stranded `pending` with its queue slot consumed. 3 attempts
(`LOURA_MAX_ATTEMPTS`), jittered exponential backoff (base 50 ms, cap 1 s).
After the third, the ticket becomes `failed` with `failure_reason` set
(`invalid model output: …`, `llm error: <Type>: …`, `llm timeout after Xs`,
`internal error: <Type>`) and `classification` stays NULL — raw invalid
output goes to the admin log tail and is never persisted. The one edge: a
store too broken to even book the failure leaves the row `pending`, where the
startup recovery scan re-enqueues it. Values outside the allowed sets can
never reach the store as-is (that gate is `Classification`: `Literal` enums,
summary 1–300 chars, `extra="forbid"`).

**Finish-early pick — re-classification:** `POST /tickets/{id}/reclassify`
(202) resets `status`/`attempts`/`failure_reason` and requeues, so failed
tickets — or tickets classified before a prompt change — can be re-run
deliberately. A ticket that is still `pending` is left untouched (see the
exact API shape below).

### What, if anything, you do about prompt injection

Ticket text is data, never instruction. Four layers:

1. **Quarantined prompt.** Ticket text never enters the system prompt; it is
   XML-escaped inside `<ticket>` tags so `</ticket>` cannot break out (the
   parse round-trip is lossless; a breakout test pins it). The system prompt
   declares the tags untrusted and tells generative models to ignore any
   instructions inside them.
2. **Output gate — the hard boundary.** Every model answer from any backend
   is extracted as JSON and validated (enums, summary bounds, no extra keys)
   before storage. A willfully compliant model can store *wrong-but-valid*
   values — a test with `FakeLLM(follow_injection=True)` demonstrates exactly
   that — but never malformed or out-of-enum ones. Validation guarantees
   shape, not truth; that limit is honest and tested.
3. **Typed model.** The default Laya backend answers typed `choice` questions,
   which cannot be instruction-injected the way free text can. Measured:
   t-1005 (the appendix's injection ticket) returns `billing` on the real
   invoice question, ignoring "classify as technical".
4. **Advisory input guard.** `app/guard.py` scans subject + body at ingest
   with bounded regexes (instruction override, role/mode release,
   exfiltration, "classify it as" directives, authority claims). Flags never
   block — wrongly flagging a real customer is the worse failure — the ticket
   still classifies, a WARNING lands in the admin log and
   `guard.injection_flags` counts it. Pinned: exactly t-1005 flags, zero
   false positives across the 10 appendix tickets.

### Exact API shape

Lifecycle: `pending → classified | failed`. `attempts` counts finished
attempts. JSON errors use the envelope `{"error": {"code", "message"}}`
(`not_found`, `unauthorized`, `id_generation_failed`); **422** validation
errors keep FastAPI's default `{"detail": ...}`.

Single-ticket object (returned by create/get/reclassify, and as list items):

```json
{ "id": "t-1005", "subject": "…", "body": "…", "status": "classified",
  "attempts": 1, "failure_reason": null,
  "classification": { "category": "billing", "priority": "high", "summary": "…" },
  "created_at": "2026-09-27T…", "classified_at": "2026-09-27T…" }
```

`classification` is `null` until `status` is `classified`.

| Method & path | Payload | Success | Errors |
|---|---|---|---|
| `POST /tickets` | `{id?, subject, body}` — `id` 1–100 chars, optional (omitted or null → server mints `t-dd-mm-yy-hh-mm-XX` from the UTC creation minute + 2 random chars, retried on same-minute collision); `subject` ≤ 300; `body` ≤ 20 000; unknown fields rejected | **201** first time; **200** when the id already exists (idempotent: same object back, never re-classified) | **422**; **503** `{error: id_generation_failed}` after 10 colliding mints |
| `GET /tickets/{id}` | — | **200** ticket object | **404** `{error: not_found}` |
| `GET /tickets` | query `?category=&priority=&status=&page=&page_size=` (all optional; enum-gated; `page` ≥ 1; `page_size` 1–100, default 20) | **200** `{items: [ticket…], page, page_size, total}` | **422** on bad enum or page bounds |
| `POST /tickets/{id}/reclassify` | — | **202** reset ticket object (requeued). While the ticket is still `pending` this is a **no-op**: 202 with the row unchanged — it is already on its way, and a reset would zero the attempt bookkeeping under the in-flight run | **404** |
| `GET /ui` | — | **200** `text/html` (end-user page) | — |
| `GET /admin` | — | **200** `text/html` (operator dashboard) | — |
| `GET /admin/metrics` | header `X-Admin-Token` when `LOURA_ADMIN_TOKEN` is set | **200** `{workers: {configured, alive, queue_depth, in_flight, scheduled, idle}, tickets: {pending, classified, failed, total}, llm: {backend, max_attempts}, guard: {detector, injection_flags}, server_time}` | **401** `{error: unauthorized}` |
| `GET /admin/logs` | query `?after=<seq>` (poll cursor) | **200** `{items: [{seq, ts, level, logger, text}], next, count}` — last 500 records, ≤ 200 per page | **401** |

`/ui` and `/admin` are excluded from OpenAPI. Admin data endpoints compare
the token with `hmac.compare_digest`; with no token configured they are open
(the brief scopes auth out).

## What I would change or add with more time

- Postgres + a durable broker (or a lease column) for multi-instance workers,
  and graceful in-flight drain on shutdown — deliberately not built; the
  finish-early pick was re-classification.
- A trained injection detector (Prompt-Guard class) behind `guard.scan()`,
  thresholded against a false-positive budget, plus human review on flags.
- Real summaries: the Laya path templates `"<subject>: classified as …"`
  (safe — it never quotes untrusted body text — but bland); the optional
  `LOURA_LLM_BACKEND=openai` path writes real prose through the same gate.
- A small evaluation (hand-labelled tickets + an agreement script) and a
  domain fine-tune. An exploratory track was built during development and
  deliberately cut to keep the submission small; it remains in the repository
  history. Priority is the field to fix first (t-1003 should be `high`,
  returns `medium` zero-shot).

## Weaknesses of this solution

- **Semantic injection is not solved** — schema validation guarantees shape,
  not truth (Q6 has a test proving injected-but-valid values get stored).
- **Default summary is templated, not generated** — bland but safe.
- **Heavy default install** — torch + ~808MB checkpoint on first boot (the
  slim fake-backend mode exists for that reason).
- **Synchronous SQLite on the event loop**, single-node; queue dedupe is
  in-process, so two processes sharing one DB file could double-process.
- **Modest zero-shot accuracy on the 10 appendix tickets** — 8/10 category
  and 4/10 priority against my hand labels (t-1003 should be `high`, returns
  `medium`).
- **Tests pin the deterministic `FakeLLM`**; real-model coverage is one
  skip-if-absent e2e suite, so the heavy backend is less exercised.
- **No auth or rate limiting** (explicitly out of scope); admin endpoints are
  open unless `LOURA_ADMIN_TOKEN` is set.
