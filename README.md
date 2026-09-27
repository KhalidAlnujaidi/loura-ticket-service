# Loura Ticket Classification Service

Small HTTP service that ingests support tickets, classifies each one through a
model treated as an **unreliable dependency that returns text**, and serves the
results back with filters and pagination. Built to the Loura engineering
take-home brief (4–6h budget, clarity over feature count).

Serving default: [Laya](https://huggingface.co/convaiinnovations/laya), a real
local typed-decision classifier (no API keys, Apache-2.0 weights) — see
[Default classifier: Laya](#default-classifier-laya). A scriptable `FakeLLM`
implements the same protocol and powers the deterministic test suite (and the
broken/injection modes the brief asks for).

## Run it (clean clone, fewest steps)

```bash
python3 -m venv .venv && make install   # core + dev + laya (torch stack)
make seed                                # loads the 10 appendix tickets as pending
make run                                 # uvicorn (startup warms the checkpoint, re-enqueues pending)
make test                                # pytest
```

Python ≥ 3.10 (developed on 3.14). No API keys, no broker, no Docker.
First `make install` pulls torch (~1GB); **first server boot downloads the
~808MB checkpoint from the HF Hub** (one-time, happens in lifespan so it never
eats into the per-call timeout). Subsequent boots load from cache in seconds.

Lightweight path (no torch — uses the deterministic fake backend instead):

```bash
make install-slim                        # core + dev only
LOURA_LLM_BACKEND=fake make run
```

## API shape

| Method & path | Success | Notes |
|---|---|---|
| `POST /tickets` | **201** first time, **200** if id exists | `{id, subject, body}`; duplicate id never re-classifies. Body is the full ticket object. |
| `GET /tickets/{id}` | **200** / **404** | Includes `status`, `attempts`, `failure_reason`, `classification` (null until classified). |
| `GET /tickets` | **200** | `?category=&priority=&status=&page=&page_size=` → `{items, page, page_size, total}`. 1-based page, default 20, max 100. Enum params validated → **422**. |
| `POST /tickets/{id}/reclassify` | **202** / **404** | Bonus pick: resets status/attempts/failure, requeues (re-run failed tickets or tickets classified before a prompt change). |

Errors: `404` uses the envelope `{"error": {"code", "message"}}`; `422` keeps
FastAPI's default `{"detail": ...}` shape (deliberate: no bespoke validation
stack for a take-home).

Lifecycle: `pending → classified | failed`.

## Decisions on the deliberately-open questions

1. **Storage — single SQLite file (`tickets.db`).** ACID, zero ops, survived
   restarts, and it is the *source of truth* for recovery: on startup every
   row still `pending` is re-enqueued (see §4). Schema has `CHECK` constraints
   on `status`/`category`/`priority` so invalid values cannot land even via a
   future code path. Trade-off: the driver is synchronous `sqlite3` called on
   the event loop (short transactions; a known trade-off for this scale).
   At real scale → Postgres + async driver, same `Database` interface.
2. **Async execution — in-process `asyncio.Queue` + worker loops.** The queue
   is deliberately *disposable* (non-durable): rebuild from DB state on boot.
   No Redis/RabbitMQ to install for a take-home; the boundary that matters
   (recovery) lives in the DB, not in broker acknowledgements.
3. **Concurrency — 2 workers by default** (`LOURA_WORKERS`): enough to show
   overlapping classification, gentle on a provider rate limit.
4. **Restart / in-flight work — at-least-once, nothing lost.** Shutdown
   cancels in-flight model calls; a cancelled call never writes anything, and
   attempt counters only increment when an attempt *finishes*, so the row
   stays `pending` and is re-run on next boot. Classification is idempotent
   (same ticket → potentially same label); a duplicate run costs model
   inference, not correctness. Tested by killing the app mid-model-call
   (`tests/test_recovery.py`).
5. **Retry policy — 3 attempts** (`LOURA_MAX_ATTEMPTS`) covering malformed
   JSON, out-of-enum values, schema violations, provider exceptions, and
   timeouts, with small jittered exponential backoff (`LOURA_LLM_*`). After
   exhaustion → `failed` + `failure_reason` stored, **the raw invalid output is
   logged and never persisted as classification**. Table columns stay NULL.
6. **Prompt injection.** Ticket text never enters the system prompt; it is
   XML-escaped inside `<ticket>` tags (so `</ticket>` cannot break out); the
   system prompt is written for generative models (ignore instructions found
   in the tags); output is hard-validated against `Literal` enums +
   `extra="forbid"` + summary ≤ 300. Two independent layers: Laya's typed
   `choice` answers cannot be instruction-injected the way free text can
   (measured: t-1005 → `billing` on the real invoice question, ignoring
   "classify as technical"), *and* even a willfully compliant generative model
   gets gated — proven by a test running `FakeLLM(follow_injection=True)` that
   stores injected-but-schema-valid values. Honest limit: schema validation
   guarantees shape, not truth. Mitigations with more time: content
   heuristics, human review on high-risk fields, adversarial eval set.
   Ticket t-1005 in the sample data is the injection attempt.
7. **Validation failure outcome — retry, then fail.** Same path as §5.
8. **Which model serves: Laya by default** (`LOURA_LLM_BACKEND=laya`), a real
   local classifier rather than a hosted LLM — the brief allows "none"
   (no provider); this keeps zero API keys while still exercising the
   unreliable-boundary machinery (adapter returns *text*, same validation gate).
   `FakeLLM` stays the scripted stand-in for tests and for `make install-slim`.

## The model boundary

Everything the worker believes about the model passes through one gate
(`app/llm.py`), regardless of backend:

```
raw text → extract JSON (raw / ``` fences / first balanced {…})
        → strict Pydantic validate (Literal enums, summary 1..300, extra=forbid)
        → store | reject(reason) → retry | failed
```

The worker stores only what came out of that gate. Failure reasons are
`invalid model output: …`, `llm error: Type: …`, `llm timeout after Xs`.
Both backends implement `async complete(system, user) -> str`, so retries,
timeouts, recovery and storage are backend-agnostic.

**LayaLLM** (default): converts the escaped `<ticket>` prompt back to
`(subject, body)`, asks Laya two `choice` questions (category, priority) in one
forward pass, and marshals the answers through the JSON contract. Predictions
are serialized behind a lock and executed in a worker thread
(`asyncio.to_thread`) so the event loop stays responsive; the checkpoint is
warmed in app lifespan before any ticket is processed.

**FakeLLM** (tests + slim mode): keyword scorer over the unescaped ticket —
categories by keyword hits (billing/technical/account, zero hits → `other`),
priority via explicit markers (`not urgent`/`nice to have` → low,
`urgent`/`production`/`blocking` → high), templated one-sentence summaries.
It is scriptable per test: invalid JSON, bad enums, raised exceptions, delayed
responses (timeout), an `asyncio.Event` backpressure gate, call counting, and
`follow_injection=True` to demonstrate the residual risk of a compliant
generative model.

## Default classifier: Laya

[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) is a
non-autoregressive typed-decision model (ModernBERT-large 421M, Apache-2.0):
give it state + typed questions, it returns calibrated `choice` answers in one
forward pass and **never generates free text** — nothing to parse, nothing to
hallucinate, free locally, no API key. The adapter still goes through the text
boundary above so the pipeline the brief grades (validate → retry → fail) is
real, not bypassed.

Measured on the 10 appendix tickets, zero-shot, this machine (CPU):

| | result |
|---|---|
| Category agreement | 8/10 (missed t-1006 dark-mode feature request → `technical`, t-1008 `asdf` → `technical`) |

> These numbers are **Laya-specific**. `FakeLLM` scores differently on the same
> 10 tickets (9/10, its one miss is t-1009 → `billing` instead of `technical`,
> because "I think I was overcharged last quarter" out-scores the export/upload
> keywords). The two backends disagree on t-1006, t-1008 and t-1009 — expected,
> since one is zero-shot general-purpose and the other is a keyword scorer.
| Priority (where asserted) | t-1003 came back `medium` despite "production/blocking" — priority is the weaker primitive |
| Latency | median ≈ 206 ms CPU, first call ≈ 2 s |
| t-1005 injection | **immune** — typed-choice output can't be instruction-injected the way free text can; came back `billing` on the real invoice question |

Caveats worth knowing (also in Laya's own model card): the shipped checkpoint
ships partially-invalid temperatures (its own calibration warning fires on
load), `base` checkpoints are near-chance on generic typed-decision benchmarks
(a fine-tuned domain checkpoint lifts this a lot — their notebook fine-tunes on
2× free T4s), and English-only — use `laya-multilingual` for other languages.
Serving knobs:

```bash
LOURA_LLM_BACKEND=fake make run           # deterministic keyword backend (slim)
LOURA_LLM_BACKEND=laya LOURA_LLM_TIMEOUT=30 make run   # verbose timeout, e.g. cold disk
LOURA_WORKERS=1 make run                  # single inference at a time (default lock already serializes)
```

Numeric overrides are **range-checked when parsed** (`Settings.from_env`):
`LOURA_WORKERS` and `LOURA_MAX_ATTEMPTS` must be ≥ 1 and `LOURA_LLM_TIMEOUT`
must be > 0, otherwise startup fails immediately with a message naming the
offending variable. This matters more than it looks: `LOURA_WORKERS=0` used to
produce zero worker loops, so the service answered `201` on every POST, looked
healthy, and silently never classified anything. `ClassificationWorker.start()`
raises on a non-positive count too, covering the path where a `Settings` object
is constructed in code rather than from the environment.

End-to-end coverage against the real model: `tests/test_laya_backend.py`
(skip-if-absent so the suite still runs after `make install-slim`).

## Tests

```bash
make install && make test        # 37 passed (includes the real-model e2e)
make install-slim && make test   # 36 passed + 1 skipped (no laya installed)
```

The deterministic suites pin `FakeLLM` explicitly in `conftest.py`
regardless of the serving default — scriptable failures and stable labels
are the point — while `test_laya_backend.py` exercises the real default
backend end-to-end (warmup → classify → validated store).

| File | Covers |
|---|---|
| `test_api_ingest.py` | 201→pending→classified; duplicate id → 200, one row, no re-run (call count == 1); 422s; 404 envelope; list filters + combined + pagination totals; reclassify; seed loads all 10 appendix tickets verbatim |
| `test_classification.py` | expected labels for t-1001/t-1003/t-1006; always-malformed → 3 attempts → `failed`, NULL classification fields; out-of-range enum → same; timeout counted as attempt; fenced JSON extracted |
| `test_injection.py` | t-1005 default mode not obeyed; `follow_injection=True` still schema-valid (documented risk) |
| `test_recovery.py` | pending rows re-enqueued on startup; restart mid-model-call leaves `pending` and re-runs to `classified` |
| `test_config_and_lifecycle.py` | `LOURA_*` env overrides reject zero/negative/non-numeric values (a bad value must fail at parse time, not degrade silently); valid values and defaults unchanged; `worker.start()` is idempotent and `stop()` cancels every loop; 40 tickets across 8 workers classify exactly once |

## Layout

```
README.md  Makefile  requirements*.txt  pytest.ini  .gitignore
data/sample_tickets.json      # appendix tickets verbatim
app/
  main.py        create_app(), lifespan (recovery + shutdown), routes
  config.py      Settings, LOURA_* env overrides (range-checked at parse time)
  schemas.py     API models + strict Classification (Literal enums, extra=forbid)
  db.py          SQLite: schema, CRUD, filtered list, lock + WAL
  prompts.py     system prompt, XML-escaped <ticket> builder, inverse parser
  llm.py         LLM protocol, JSON extraction + validation, FakeLLM
  laya_llm.py    default Laya backend (warmup + lock, same text contract)
  worker.py      queue, N loops, retries+backoff, recovery, wait_idle
  seed.py        python -m app.seed
tests/           conftest + 5 suites (+ real-model laya e2e)
```

## Weaknesses (honest)

- **Semantic injection is not solved** — by design there is a test *showing*
  model-compliant injection still passes schema validation. Schema validation
  guarantees shape, not truth.
- **Summary of the serving default is templated, not generated.** Laya is
  non-generative; the adapter writes `"<subject>: classified as …"`. Honest and
  safe (never quotes untrusted body text) but bland — a generative model or a
  summary head would write better prose.
- **Heavy default install** — torch + 808MB checkpoint; `make install-slim`
  + `LOURA_LLM_BACKEND=fake` exists for that reason, but the default run
  experience now has a real download step.
- **Synchronous sqlite3 on the event loop** — fine for a take-home, wrong at
  volume; also single-node only.
- **Zero-shot real-model priority is weak** (see Laya table): t-1003 should be
  `high` (production + blocking) but the stock checkpoint said `medium`; base
  checkpoints really want a fine-tune on our own labelled tickets.
- **Queue dedupe is in-process memory** — two app instances sharing one DB
  file could double-process (irrelevant for one process; a broker/lease column
  would fix it).
- **No auth, no rate limiting** — explicitly out of scope for the brief.
- **Keyword classifier and the real model disagree on edge cases.** `FakeLLM`
  is a scorer tuned to make the suite deterministic, not an accuracy claim; it
  misses t-1009 (billing instead of technical). Treat its labels as test
  fixtures, not a baseline.

## Future work / deliberately out of scope

- Real provider client behind the same `LLM` protocol (OpenAI/Anthropic
  format adapters), with keys from env only (none committed).
- Durable broker (SQS/Redis Streams) if workers scale beyond one box.
- Graceful in-flight drain on shutdown (chose the reclassify bonus instead —
  pick one, per brief).
- Eval script: labelled tickets → agreement report; fine-tune the domain
  classifier (Laya ships a Kaggle-ready notebook for this).
- Content-security heuristics / human review for high-risk tickets.