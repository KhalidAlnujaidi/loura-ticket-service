# Loura Ticket Classification Service

Small HTTP service that ingests support tickets, classifies each one through a
model treated as an **unreliable dependency that returns text**, and serves the
results with filters and pagination. Built to the Loura engineering take-home
brief (4–6h budget, clarity over feature count).

Serving default: [Laya](https://huggingface.co/convaiinnovations/laya), a real
local typed-decision classifier (no API keys, Apache-2.0 weights). A scriptable
`FakeLLM` implements the same protocol and powers the deterministic test suite
and the broken/injection modes the brief asks for.

## Run it (clean clone, fewest steps)

```bash
python3 -m venv .venv && make install   # core + dev + laya (torch stack)
make seed                                # loads the 10 appendix tickets as pending
make run                                 # uvicorn; warms the checkpoint, re-enqueues pending
                                         # then open /ui (users) or /admin (ops)
make test                                # pytest
```

Python ≥ 3.10 (developed on 3.14). No API keys, no broker, no Docker. The
first `make install` pulls torch (~1GB) and the first boot downloads the
~808MB checkpoint from the HF Hub (one-time, inside lifespan). No-torch path:

```bash
make install-slim                        # core + dev only
LOURA_LLM_BACKEND=fake make run
```

## API shape

| Method & path | Success | Notes |
|---|---|---|
| `POST /tickets` | **201** first time, **200** if id exists | `{subject, body}` + optional `id`. Without one the server mints `t-dd-mm-yy-hh-mm-XX` (UTC creation minute + 2 random chars), retrying on same-minute collision so a clash can never return another caller's ticket. Explicit ids pass through untouched; a duplicate id never re-classifies. |
| `GET /tickets/{id}` | **200** / **404** | `status`, `attempts`, `failure_reason`, `classification` (null until classified). |
| `GET /tickets` | **200** | `?category=&priority=&status=&page=&page_size=` → `{items, page, page_size, total}`. 1-based page, default 20, max 100; bad enum → **422**. |
| `POST /tickets/{id}/reclassify` | **202** / **404** | Resets status/attempts/failure and requeues. |
| `GET /ui` | **200** | End-user page: submit a ticket, watch it classify. |
| `GET /admin` | **200** | Operator dashboard: metrics, filterable table, live log tail. |
| `GET /admin/metrics` | **200** / **401** | Worker/queue snapshot, status counts, `guard.injection_flags`. Gated by `LOURA_ADMIN_TOKEN` (`X-Admin-Token`) when set. |
| `GET /admin/logs` | **200** / **401** | `?after=<seq>` cursor over the last 500 log records. Same gate. |

Errors: `404` returns `{"error": {code, message}}`; `422` keeps FastAPI's
default `{"detail": ...}` — no bespoke validation stack for a take-home.
Lifecycle: `pending → classified | failed`.

## Interfaces: `/ui` and `/admin`

Hand-written single-file pages — vanilla HTML/JS, no build step, no CDN:

- **`/ui`** — submit subject + body → `POST /tickets`, polls until the row
  leaves `pending`, shows status/category/priority pills, summary, attempts,
  timestamp. Recent list in `localStorage`; failed cards show `failure_reason`
  with a **Try again** (reclassify) button.
- **`/admin`** — metrics strip (status counts, queue depth, in-flight,
  workers, backend, injection flags), filterable + paginated table with
  inline **Reclassify**, live level-colored log tail (pause/clear, 2 s poll).

Admin data endpoints are gated by `LOURA_ADMIN_TOKEN` when set
(`hmac.compare_digest`, standard `401` envelope); unset means open, matching
the brief's no-auth scope. Logs are a bounded in-memory ring (500 records)
attached to the root + uvicorn loggers for the app's lifespan.

## Decisions on the deliberately-open questions

1. **Storage — SQLite (`tickets.db`).** ACID, zero ops, survives restarts and
   is the source of truth for recovery: rows still `pending` at startup are
   re-enqueued. `CHECK` constraints on status/category/priority. Trade-off:
   synchronous `sqlite3` on the event loop at this scale; Postgres + async
   driver at real scale behind the same `Database` interface.
2. **Async execution — in-process `asyncio.Queue` + worker loops.** The queue
   is disposable by design: rebuilt from DB state on boot. The boundary that
   matters (recovery) lives in the DB, not in broker acknowledgements.
3. **Concurrency — 2 workers** (`LOURA_WORKERS`): overlapping classification,
   gentle on a provider rate limit.
4. **Restart / in-flight — at-least-once, nothing lost.** Shutdown cancels
   in-flight model calls; a cancelled call writes nothing and attempt counters
   increment only when an attempt *finishes*, so the row stays `pending` and
   re-runs on boot. A duplicate run costs inference, not correctness. Tested
   by killing the app mid-model-call (`tests/test_recovery.py`).
5. **Retry policy — 3 attempts** (`LOURA_MAX_ATTEMPTS`) covering malformed
   JSON, out-of-enum values, schema violations, provider exceptions and
   timeouts, with jittered exponential backoff. Exhausted → `failed` +
   `failure_reason`; the raw invalid output is logged and never stored.
6. **Prompt injection.** Ticket text never enters the system prompt; it is
   XML-escaped inside `<ticket>` tags (so `</ticket>` cannot break out); the
   system prompt tells generative models to ignore instructions found in the
   tags; output is hard-validated (`Literal` enums, `extra="forbid"`,
   summary ≤ 300). Laya's typed `choice` output cannot be instruction-injected
   the way free text can (measured: t-1005 → `billing` on the real invoice
   question), and even a compliant generative model gets gated — proven by a
   test with `FakeLLM(follow_injection=True)` storing injected-but-schema-valid
   values. An advisory input guard (`app/guard.py`) scans subject + body at
   ingest (bounded regexes: override / role-override / exfiltration /
   "classify it as" / authority claims); flagged tickets still classify (the
   schema gate is the hard boundary), a WARNING lands in the admin log and
   `guard.injection_flags` counts them — zero false positives across the 10
   appendix tickets, exactly t-1005 flags. Limits: schema validation
   guarantees shape, not truth; a trained detector (Prompt-Guard class) would
   fit the same `scan()` contract — thresholded carefully, since published
   false-positive rates on real traffic span ~1%–88% across open detectors and
   a wrongly flagged customer is the worse failure.
7. **Validation failure outcome — retry, then fail.** Same path as §5.
8. **Which model serves: Laya by default** (`LOURA_LLM_BACKEND=laya`), a real
   local classifier instead of a hosted LLM — the brief allows "none"; this
   keeps zero API keys while exercising the unreliable-boundary machinery
   (the adapter returns *text* through the same validation gate).
   `LOURA_LLM_BACKEND=openai` enables any OpenAI-compatible endpoint
   (`LOURA_LLM_API_KEY`/`BASE_URL`/`MODEL` — OpenAI, OpenRouter, vLLM,
   Ollama's `/v1`); `LOURA_LLM_TIMEOUT` stays the single deadline for every
   backend.

## The model boundary

Everything the worker believes about the model passes through one gate
(`app/llm.py`), regardless of backend:

```
raw text → extract JSON (raw / ``` fences / first balanced {…})
        → strict Pydantic validate (Literal enums, summary 1..300, extra=forbid)
        → store | reject(reason) → retry | failed
```

Failure reasons: `invalid model output: …`, `llm error: Type: …`,
`llm timeout after Xs`. All backends implement `async complete(system, user)
-> str`, so retries, timeouts, recovery and storage are backend-agnostic.

- **LayaLLM** (default): parses the escaped prompt back to `(subject, body)`,
  asks Laya two `choice` questions (category, priority) in one forward pass,
  marshals the answers through the JSON contract. Predictions serialize behind
  a lock in a worker thread (`asyncio.to_thread`); the checkpoint warms during
  lifespan before any ticket is processed.
- **FakeLLM** (tests + slim): keyword scorer over the unescaped ticket,
  scriptable per test — invalid JSON, bad enums, exceptions, delays, a
  backpressure gate, call counting, `follow_injection=True`.
- **HostedLLM** (optional): OpenAI-compatible chat completions via `httpx`,
  imported only on that path, fails fast without a key.

## Default classifier: Laya

[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) is a
non-autoregressive typed-decision model (ModernBERT-large 421M, Apache-2.0):
state + typed questions → calibrated `choice` answers in one forward pass, and
it never generates free text — nothing to parse, nothing to hallucinate. The
adapter still crosses the text boundary above, so the pipeline the brief
grades (validate → retry → fail) is real.

Zero-shot on the 10 appendix tickets, this machine (CPU): category agreement
8/10 (missed t-1006, t-1008), priority 4/10 (t-1003 returned `medium` despite
"production/blocking" — priority is the weaker primitive), median ≈ 206 ms,
t-1005 injection **immune**. These numbers are Laya-specific: `FakeLLM` scores
9/10 categories (its miss: t-1009); the backends disagree on edge cases, as
expected (zero-shot general-purpose vs keyword scorer). Caveats from Laya's
own model card: the shipped checkpoint's temperatures are partially invalid
(its calibration warning fires on load), `base` checkpoints are near-chance on
generic typed-decision benchmarks, English-only (`laya-multilingual` exists).

```bash
LOURA_LLM_BACKEND=fake make run                      # deterministic keyword backend (slim)
LOURA_LLM_BACKEND=laya LOURA_LLM_TIMEOUT=30 make run # verbose timeout, e.g. cold disk
LOURA_WORKERS=1 make run                             # one inference at a time
LOURA_LLM_BACKEND=openai LOURA_LLM_API_KEY=sk-… make run  # optional generative backend
LOURA_ADMIN_TOKEN=change-me make run                 # gate /admin/metrics + /admin/logs
```

Numeric overrides are range-checked when parsed (`Settings.from_env`):
`LOURA_WORKERS` / `LOURA_MAX_ATTEMPTS` ≥ 1, `LOURA_LLM_TIMEOUT` a finite
number > 0 (nan/inf parse as floats and would hang or fire instantly — they
die at the edge), else startup fails naming the variable. `LOURA_WORKERS=0`
used to produce zero loops: the service answered `201`, looked healthy and
classified nothing. `ClassificationWorker.start()` raises on a non-positive
count too.

## Fine-tuning on labelled tickets

Ten tickets do not make a training set, so the repo ships the loop that grows
one and tunes Laya on it (`make eval|synthetic|finetune`):

```
data/labelled_tickets.json     10 gold labels + the rubric they follow
scripts/generate_synthetic.py  free-pool generation, two-model cross-verification
  -> data/synthetic_{raw,tickets,dropped}.json   kept rows + rejected rows w/ reason
scripts/finetune_laya.py       RLCD fine-tune on MPS/CPU -> checkpoints/<tag>/
scripts/eval_agreement.py      agreement report through the serving path -> reports/
```

Synthetic rows survive only when independent verifiers agree on each field
(no unanimous contradiction, label text never embedded in the ticket, no
near-duplicates of real tickets). Training reuses Laya's own RLCD loop over
two `choice` questions built with the same `build_sequence` the service serves
with (asserted equal at startup), and fits the temperature the stock
checkpoint is missing. `LOURA_LLM_MODEL` selects a checkpoint, so serving a
tuned model is one env var. Baseline reports (`reports/zeroshot-*`) are
committed; `make eval` reproduces them.

Honest limits: synthetic labels are model-agreed, not human-adjudicated (fine
for training, unusable as an eval set — synthetic eval is reported separately);
10 real tickets is smoke-scale; one config was trained, no hyperparameter
search; the cross-verification gate is biased against `other` (feature
requests read as `technical`).

## Tests

```bash
make install && make test        # 70 passed + 1 skipped (laya e2e runs; hosted e2e needs LOURA_LLM_API_KEY)
make install-slim && make test   # 69 passed + 2 skipped (verified on this machine: no laya, no key)
# with LOURA_LLM_API_KEY set -> 71 passed
```

Deterministic suites pin `FakeLLM` in `conftest.py`; only
`test_laya_backend.py` exercises the real backend (skip-if-absent).

| File | Covers |
|---|---|
| `test_api_ingest.py` | 201→pending→classified; duplicate id → 200, one row, no re-run; 422s; 404 envelope; filters + pagination totals; reclassify; seed loads the 10 appendix tickets verbatim |
| `test_classification.py` | expected labels for t-1001/t-1003/t-1006; always-malformed → 3 attempts → `failed`, NULL classification columns; out-of-range enum → same; timeout counts as attempt; fenced JSON extracted; schema edges: summary 301 → rejected (300 passes), extra key → rejected, empty/whitespace summary → rejected |
| `test_injection.py` | t-1005 not obeyed; `follow_injection=True` still schema-valid (documented risk); `</ticket>` breakout cannot escape the wrapper (lossless round-trip); system prompt never contains ticket content (canary); guard flags exactly t-1005 across the corpus; flagged ticket still classifies, counter/log correct, duplicates never double-count |
| `test_recovery.py` | pending rows re-enqueued on startup; restart mid-model-call leaves `pending`, re-runs to `classified` |
| `test_config_and_lifecycle.py` | `LOURA_*` overrides reject zero/negative/non-numeric/**non-finite** values at parse time (nan would hang the timeout, inf would fire instantly); valid values + defaults unchanged; `worker.start()` idempotent, `stop()` cancels every loop; 40 tickets / 8 workers classify exactly once; `build_llm` wires `LOURA_LLM_MODEL` into the laya checkpoint |
| `test_ui_and_admin.py` | `/ui` + `/admin` serve; log buffer cursor semantics; metrics reflect workers/tickets; `LOURA_ADMIN_TOKEN` gates data (401) not the page; hosted backend: missing-key fail-fast, env wiring, payload shape, errors propagate, skip-if-no-key e2e |
| `test_ticket_ids.py` | minted `t-dd-mm-yy-hh-mm-XX` format; POST without/with-null id mints and classifies; explicit ids preserved byte-for-byte; forced same-minute collision retries to a fresh id |
| `test_labelled_data.py` | labelled set joins the appendix 1:1, labels inside enums; synthetic set well formed (splits populated, ids unique/disjoint, verifier votes recorded, no unanimous contradiction) |

## Layout

```
README.md  Makefile  requirements*.txt  pytest.ini  .gitignore
data/        sample_tickets.json (appendix verbatim), labelled_tickets.json,
             synthetic_{raw,tickets,dropped}.json (generated, with evidence)
app/
  main.py        create_app(), lifespan (recovery + shutdown), routes, id minting
  config.py      Settings, LOURA_* overrides (range-checked at parse time)
  schemas.py     API models + strict Classification (Literal enums, extra=forbid)
  db.py          SQLite: schema, CRUD, filtered list, lock + WAL
  prompts.py     system prompt, XML-escaped <ticket> builder, inverse parser
  llm.py         LLM protocol, JSON extraction + validation, FakeLLM
  laya_llm.py    Laya backend (warmup + lock, same text contract)
  hosted_llm.py  optional OpenAI-compatible backend
  guard.py       advisory prompt-injection heuristics at ingest
  observability.py  log ring buffer + /admin/{logs,metrics} (token-gated)
  worker.py      queue, N loops, retries+backoff, recovery, wait_idle
  seed.py        python -m app.seed
  static/        user.html (/ui) + admin.html (/admin)
scripts/       generate_synthetic.py, finetune_laya.py, eval_agreement.py
reports/       zero-shot agreement + eval reports (git-rev stamped)
tests/         conftest + 9 files
```

## Weaknesses (honest)

- **Semantic injection is not solved** — a test deliberately shows model
  compliant injection passing schema validation: shape ≠ truth.
- **Default summary is templated, not generated.** Laya is non-generative; the
  adapter writes `"<subject>: classified as …"` — safe (never quotes untrusted
  body text) but bland. The optional generative backend writes real prose.
- **Heavy default install** — torch + 808MB checkpoint; `make install-slim` +
  `LOURA_LLM_BACKEND=fake` exists for that reason.
- **Synchronous sqlite3 on the event loop** — fine here, wrong at volume;
  single-node only.
- **Zero-shot priority is weak** — t-1003 should be `high`, the stock
  checkpoint said `medium` (4/10). The fine-tune track targets this, with the
  limits listed above.
- **Queue dedupe is in-process memory** — two instances sharing one DB file
  could double-process (a broker/lease column would fix it).
- **No auth, no rate limiting** — explicitly out of scope for the brief.
- **`FakeLLM` labels are fixtures, not an accuracy claim** — its keyword
  scorer misses t-1009 (billing instead of technical).

## Future work / deliberately out of scope

- Anthropic-format / streaming adapters next to the OpenAI-compatible one;
  hosted keys stay env-only, none committed.
- Durable broker (SQS/Redis Streams) if workers scale beyond one box.
- Graceful in-flight drain on shutdown (chose the reclassify bonus instead —
  pick one, per brief).
- Human-adjudicated labels; a trained injection detector behind
  `guard.scan()`; human review for high-risk tickets.
