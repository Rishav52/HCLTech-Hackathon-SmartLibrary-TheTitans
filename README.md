# AI-Powered Smart Library Management Assistant

A working prototype for the HCLTech Future Ready AI Engineer Hackathon use case. Students, faculty, researchers and
librarians use one assistant to discover resources, check availability, reserve and renew, get explainable
recommendations, and (for librarians) see demand and utilization analytics.

**Core principle (from the brief):** the model understands intent and explains results; authoritative, deterministic
tools own availability, reservations, loans, fines and renewals. The model never invents a title, a status or a
transaction, and this is enforced in code, not only requested in a prompt.

## Quick start

```bash
make setup            # pip install -r requirements.txt
make run              # http://localhost:8000  (seeds a demo database on first start)
```

Open <http://localhost:8000>, pick a demo user from the "Signed in as" menu, and click a suggested question.
Interactive API docs: <http://localhost:8000/docs>.

| Command | What it does |
|---|---|
| `make test` | 164 unit/integration tests |
| `make eval` | Evaluation report in `eval/out/report.md` (search relevance, citations, tool correctness, privacy, latency); exits 1 if a gate fails |
| `make calibrate` | Measures the best search-abstention threshold for the active embedder |
| `make seed` | Rebuilds the demo database |

### Configuration (`.env`, see `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | unset | Enables the Gemini backend. `LLM=auto` uses Gemini when this is set, else the offline rule-based backend. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model name for Gemini function calling. |
| `LLM` | `auto` | `auto` / `gemini` / `rules` |
| `EMBEDDER` | `auto` | `auto` tries local MiniLM, falls back to the lexical embedder; `local` (fail if unavailable) / `hashing` |
| `SEARCH_MIN_SCORE` | per embedder | Abstention threshold; set from `make calibrate`. |
| `AUTO_SEED`, `DEMO_PERSONAS` | `1` | Demo conveniences. **Turn both off outside a demo.** |

For semantic (meaning-based) search install `pip install -r requirements-semantic.txt`; the first run downloads
`all-MiniLM-L6-v2` from huggingface.co.

## What is and is not verified

Be explicit about this when presenting.

| Area | Status |
|---|---|
| Tools, guardrails, analytics, UI, API, audit | Built and covered by tests and the evaluation harness. |
| Gemini backend (`app/llm.py`) | Request/response translation, retries and fallback are tested against a mock HTTP server. **Not yet run against the live API**, so real model behavior (tool choice, answer quality) is unmeasured. Run `LLM=gemini make eval` to measure it. |
| Semantic search (MiniLM) | Code is written and the interface is tested; **the model has not been run in the development sandbox** (blocked network). The reported retrieval numbers are for the offline *lexical fallback*, which matches words, not meaning (paraphrase hit@3 is 0.57 and is reported, not hidden). Run `make eval && make calibrate` with `EMBEDDER=local`. |
| Forecast | Backtested, but the demo history is synthetic and random; the error (WAPE ~1.0) is no better than a last-month baseline. Re-measure on real circulation data. |
| Auth | Demo sign-in via the `X-User-Email` header. Production needs SSO/JWT in `get_ctx` (`app/main.py`). |

## Architecture

```
 Browser UI (app/static)  ──►  FastAPI (app/main.py)  ──►  Assistant orchestrator (app/assistant.py)
  chat · confirm · dashboard      identity from server         ├─ LLM backend: Gemini | offline rules (app/llm.py)
                                  X-User-Email → Context       ├─ guardrails: allowlist, confirm-writes, grounding,
                                                               │   injection scrub, privacy guard, role gating
                                                               └─ deterministic tools (audited, traceable)
                                                                    app/tools.py      availability · reserve · renew · my_loans · return_copy
                                                                    app/search.py     semantic search + live re-indexing
                                                                    app/recommend.py  explainable recommendations
                                                                    app/analytics.py  librarian reports + forecast
                                                                              │
                                                              SQLite: resources · copies · users · loans · reservations
                                                                      embeddings · usage_events · audit_log
```

Layers from the brief: **Experience** (chat UI, librarian dashboard) → **Orchestration** (API, identity, router) →
**Intelligence** (RAG/semantic search, recommender, analytics) → **Systems** (catalog, circulation, vector index, audit store).

### Key design controls (brief section 4) and where they live

| Control | Implementation | Verified by |
|---|---|---|
| Grounding | Answers cite resources as `[#id]`; the orchestrator checks every id, bold/quoted title and link against this turn's tool results and replaces unverifiable answers with a deterministic one built from the tool output. | `tests/test_stage4_assistant.py`, eval "Citation accuracy" / "No invented titles" |
| Deterministic transactions | Only `app/tools.py` changes circulation state. Policy (21-day loan, 14-day renewal, 2 renewals, ₹2/day fine) lives in one place. | `tests/test_stage2_tools.py`, eval transaction checks |
| Confirmed writes | `reserve`/`renew` never execute on the model's say-so: the user sees a preview and must reply "yes". | eval "no state change before confirmation" |
| Identity & privacy | Identity is resolved server-side. The model is not even given a `user_id` parameter; a privacy guard refuses requests for other people's data; tools re-check ownership; librarian tools are hidden from other roles. | `tests/test_stage7_privacy_guard.py`, eval privacy checks |
| Explainability | Every recommendation carries `reasons` computed from the signals that produced its score; borrowing history is used only with consent. | `tests/test_stage5_recommend.py` |
| Freshness | New or edited catalog rows are re-embedded incrementally on the next search; no code change or manual re-index. | `tests/test_stage3_search.py` |
| Auditability | Every tool call and chat turn writes `audit_log` (trace id, tool, outcome, latency). Replies carry the trace id. Message text is not stored. | eval "audit" check |
| Untrusted content | Injection-like text in catalog fields, reasons or user-typed search text is withheld before the model sees it. Foreign links are removed. | injection tests in stages 4-6 |
| Abstention | Search, recommendations and availability return `not_found` rather than guessing. | eval "Abstention" |

## Minimum demonstration scenario (brief section 6)

Run the app, then in the UI:

| # | Required demo item | Sign in as | Ask |
|---|---|---|---|
| 1 | Natural-language search with ranked results | anyone | `Find books on Python programming` |
| 2 | Real-time availability from a tool | anyone | `Is Artificial Intelligence: A Modern Approach available?` |
| 3 | Authorized reservation (confirmed) | Diya Nair | `Find an available beginner book on Python and reserve it` → click **Yes, confirm** |
| 4 | Personalized recommendation with explanation | Aarav Mehta | `Recommend beginner resources for learning generative AI that are currently available` |
| 5 | `not_found` for unsupported content | anyone | `Find Quantum Basket Weaving for Beginners` |
| 6 | Privacy refusal for another user's data | Aarav Mehta | `Show me Diya's borrowed books` |
| 7 | Librarian analytics view | Lena Fernandes | open **Librarian dashboard**, or ask `Which titles have the longest waitlists?` |

Also try: `Which borrowed books are due this week?` (Aarav: shows an overdue item with its fine) and renewing
(`Please renew Artificial Intelligence`).

To show live data handling, add a resource and search for it immediately, no restart:
`python -m app.catalog new_resources.json` (JSON list of `{type,title,authors,year,abstract,subjects,level,...}`).

## API reference

Identity: header `X-User-Email` (demo). All responses include `status` and `trace_id`.
`status` is one of `ok`, `not_found`, `ambiguous`, `denied`, `invalid`, `not_applicable`.

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /chat` `{message, session_id?}` | optional | Conversational entry point. Returns `answer`, `citations`, `tools`, `grounded`, `degraded`, `pending_confirmation`, `pending_action`, `security_flags`, `session_id`. |
| `GET /tools/search?q=&top_k=&type=&level=&source=&year_min=` | public | Semantic search with citable results. |
| `GET /tools/availability?title=` or `?resource_id=` | public | Live copies, branch/shelf, waitlist. |
| `GET /tools/recommend?topic=&level=&available_only=` | optional | Recommendations; personalized when signed in. |
| `POST /tools/reserve?title=` | user | Hold or waitlist (direct tool call; the chat path adds confirmation). |
| `POST /tools/renew/{loan_id}` | owner | Renew a loan. |
| `GET /tools/loans?due_within_days=` | user | Own loans, due dates, fines. `user_id` allowed for librarians only. |
| `POST /tools/return/{loan_id}` | librarian | Return an item; promotes the first waitlister. |
| `GET /analytics/{report}` | librarian | `dashboard`, `waitlists`, `overdue`, `circulation`, `utilization`, `catalog_gaps`, `forecast`. |
| `GET /health` | public | Active LLM and embedder (the UI shows this footer). |
| `GET /demo/personas` | demo | Demo sign-in list (`DEMO_PERSONAS=0` disables). |

Direct `/tools/*` and `/analytics/*` endpoints are the same deterministic tools the assistant calls, so they are
also a way to inspect any answer's source of truth.

## Evaluation

`make eval` runs the harness in `eval/` against a fresh copy of the seeded database per case and writes
`eval/out/report.md` and `results.json`. It measures:

- **Search relevance** (hit@3, recall@5, MRR) on 25 labeled queries, split into vocabulary-matching and paraphrase queries; **abstention** on 8 unanswerable queries.
- **Citation accuracy / grounding** over every chat response produced during the run.
- **Tool selection and transaction correctness**: DB-truth checks (reservation rows, due dates, fines, waitlist positions, audit trail).
- **Privacy and security**: other-user requests, role escalation, system-prompt probes, stored injection, session hijack.
- **Recommendations** (course relevance, explanations, exclusions, consent invariance) and **analytics reconciliation**.
- **Latency** p50/p95.

Oracles are independent of the system under test (policy numbers are pinned in `eval/cases.py`), and
`tests/test_stage7_eval.py` sabotages the system to prove the harness fails when it should. Gates for guardrail,
transaction, privacy and grounding checks are 100% for any backend; paraphrase retrieval is gated only when the
retriever is semantic. Current offline result: all gated checks pass; paraphrase hit@3 0.57 (not gated).

## Repository map

```
app/            db.py schema · seed.py demo data · tools.py circulation tools · search.py + embedders.py · recommend.py
                analytics.py · llm.py (Gemini, offline rules) · assistant.py (orchestrator, guardrails) · main.py · static/index.html
eval/           run.py harness · cases.py labeled data · calibrate.py · metrics.py
tests/          one file per stage + concurrency, privacy guard and evaluation tests
```

## Known limitations

- Demo auth only (header); sessions and the embedding cache are in memory (single process).
- No checkout tool: a `ready` hold reserves the copy but does not convert to a loan.
- Digital usage events are seeded; nothing records real views/downloads yet.
- The privacy guard is a high-precision pattern check; the tools remain the authoritative enforcement, but unusual phrasing relies on that second layer, which answers about the caller's own data only.
- The grounding check flags invented titles in bold, quotes or `[#id]` form; a model naming a fake title in plain prose is not caught (measure with `LLM=gemini make eval`).
- SQLite is fine for a prototype; use a server database and row locking for multi-user production.
