# CLAUDE.md — standing instructions for this repo

Read this fully before doing anything. Then read `docs/PROJECT_PLAN.md` for the architecture.

## What this project is

SDE Intern take-home: an AI customer-support agent for **Uber_Support**, built from the Kaggle
"Customer Support on Twitter" dataset. It must (1) classify messages into intents, (2) draft a reply
grounded in Uber's historical replies and hand-curated help-centre articles, (3) decide auto-handle
vs escalate with a stated reason — and then **prove it works** with a golden set, an eval harness,
two baselines, and a report. The proof is graded above the system. I will be asked to explain and
modify every line of this code live, so keep it small and obvious.

## Architecture (fixed — do not redesign)

Pipeline, in this order, plain Python functions, no framework:

```
cache lookup → enrich (LLM, JSON) → rules (no LLM) → retrieve (MiniLM + Chroma) → draft (LLM) → safety check (LLM) → persist → respond
```

- **Enrich**: one Groq call returns `intent, confidence, sentiment, urgency, entities`. Pydantic-validated.
- **Rules**: ordered list, first match wins, reason = rule name. Runs BEFORE retrieval so escalations cost one LLM call.
- **Retrieve**: Chroma `pairs` (filtered by intent, top 5) + `help` (top 2). Unfiltered fallback if < 3 hits.
- **Draft**: fixed shape — acknowledge · one self-serve step if a help chunk matches · ask for one identifier · what happens next. In-app chat: **never write "DM us"**.
- **Safety check**: Groq 8B, `{safe, why}`. Fails closed.
- `pipeline/run.py` is the single entry point used by BOTH `api/main.py` and `eval/run_eval.py`.

Rationale for the shape is in `docs/PROJECT_PLAN.md` §4.3. If you think it should change, say so in chat; do not silently change it.

## Tech stack (fixed, all free tier)

- Python 3.11 · FastAPI · Pydantic v2 · Uvicorn
- LLM: `groq` SDK. Primary `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b` (Llama 3.x ids were decommissioned by Groq on 2026-08-16). Read model ids from env.
- Judge (eval only): Gemini Flash-Lite via `google-generativeai`. Never used in the pipeline.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, CPU.
- Vector store: ChromaDB, persistent dir `data/chroma/`.
- Storage: `sqlite3` from stdlib. Tables: `tickets`, `escalations`, `llm_cache`. No ORM.
- Eval: scikit-learn, pandas, matplotlib.
- Frontend (last, optional): React 18 + Vite + Tailwind.

**Do not add**: LangChain, LangGraph, LlamaIndex, any hosted DB or vector service, any paid API, SQLAlchemy, Redis, Celery.
If a task seems to need one of these, stop and ask.

## Hard rules

1. **Every LLM call goes through `pipeline/cache.py`.** Key = SHA-256 of model + JSON-dumped messages. When `CACHE_ONLY=1`, a cache miss raises — never hits the network. `make reproduce` must work with no API keys.
2. **Never write labels into `data/golden_set.csv`.** It is hand-labelled by a human; that is a graded deliverable. You may create the empty sampled file and the tooling; you may not fill `intent`, `escalate`, `reason`, `urgency`, or `sentiment` columns.
3. **Never scrape help.uber.com.** Help articles in `data/help_center/*.md` are hand-written paraphrases with `source_url` front matter.
4. **No leakage.** The golden set and the retrieval corpus must not share `customer_tweet_id`s. `scripts/validate_data.py` checks this; keep it passing.
5. **Deterministic where possible.** Enrich at `temperature=0`. Rules are pure functions. Seeds fixed in sampling and baselines.
6. **Free-tier aware.** Groq free tier is tightly rate-limited (see console.groq.com/settings/limits for the current per-model numbers). Batch scripts sleep between calls, back off exponentially on 429, and downgrade to the fallback model after 3 retries. Never run the golden set uncached in a tight loop.
7. **Cite what you borrow.** If you adapt a pattern from library docs, a blog, or a known snippet, add a one-line comment with the source and append it to `docs/BORROWED.md`.
8. **Small files, clear names.** One responsibility per module. No file over ~200 lines without a reason. Type hints everywhere. Docstring at the top of every module saying what it does in two sentences.
9. **Don't over-clean text.** Keep casing, emoji, punctuation, misspellings. Only strip `@mentions`, `t.co` links, HTML entities, `^XX` sign-offs, whitespace.
10. **UI last.** Do not touch `frontend/` until `eval/results/` exists and `make reproduce` runs under 15 minutes.
11. **Say manual steps to do next** Say manual steps to do if any before the next session with steps.

## Repo layout

```
data/        uber_pairs.jsonl  uber_threads.jsonl  uber_tweets.csv  stats.json  golden_set.csv  help_center/*.md  chroma/  app.db (gitignored)
scripts/     extract_uber_chunked.py  split_replies.py  validate_data.py  sample_golden.py  build_index.py  peek_retrieval.py  eval_intents.py
pipeline/    models.py  taxonomy.py  cache.py  db.py  enrich.py  rules.py  retrieve.py  draft.py  check.py  run.py
api/         main.py
baselines/   trivial.py  simple.py
eval/        run_eval.py  metrics.py  judge.py  kappa.py  results/
frontend/    (last)
docs/        PROJECT_PLAN.md  BORROWED.md  REPORT.md  DECISIONS.md
tests/       test_cache.py  test_rules.py  test_models.py
Dockerfile  docker-compose.yml  Makefile  requirements.txt  .env.example  README.md
```

## Data facts (from `data/stats.json`)

- 2,811,774 rows in the source CSV → 125,631 Uber-related tweets → 53,651 unique customer→Uber pairs → 40,886 threads. Apr 2016 – Sep 2017.
- `inbound=True` = customer message (pipeline input). `inbound=False` + `author_id=Uber_Support` = human agent reply (imitation target).
- Uber's Twitter replies are overwhelmingly triage ("send us a DM with your email"). Top-15 exact templates = 16%; near-duplicates much higher. Real procedures come from `data/help_center/`. Do not pretend the tweets contain resolutions they don't.

## Contracts (see `pipeline/models.py`)

```python
class Entities(BaseModel):
    email: str | None = None
    trip_date: str | None = None
    city: str | None = None
    amount: float | None = None
    mentions_safety: bool = False
    mentions_legal: bool = False
    is_repeat_contact: bool = False

class Enrichment(BaseModel):
    intent: Intent            # enum from taxonomy.py
    confidence: float         # 0–1
    sentiment: Literal["negative", "neutral", "positive"]
    urgency: Literal["low", "medium", "high"]
    entities: Entities

class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []  # last 2 turns max
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    conversation_id: str
    intent: Intent
    confidence: float
    sentiment: str
    urgency: str
    entities: Entities
    reply: str
    action: Literal["auto", "escalate"]
    reason: str               # rule name, "unsafe_draft", or "auto"
    retrieved_ids: list[str]
    latency_ms: int
```

## Rule table (`pipeline/rules.py`) — keep this exact order

```
safety → legal → amount_over (> AMOUNT_LIMIT) → urgent (urgency == high) → repeat_contact → low_confidence (< 0.7) → not_allowed (intent ∉ AUTO_ALLOW)
```
`AMOUNT_LIMIT` and `AUTO_ALLOW` live in `pipeline/taxonomy.py` and are set from golden-set results, not guessed.

## Working style

- Before writing code for a session, restate the task in 2–3 lines and list the files you will touch. Wait for a go if the list is longer than 4 files.
- After writing, run the relevant script or test and paste the real output. Do not claim something works without running it.
- When something is ambiguous (threshold, intent boundary, prompt wording), propose a default and flag it as a decision for `docs/DECISIONS.md` rather than deciding silently.
- Prefer editing an existing file over creating a new one.
- Keep commit messages short and specific; one session ≈ one commit.

## Commands

```
make data        # extract + split + validate
make index       # build data/chroma from pairs + help_center
make eval        # run golden set through baselines + system, write eval/results/
make reproduce   # CACHE_ONLY=1 make eval  (must finish < 15 min, no keys)
make api         # uvicorn api.main:app --reload
make test        # pytest tests/
```