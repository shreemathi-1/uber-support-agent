# Uber Support AI Agent

An AI customer-support agent for **Uber_Support**, built from the Kaggle "Customer Support on
Twitter" dataset. Classifies each message, drafts a reply grounded in Uber's historical replies
and help-centre articles, and decides auto-handle vs escalate with a stated reason.

## ⚠️ Current status (read before the numbers)

- The golden evaluation set (`data/golden_set.csv`) has **150 rows**, of which **138 are fully
  labelled** across all five ground-truth columns (147 rows have an `intent`, 146 have an
  `escalate` value) as of **2026-09-13**. A handful of the filled-in values are not yet clean
  (a few blank or invalid entries) — see `docs/DECISIONS.md` #48. Any metric run on this set today
  is close to real but still a **partial** result, not the final one; `eval/run_eval.py` prints
  **"NO LABELS — plumbing check only"** if it is ever run against a completely unlabelled file.
- Help-centre articles are **hand-written placeholders**, not yet verified against
  help.uber.com. All 12 files in `data/help_center/` are flagged `date_checked:
  UNVERIFIED-PLACEHOLDER`.
- Two suspected gaps in the current rule table, not yet confirmed by a full failure analysis:
  `driver_onboarding_or_earnings` and `app_or_account_issue` are both auto-allowed today, but
  spot checks in the golden set (a misdirected driver payout, a hacked account) suggest some
  rows in those intents should escalate. Tracked as an open item in `docs/PROJECT_PLAN.md` §14
  and to be confirmed in `docs/REPORT.md`'s failure-analysis section (not yet written).
- `docs/REPORT.md` itself does not exist yet; `docs/HANDBOOK.md` is the current onboarding
  document and `docs/DECISIONS.md` the decision log.
- `data/app.db` (the SQLite file holding `llm_cache`) is gitignored. This repo's copy already has
  every call the pipeline needs cached, so `make reproduce` below runs offline as-is — but a
  literally fresh `git clone` starts with an empty cache and would need one built up (or copied in)
  before `CACHE_ONLY=1` can replay anything.

## Quickstart — reproduce the headline results (no API keys required)

Plain Python, no containers. This is what graders should run.

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # see requirements.txt, avoids a CUDA pull
pip install -r requirements.txt
cp .env.example .env                # every key can stay blank
make reproduce                      # CACHE_ONLY=1, replays every LLM call from the committed cache
```

**Timed on a clean checkout (`rm -rf venv`, all steps above) on 2026-09-13: 6m48s end to end**
(`venv` build + every `pip install`, real wall-clock via `time`) — well inside the 15-minute
budget `CLAUDE.md` sets. No network access happens during `make reproduce` itself — see
`pipeline/cache.py` and `CLAUDE.md` rule 1.

That same timed run surfaced a real, pre-existing gap worth knowing about: `make reproduce`
currently exits early, a few seconds in, once it reaches the `simple` baseline — it needs 1,000
silver-labelled pairs and only 590 are cached (`docs/DECISIONS.md` #48), a Groq daily-quota limit
hit earlier — nothing to do with how the environment is set up. The `trivial` baseline runs to completion first (all
150 rows). Rerun after the quota resets, or `python eval/run_eval.py --systems trivial,ours
--skip-judge` to exercise the actual pipeline (enrich/rules/retrieve/draft/check) without hitting
that gap.

Outputs land in `eval/results/`: `summary.md` (the comparison table across `trivial`, `simple`,
and `ours`), `summary.json` (the same data, machine-readable), `*_confusion.png`, and
`*_predictions.jsonl` per system.

### Run the API and UI locally

```bash
make data && make index      # skip if data/ and data/chroma/ are already populated (they are, here)
make api                     # terminal 1 — uvicorn on :8000, http://localhost:8000/docs
cd frontend && npm install && npm run dev   # terminal 2 — Vite on :5173
```

`.env.example` keys: `GROQ_API_KEY`, `GEMINI_API_KEY`, `PRIMARY_MODEL`, `FALLBACK_MODEL`,
`JUDGE_MODEL`, `CACHE_ONLY`, `AMOUNT_LIMIT`, `DB_PATH`, `RATE_SLEEP`. `make reproduce` works with
every key left blank; live use (`make api` against real messages) needs `GROQ_API_KEY` at least.

## Architecture

See `docs/HANDBOOK.md` for the full walkthrough (data, design principles, every pipeline stage,
storage, API, evaluation, decisions) and `docs/PROJECT_PLAN.md` for the original design brief.

Pipeline: `cache lookup → enrich (LLM) → rules (deterministic) → retrieve (MiniLM + Chroma) →
draft (LLM) → safety check (LLM) → persist → respond`. The same entry point
(`pipeline/run.py`) serves both `api/main.py` and `eval/run_eval.py`, so what is measured is
exactly what is served.

## Data

Kaggle `thoughtvector/customer-support-on-twitter`, filtered to Uber_Support and everything it
replied to: 125,631 related tweets → 53,651 unique customer↔Uber pairs → 40,886 threads, Uber's
own tweets spanning Dec 2014 – Dec 2017. Extraction and cleaning scripts are in `scripts/`
(`extract_uber_chunked.py`, `split_replies.py`, `sample_golden.py`, `validate_data.py`,
`build_index.py`). Counts and validation are in `data/stats.json` and `data/validation.json`.

## Evaluation

```bash
python eval/run_eval.py                        # full run, needs API keys, hits the network on cache misses
python eval/run_eval.py --skip-judge --n 20     # quick smoke check, first 20 golden rows only
CACHE_ONLY=1 python eval/run_eval.py            # offline replay from the cache (what `make reproduce` runs)
```

## Decision log and report

`docs/DECISIONS.md` — every non-obvious choice, numbered, with reasons (56 entries as of this
commit). `docs/REPORT.md` — problem framing, baselines, failure analysis, what's misleading
about the headline number, what's next — **planned, not yet written**.

## Citations

Kaggle dataset · sentence-transformers (MiniLM `all-MiniLM-L6-v2`) · ChromaDB · Groq API
(`openai/gpt-oss-120b` / `openai/gpt-oss-20b`) · Gemini API (`gemini-3.5-flash-lite`, judge
only) · scikit-learn · help.uber.com (paraphrased, not scraped). Anything adapted from library
docs, a blog post, or a known snippet is listed in `docs/BORROWED.md`.

## Tests

```bash
make test    # 100 pytest tests, offline, fake LLM, temp DB — a couple of seconds
```
