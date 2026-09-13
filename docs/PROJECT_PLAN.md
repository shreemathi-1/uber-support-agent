# Uber Support AI Agent — Project Plan

SDE Intern take-home. Working reference, updated 12 Sep 2026 (rev 2: bootstrap-first build order).

---

## 1. The assignment in one paragraph

Pick one brand from the Kaggle "Customer Support on Twitter" dataset and build an AI support agent that (1) classifies each customer message into intents defined from the data, (2) drafts a reply grounded in how the brand historically resolved similar issues, and (3) decides auto-handle vs escalate with a stated reason. Then prove it works: a hand-labelled golden set, an evaluation harness with automated metrics and an LLM judge validated against a human, two baselines, a failure analysis, a section on what is misleading about the headline number, and a decision log. Reproducible from the README in under 15 minutes. **The proof is worth more than the system.**

Brand chosen: **Uber_Support**.

## 2. What "good" means for this brand (problem framing)

Findings from the extracted data (53,651 unique customer→Uber pairs, 40,886 threads, Apr 2016 – Sep 2017):

- Uber's Twitter team is a **triage desk, not a resolution channel**. The top-15 exact-match reply templates are 16% of all replies; near-duplicates ("Send us a DM with your email so we can connect") are estimated at 60–75%.
- A minority of replies are informative (policy answers: ride pass eligibility, driver requirements, app availability).
- Customers frequently re-contact ("I did. No response.", "I already DM'd 30 mins ago") — Uber's humans sometimes re-sent the same template. The bot must not.

Therefore a **good reply**: acknowledges the specific problem in Uber's tone, gives one self-serve step from the help centre when one exists, asks for exactly the one identifier that intent needs, and states what happens next. It never promises money or timelines, and — because the UI is an in-app chat — never says "DM us".

A **good escalation decision**: catches every safety/legal/high-value/urgent/repeat case (recall first), and auto-handles only intents whose measured precision is high.

Deliberately **not built**: multi-turn memory beyond the last two turns, authentication, real account lookup, streaming, an autonomous tool-using agent loop, fine-tuning, a feedback/learning loop, SLA nudges, document reading (Hiver Copilot-style features).

## 3. Reference model: Hiver's six steps

The graders' own product model (from Hiver's "deploy AI support agents" video) maps directly onto the design:

| Hiver step | This project |
|---|---|
| 1. Auto-tagging | Intent classification (enrich step) |
| 2. Sentiment analysis | `sentiment` + `urgency` fields, same LLM call |
| 3. Extract key information | `entities` block, same LLM call |
| 4. Let AI handle tickets completely | Retrieve + draft + safety check, only for allow-listed intents |
| 5. Intelligent routing and escalation | Deterministic rule table over the enriched ticket → specialist queue |
| 6. Monitor performance | Evaluation harness (model metrics, not business KPIs) |

Key sentence from the video, adopted verbatim as design principle: *"The AI handles the detection and classification. Then your rules determine where each ticket goes."*

## 4. Architecture

### 4.1 High level

```
CLIENT   React + Vite + Tailwind      Uber Help chat screen  |  Specialist queue view
            |  JSON                              |                       |
BACKEND  FastAPI (Docker)             POST /chat  ·  GET /escalations  ·  GET /health
            |
         Pipeline run.py              enrich → rules → retrieve → draft → check
            |            |            |
         ChromaDB     SQLite       MiniLM embedder            (all local)
            |
LLM      Groq gpt-oss-120b (enrich, draft)  ·  Groq gpt-oss-20b (fallback, safety)  ·  Gemini Flash-Lite (judge)
            
OFFLINE  extract_uber.py → build_index.py → data/chroma/   ·   golden_set.csv   ·   help_center/*.md   ·   eval/run_eval.py
```

### 4.2 Pipeline, per message

1. **Cache lookup** — SQLite `llm_cache`, key = SHA-256(model + messages). Every LLM call goes through it. `CACHE_ONLY=1` forbids network.
2. **Enrich** — one Groq gpt-oss-120b call, JSON mode, temperature 0. Output:
   ```json
   {"intent": "...", "confidence": 0.0-1.0,
    "sentiment": "negative|neutral|positive", "urgency": "low|medium|high",
    "entities": {"email": null, "trip_date": null, "city": null, "amount": null,
                 "mentions_safety": false, "mentions_legal": false, "is_repeat_contact": false}}
   ```
   Includes last two history turns. Pydantic-validated; one retry with the error appended; on second failure → `intent=other, confidence=0`.
3. **Rules** — ordered list, first match wins, reason = rule name:
   `safety → legal → amount > 10 → urgency == high → repeat_contact → confidence < 0.7 → intent ∉ AUTO_ALLOW`.
   `AUTO_ALLOW` is chosen from golden-set precision, not by hand. Escalated messages stop here (one LLM call total).
4. **Retrieve** — MiniLM embeds the message. Chroma `pairs` collection, `where={"intent": X}`, top 5; `help` collection top 2; unfiltered fallback if < 3 hits.
5. **Draft** — Groq gpt-oss-120b, temp 0.3, ≤120 tokens. Fixed shape: acknowledge · self-serve step (if help chunk matches) · ask one identifier · next step. Style rules learned from Uber's templates; channel rule: in-app chat, never "DM us". Post-process: strip URLs, ≤280 chars, prepend repeat-contact acknowledgement if flagged.
6. **Safety check** — Groq gpt-oss-20b, `{safe, why}`. Flags promised refunds or invented policy. Fails closed (error → unsafe → escalate).
7. **Persist + respond** — `tickets` row (auto) or `escalations` row (escalate). `ChatResponse{intent, confidence, sentiment, urgency, entities, reply, action, reason, retrieved_ids, latency_ms}`.

### 4.3 Why this shape (answers to expected interview questions)

- **Why a fixed pipeline, not an agent loop?** No tools with side effects to choose between. Fixed order is measurable per stage and explainable line-by-line.
- **Why rules for escalation?** Auditable one-word reasons; 100% recall on enumerated conditions; thresholds changeable without retraining; cannot be talked out of escalating. The LLM still does the reading (flags); rules only pull the trigger.
- **Why one enrich call instead of three?** Same read of the message; zero extra latency or quota; one JSON to validate.
- **Why rules before retrieval?** Escalated messages should not spend two LLM calls on a draft nobody sends.
- **Why local SQLite/Chroma?** 15-minute reproduction from a clone; no accounts; cannot silently pause; data is a few hundred MB at most.
- **Why a different model family for the judge?** So the judge is not grading its own writing style.

## 5. Tech stack (all free tier, verified Sep 2026)

| Layer | Choice | Note |
|---|---|---|
| Language | Python 3.11 | |
| API | FastAPI + Uvicorn + Pydantic v2 | `/docs` for graders |
| LLM pipeline | Groq `openai/gpt-oss-120b` | free, no card; Llama 3.x decommissioned 2026-08-16 — cache everything |
| LLM fallback/safety | Groq `openai/gpt-oss-20b` | same free tier, higher throughput |
| LLM judge | Gemini `3.5-flash-lite` (2.5 retired for new users, Sep 2026) | free; Pro models are paid-only; free-tier data may be used by Google |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | CPU, 22 MB, offline |
| Vector store | ChromaDB persistent dir | committed to repo |
| Storage | SQLite (stdlib) | tickets, escalations, llm_cache |
| Eval | scikit-learn, pandas, matplotlib | |
| Frontend | React 18 + Vite + Tailwind | Vercel free, optional |
| Hosting | HF Spaces (Docker) | optional demo link; graders run locally |
| Repro | Docker Compose + Makefile, `CACHE_ONLY=1` | zero API keys needed |

Not used, on purpose: LangChain, LangGraph, hosted vector DBs, paid APIs, ORMs.

Plan B if 120B quota is painful: run enrich on gpt-oss-20b, compare macro-F1 on the golden set, switch if it holds (report the comparison).

## 6. Data

### 6.1 Extraction (`scripts/extract_uber_chunked.py`)
Streams `twcs.csv` in 100k-row chunks, three passes: Uber_Support rows → tweets they link to (hop 1) → follow-ups (hop 2). Outputs `uber_tweets.csv`, `uber_pairs.jsonl` (RAG corpus), `uber_threads.jsonl` (for labelling), `stats.json`.

Column semantics: `inbound=True` = customer (input); `inbound=False` + `author_id=Uber_Support` = human agent reply (imitation target).

### 6.2 Cleaning (deliberately light)
Remove `@mentions`, `t.co` links, HTML entities, `^XX` sign-offs, empty rows, exact duplicate customer texts, customer texts < 8 chars, non-English (count reported). Keep casing, punctuation, emoji, misspellings.

### 6.3 Validation checklist
- Counts + date range in README
- 30 random pairs checked by eye: reply answers the question
- Template ratio (exact + embedding-clustered) reported
- **No overlap** between golden set and retrieval corpus (leakage check)
- Length and language histograms

### 6.4 Reply split (`scripts/split_replies.py`)
Regex on "DM / send us a note / reach out / follow up / connect" → `deflection` vs `informative`. Informative pairs are the high-value RAG subset; deflection pairs teach which identifier to ask per intent.

### 6.5 Help centre (`data/help_center/*.md`)
Target: 25–40 articles from help.uber.com, **hand-curated and paraphrased**, 3–5 per intent. Front matter: `title, source_url, intent, date_checked`. Not scraped in bulk (ToS, noise).

**Bootstrap state:** 12 placeholder articles written from general knowledge, stamped `date_checked: UNVERIFIED-PLACEHOLDER`. The pipeline reads them like any other input. Replace one at a time with a hand-checked paraphrase, set the date, run `make index`. `grep -l UNVERIFIED data/help_center/` lists what remains.

## 7. Intent taxonomy (to be finalised from data)

Target 6–8 mutually exclusive, text-observable intents.

**Bootstrap state:** `pipeline/taxonomy.py` ships with 8 provisional intents guessed from a first look at the data, each with a definition, two examples and an `ask_for` string:

`fare_dispute` · `trip_or_driver_issue` · `app_or_account_issue` · `driver_onboarding_or_earnings` · `uber_eats` · `lost_item` · `policy_or_info_question` · `other`

The enrich prompt is generated from this file, so revising intents after the 100-thread read touches one file. Test before finalising: label 30 tweets; if hesitating on > 5, merge or redefine. Note: nothing is *trained* on the taxonomy — it is read at prompt time — so changing it needs no retraining, only a re-run of the golden set (cached calls are invalidated because the prompt text changes).

## 8. Golden set

- 150 rows, stratified across intents (oversample rare ones), sampled from `uber_threads.jsonl` with context.
- Columns: `text, history, intent, sentiment, urgency, escalate (y/n), reason, ideal_reply_notes`.
- Labelled by me; a second labeller does 50 rows → Cohen's kappa for inter-annotator agreement.
- Labelling note in README: sampling method, edge-case rules, disagreements and how resolved.

## 9. Evaluation harness (`eval/`)

- **Intent**: accuracy, per-class P/R/F1, macro-F1, confusion matrix PNG.
- **Escalation**: precision/recall/F1 with escalate positive; breakdown by rule reason; auto-handle rate.
- **Reply quality**: Gemini judge, rubric 1–5 on (right issue acknowledged, right identifier asked, self-serve step correct if given, no promises, tone). Judge vs 40 human-scored replies → kappa. Secondary: embedding similarity to Uber's real reply (tone signal only).
- **Baselines** (same golden set):
  - Trivial: majority intent · always escalate · most common template as reply.
  - Simple: TF-IDF + logistic regression (silver-labelled by the LLM on the corpus) · keyword escalation rules · nearest-neighbour historical reply verbatim.
- `make reproduce` runs everything with `CACHE_ONLY=1` and writes `eval/results/*.json|md`.

## 10. Known traps for the "misleading headline number" section

- Similarity to Uber's real reply rewards "DM us your email" for everything.
- Golden set is small (150) and self-labelled; kappa on 50 is the only external check.
- Escalation recall is measured only on conditions I enumerated.
- Judge and drafter are both LLMs; agreement with 40 human scores is thin evidence.
- Retrieval corpus and golden set are from the same period; distribution shift not tested.
- Nearest-neighbour reply baseline may score close to the LLM drafter on the judge.
- Any metric produced during the bootstrap phase (provisional taxonomy, placeholder articles, dummy labels) is a plumbing check and must not appear in the report.

## 11. Build order and time budget (rev 2 — pipeline first on bootstrap data)

Extraction is done (`stats.json` present). Manual data work is deferred; the pipeline is built against the provisional taxonomy and placeholder articles so that evaluation tooling exists first.

| # | Task | Time |
|---|---|---|
| 1 | Claude Code session 1: scaffold, models, db, cache, tests, Makefile | 0.25 d |
| 2 | Session 2: `split_replies.py`, `validate_data.py`, `sample_golden.py` (creates an *unlabelled* 150-row file) | 0.25 d |
| 3 | Session 3: `enrich.py` + `eval_intents.py`; smoke on 20 tweets | 0.5 d |
| 4 | Session 4: `build_index.py`, `retrieve.py`, `peek_retrieval.py` | 0.25 d |
| 5 | Session 5: `draft.py`, `rules.py`, `check.py`, `run.py`, `api/main.py`; curl smoke test | 0.75 d |
| 6 | Session 6: baselines, `run_eval.py`, `metrics.py`, `judge.py`, `kappa.py` — runs end-to-end on the unlabelled sample with dummy labels to prove the harness works | 0.75 d |
| 7 | **Manual:** read 100 threads + 100 informative pairs → revise `taxonomy.py` | 0.5 d |
| 8 | **Manual:** label the 150-row golden set; second labeller 50 rows | 1 d |
| 9 | **Manual:** replace placeholder help articles with hand-checked paraphrases; `make index` | 0.5 d |
| 10 | Re-run `make eval` on real labels; set `AMOUNT_LIMIT`, `AUTO_ALLOW`; hand-score 40 replies | 0.5 d |
| 11 | REPORT.md, DECISIONS.md, README, Docker Compose, `make reproduce` timed | 1 d |
| 12 | React UI (chat + queue) | remaining |

Rule: any number computed before step 8 is a **plumbing check**, not a result. Headline numbers come only from real labels and hand-checked articles.

## 12. Repo layout

```
uber-support-agent/
  data/        uber_pairs.jsonl  uber_threads.jsonl  golden_set.csv  help_center/*.md  chroma/
  scripts/     extract_uber_chunked.py  split_replies.py  build_index.py
  pipeline/    models.py taxonomy.py cache.py db.py enrich.py rules.py retrieve.py draft.py check.py run.py
  api/         main.py
  baselines/   trivial.py  simple.py
  eval/        run_eval.py  metrics.py  judge.py  kappa.py  results/
  frontend/    React + Vite + Tailwind
  report/      REPORT.md  DECISIONS.md
  Dockerfile  docker-compose.yml  Makefile  requirements.txt  .env.example  README.md
```

`.env.example`: `GROQ_API_KEY`, `GEMINI_API_KEY`, `PRIMARY_MODEL`, `FALLBACK_MODEL`, `CACHE_ONLY`.

## 13. Decision log (draft — expand to 10–15 with evidence)

1. Uber_Support chosen: high volume, repetitive resolution patterns, clear escalation signals.
2. Fixed pipeline over agent loop: no side-effect tools to choose between; per-stage measurability.
3. One enrich call returns intent + sentiment + urgency + entities: same read, no extra quota.
4. Rules decide escalation over LLM output: auditable, recall-guaranteed on enumerated cases (Hiver's own model).
5. Rules run before retrieval/draft: escalations cost one LLM call.
6. Light cleaning only; casing/emoji kept: sentiment and urgency signal.
7. Historical tweet pairs as primary grounding; hand-curated help articles as secondary: assignment wording + in-app usefulness.
8. Help articles paraphrased by hand, not scraped: ToS and noise.
9. "Triage is the resolution" framing for Uber; reply shape adapted for in-app chat (no "DM us").
10. Judge on a different model family (Gemini) from the drafter (Groq/Llama).
11. Reply quality scored by rubric, not similarity to Uber's reply: similarity rewards deflection templates.
12. Local SQLite + Chroma committed: 15-minute repro with zero accounts.
13. LLM cache with `CACHE_ONLY` mode: deterministic, network-free reproduction.
14. `AUTO_ALLOW` intents chosen from measured precision, not by hand.
15. UI built last: not a graded deliverable.
16. Built the pipeline on a provisional 8-intent taxonomy and 12 placeholder help articles before the manual data work, so evaluation tooling existed early; taxonomy revised and articles replaced on <date>. Bootstrap-phase numbers never reported.

## 14. Open items

- [x] Run extraction (stats.json: 53,651 pairs, 40,886 threads)
- [ ] Run reply split; record deflection vs informative counts
- [ ] Claude Code sessions 1–6 (pipeline + harness on bootstrap data)
- [ ] 100-thread read → revise `taxonomy.py`
- [ ] Replace all `UNVERIFIED-PLACEHOLDER` help articles
- [ ] Line up second labeller
- [ ] Decide `amount` threshold from data (distribution of amounts mentioned)
- [ ] Confirm current Groq/Gemini model IDs at build time
- [ ] Time `make reproduce` on a clean machine

## 15. Citations to include in the repo

Kaggle dataset (thoughtvector/customer-support-on-twitter) · sentence-transformers / MiniLM · ChromaDB · Groq and Gemini APIs · scikit-learn metrics · Hiver "AI support agents" video (design reference) · help.uber.com (paraphrased articles, URLs listed) · any prompt patterns or code borrowed, per file header.