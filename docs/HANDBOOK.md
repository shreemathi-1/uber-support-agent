# Uber Support Agent — Project Handbook

*For anyone joining the project. Read top to bottom in about fifteen minutes; each section maps to one or two slides. State of the repo as of 13 Sep 2026.*

---

## 1. The project in one paragraph

An AI customer-support agent for **Uber_Support**, built from the public Kaggle "Customer Support on Twitter" dataset. For every incoming customer message it does three things: **classifies** the message into one of eight intents, **drafts** a reply in Uber's voice grounded in Uber's own historical replies and a small hand-written help centre, and **decides** whether to auto-send that reply or **escalate** to a human, always with a one-word reason. The system is a fixed, seven-step Python pipeline with two LLM calls at most on the happy path and one on an escalation. The larger half of the project is the **proof**: a 150-row hand-labelled golden set, an evaluation harness, two baselines to beat, an LLM judge checked against a human, and a decision log. The whole thing reproduces offline in under fifteen minutes with no API keys because every LLM call is cached.

**One-sentence pitch:** *The LLM reads the message; deterministic rules decide where it goes; everything is measured.*

---

## 2. The problem and the data

### 2.1 Where the data comes from

| Step | Count |
|---|---|
| Rows in the raw Kaggle CSV (all brands) | 2,811,774 |
| Tweets related to Uber_Support (three extraction hops) | 125,631 |
| Unique customer → Uber reply pairs | 53,651 |
| Conversation threads | 40,886 |
| Uber_Support-authored tweets, date range | Dec 2014 – Dec 2017 |

A **pair** is one customer tweet and the Uber reply to it. A **thread** is the whole back-and-forth. Column semantics: `inbound=True` is a customer message (the pipeline's input); `inbound=False` with `author_id=Uber_Support` is a human agent reply (what we imitate).

### 2.2 What the data taught us (this shapes the whole design)

- **Uber's Twitter team is a triage desk, not a resolution channel.** The top 15 exact reply templates are 16% of all replies, and after normalising near-duplicates the top 30 cover 26%. Most replies are some version of "send us a DM with your email".
- Only **13.6%** of replies are *informative* (an actual answer). The rest are *deflections*. We split them with a regex (`scripts/split_replies.py`) into `pairs_informative.jsonl` (7,301) and `pairs_deflection.jsonl` (46,350).
- Customers **re-contact** often ("I did. No response."). Uber's humans sometimes re-sent the same template. Our bot must not.
- Real procedures ("open Your Trips → Help → report a lost item") are **not in the tweets**. They come from a hand-written help centre (`data/help_center/*.md`, 12 articles).

### 2.3 Consequence for "what is a good reply"

A good reply **acknowledges the specific problem** in Uber's tone, gives **one self-serve step** when the help centre has one, asks for **exactly one identifier** that intent needs (trip date, account email, order number), and says **what happens next**. It never promises money or timelines and, because our channel is an in-app chat, **never says "DM us"**.

A good escalation decision **catches every** safety, legal, high-value, urgent and repeat-contact case (recall first) and auto-handles only intents whose measured precision is high.

---

## 3. Design principles

1. **Proof over system.** The grader values the evaluation more than the agent. Every number in the report must come from real labels, never from the bootstrap phase.
2. **Fixed pipeline, not an agent loop.** There are no tools with side effects to choose between, so an autonomous loop adds cost and unpredictability without adding capability. A fixed order is measurable per stage and explainable line by line.
3. **The LLM reads, the rules decide.** Borrowed from Hiver's support-automation model: *"The AI handles the detection and classification. Then your rules determine where each ticket goes."* Rules give auditable one-word reasons, 100% recall on enumerated conditions, thresholds that change without retraining, and cannot be talked out of escalating.
4. **One enrich call, not three.** Intent, sentiment, urgency and entities come from a single JSON response. Same read of the message, one JSON to validate, a third of the quota.
5. **Escalate before you spend.** Rules run *before* retrieval and drafting, so an escalated message costs one LLM call.
6. **Everything local, everything cached.** SQLite, ChromaDB on disk, MiniLM on CPU. Every LLM call goes through a content-addressed cache; `CACHE_ONLY=1` makes a network call an error.
7. **Small and obvious.** Plain Python functions, no framework, no ORM, no LangChain. Files stay under ~200 lines with a two-sentence docstring at the top.
8. **Don't over-clean.** Casing, emoji, misspellings are sentiment and urgency signal. Only mentions, links, HTML entities, agent sign-offs and whitespace are stripped.

Mapping to Hiver's six steps, which the grader uses as a mental model:

| Hiver step | Here |
|---|---|
| Auto-tagging | Intent (enrich step) |
| Sentiment analysis | `sentiment`, `urgency` (same call) |
| Extract key information | `entities` block (same call) |
| Let AI handle tickets | Retrieve → draft → safety check, only for allow-listed intents |
| Intelligent routing | Deterministic rule table → specialist queue |
| Monitor performance | Eval harness + Trace page |

---

## 4. Architecture

### 4.1 Big picture

```
 CLIENT     React 18 + Vite + Tailwind        Help (chat) | Queue (escalations) | Trace (per-ticket steps)
               │ JSON over HTTP
 API        FastAPI  api/main.py             POST /chat · GET /escalations · POST /escalations/{id}/resolve
                                             GET /traces · GET /traces/{id} · GET /metrics · GET /health
               │
 PIPELINE   pipeline/run.py                  cache → enrich → rules → retrieve → draft → check → persist
               │            │           │
 LOCAL      SQLite       ChromaDB     MiniLM embedder (CPU)
            tickets      pairs 20,000
            escalations  help 12 chunks
            llm_cache
               │
 LLM (Groq) openai/gpt-oss-120b  enrich + draft      openai/gpt-oss-20b  safety check + fallback
 JUDGE      Gemini 3.5 Flash-Lite  eval only, never in the pipeline

 OFFLINE    extract → split_replies → sample_golden → validate_data → build_index → run_eval
```

### 4.2 One request, end to end

```
customer message (+ last 2 turns)
   │
   ▼
[1] ENRICH   one Groq call, JSON, temperature 0         → intent, confidence, sentiment, urgency, entities
   │
   ▼
[2] RULES    seven ordered pure-Python checks           → reason (rule name) or None
   │  reason?  ──yes──▶ ESCALATE: template reply + ticket reference, escalation row, stop (1 LLM call)
   │  intent == other? ──yes──▶ canned reply, stop (1 LLM call)
   ▼
[3] RETRIEVE MiniLM embed → Chroma: 5 similar pairs (filtered by intent) + 2 help chunks
   │
   ▼
[4] DRAFT    one Groq call, temperature 0.3            → reply in the fixed shape, post-processed (≤280 chars, no links)
   │
   ▼
[5] CHECK    one Groq call on the smaller model        → {safe, why}; any error = unsafe
   │  unsafe? ──yes──▶ ESCALATE with reason "unsafe_draft" (draft kept for the specialist)
   ▼
[6] PERSIST  tickets row (+ escalations row) + full trace JSON
   │
   ▼
ChatResponse {intent, confidence, sentiment, urgency, entities, reply, action, reason, retrieved_ids, latency_ms}
```

Typical cost: **3 LLM calls** on an auto-handled message, **1** on an escalation, **0** on a cache replay. Latency on a warm cache is ~250 ms; with live calls 1–4 s.

---

## 5. Stage by stage

Every stage is one module in `pipeline/`. Each takes a Pydantic model in and hands one out (`pipeline/models.py`); nothing else crosses a stage boundary.

### 5.1 Cache — `pipeline/cache.py`

- **What:** `cached_llm(model, messages, call_fn)` looks up SHA-256(model + JSON-dumped messages, sorted keys) in the SQLite `llm_cache` table. Hit → return stored text. Miss → call `call_fn`, store, return.
- **Why:** deterministic, network-free reproduction. `make reproduce` runs the whole eval with `CACHE_ONLY=1` and zero API keys.
- **Details:** sampling parameters are not in the key because they are fixed per stage. Empty completions are never cached (the model once spent its whole token budget on reasoning). Each call is also appended to a per-request `LAST_CALLS` list that the trace reads. A `ContextVar` lets one API request run cache-only without touching the environment (the Trace page's Replay button).

### 5.2 Enrich — `pipeline/enrich.py`

- **In:** message + up to two history turns. **Out:** `Enrichment` (intent, confidence 0–1, sentiment, urgency, entities).
- **Model/params:** `openai/gpt-oss-120b`, temperature 0, JSON mode, `reasoning_effort="low"`, max 300 tokens.
- **Prompt:** generated from `taxonomy.py`, so changing an intent definition needs no prompt edit. Compacted to 621 tokens because prompt tokens are paid on every call.
- **Entities:** `email, trip_date, city, amount, mentions_safety, mentions_legal, is_repeat_contact`. The last three are the flags the rules pull the trigger on, and each has a one-line definition in the prompt so the model's judgement is pinned down (e.g. "vehicle smell alone is not safety").
- **Failure handling:** invalid JSON → one retry with the validation error appended → on a second failure `intent=other, confidence=0`, which the rules then escalate as `low_confidence`. Currency strings like "£4.25" in `amount` are coerced to a float by a validator so they parse first time.

### 5.3 Rules — `pipeline/rules.py`

Ordered list, first match wins, the rule name is the reason. Pure functions, no LLM.

| # | Rule | Fires when | Why this order |
|---|---|---|---|
| 1 | `safety` | `mentions_safety` | Never auto-handle danger |
| 2 | `legal` | `mentions_legal` | Police, lawyer, fraud, regulator |
| 3 | `amount_over` | `amount > AMOUNT_LIMIT` (10, env-overridable) | Money above a threshold needs a human |
| 4 | `urgent` | `urgency == high` | Stranded, time-critical |
| 5 | `repeat_contact` | `is_repeat_contact` | Do not re-send a template to someone already ignored |
| 6 | `low_confidence` | `confidence < 0.7` | Classifier unsure → human |
| 7 | `not_allowed` | intent ∉ `AUTO_ALLOW` | Only intents with measured precision are auto-handled |

`AUTO_ALLOW` today: `policy_or_info_question, driver_onboarding_or_earnings, app_or_account_issue, lost_item, other`. Fare disputes, trip/driver issues and Eats problems always escalate for now. Both `AUTO_ALLOW` and `AMOUNT_LIMIT` are **provisional** and will be set from golden-set precision, not guessed.

`explain()` also reports every rule's checked value and a 0–1 *margin* (how close it was to firing) for the Trace page; `evaluate()` itself is untouched.

### 5.4 Retrieve — `pipeline/retrieve.py`

- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2`, CPU, normalised vectors, cosine distance (0 identical, 2 opposite).
- **Collections (ChromaDB, `data/chroma/`):** `pairs` = all ~7.3k informative pairs plus a seeded sample of deflection pairs, capped at 20,000; `help` = the 12 help-centre articles chunked at ~220 words.
- **Query:** top 5 pairs filtered by `intent` metadata, top 2 help chunks. If the filtered query returns fewer than 3 pairs (or 0 chunks) or the best hit is farther than 0.6, re-query unfiltered.
- **Note:** the `intent` metadata on pairs is a keyword-bucket guess (`pipeline/buckets.py`), a retrieval hint, never a label. No bucket covers `policy_or_info_question`, so that intent always takes the unfiltered path.

### 5.5 Draft — `pipeline/draft.py`

- **Model/params:** `gpt-oss-120b`, temperature 0.3, `reasoning_effort="low"`, max 400 tokens (160 returned empty drafts; the model reasons before it writes).
- **Prompt shape, fixed:** (a) acknowledge the specific issue, (b) one self-serve step *only if* a help chunk within distance 0.6 was supplied, (c) ask for exactly one identifier (the `ask_for` string of the intent), (d) what happens next. Style examples are the retrieved pairs, informative first, marked "tone only, do not copy links or DM requests".
- **Hard rules in the prompt:** in-app chat, never "DM us"; never promise refund, credit or timeline; never state a policy not in the facts.
- **Post-processing:** strip URLs, collapse whitespace, prefix "Sorry you've had to contact us again." for repeat contacts, cut at 280 characters on a sentence boundary. The trace records which of these changed the text.
- **`intent == other`:** no LLM draft at all; a fixed canned line. Escalating "thanks" wastes a specialist and drafting for a rant invites invention.

### 5.6 Safety check — `pipeline/check.py`

- **Model:** `gpt-oss-20b` (the smaller, faster one), temperature 0, JSON `{safe, why}`.
- **Checks:** promised money or timeline, policy not in the facts, links or DM requests, rudeness, empty or truncated text.
- **Fails closed:** empty draft → unsafe without calling the model; any exception or unparseable JSON → unsafe. Unsafe → escalate with reason `unsafe_draft`, draft kept so the specialist can see it.

### 5.7 Persist and respond — `pipeline/run.py`, `pipeline/db.py`

- Every request writes a `tickets` row. Escalations add an `escalations` row pointing back by `ticket_id`, carrying the draft (null when a rule fired first). The customer sees "Reference: ticket N".
- The full **trace** dict (`docs/TRACE_SPEC.md`) is stored as JSON on the ticket: what each step saw and produced, per-step LLM calls and cache hits, token estimates, a list-price cost estimate, and what the two baselines would have done.
- `run()` is the **single entry point** used by both the API and the eval harness, so what is measured is exactly what is served.

---

## 6. The offline data pipeline

```
twcs.csv (Kaggle, 2.8M rows)
   │  scripts/extract_uber_chunked.py   100k-row chunks, 3 hops (Uber replies → tweets they answer → follow-ups)
   ▼
uber_tweets.csv · uber_pairs.jsonl · uber_threads.jsonl · stats.json
   │  scripts/split_replies.py          regex: deflection vs informative
   ▼
pairs_deflection.jsonl (46,350) · pairs_informative.jsonl (7,301)
   │  scripts/sample_golden.py          150 rows, seed 42, stratified by 6 keyword buckets (min 15 each), one row per thread;
   │                                    removes those tweets AND their history turns from the corpus (no leakage)
   ▼
golden_set.csv (unlabelled; suggested_* columns filled by the enricher as hints)
   │  scripts/validate_data.py          counts, template share, lengths, English share, LEAKAGE CHECK (exit 1 on overlap)
   ▼
   │  scripts/build_index.py            embed + write Chroma `pairs` and `help`
   ▼
data/chroma/
```

`make data` runs split → sample → validate. `make index` builds Chroma. Cleaning is deliberately light: `@mentions`, `t.co` links, HTML entities, `^XX` sign-offs and whitespace only.

**Help centre** (`data/help_center/*.md`): 12 hand-written paraphrases with `title, source_url, intent, date_checked` front matter. They are **never scraped** (terms of service, noise). Today all 12 are still stamped `UNVERIFIED-PLACEHOLDER` and must be replaced one at a time with hand-checked text, then `make index`.

**Golden set** (`data/golden_set.csv`): 150 rows (g001–g150), 45 with a history turn. Columns `id, customer_tweet_id, text, history, intent, sentiment, urgency, escalate, reason, ideal_reply_notes, labeller` plus five `suggested_*` hint columns. **Ground-truth columns are filled by a human only, never by code.** The sampler refuses to overwrite a file that has any label in it.

---

## 7. Intent taxonomy — `pipeline/taxonomy.py`

Eight mutually exclusive, text-observable intents. Each has a definition, two real-style examples (the shorter one goes into the prompt) and the single identifier the drafter asks for.

| Intent | Covers | Asks for | Auto-allowed today |
|---|---|---|---|
| `fare_dispute` | wrong fare, double charge, cancellation fee, surge, refund | trip date and city | no |
| `trip_or_driver_issue` | no-show, detour, rude or unsafe driver, vehicle, wait, stranded | trip date + short description | no |
| `app_or_account_issue` | login, codes, app errors, payment method, locked account | account email | yes |
| `driver_onboarding_or_earnings` | signup, background check, documents, payouts | driver-application email | yes |
| `uber_eats` | missing/wrong/late order, Eats promo | order number or Eats email | no |
| `lost_item` | left something in a vehicle | trip date | yes |
| `policy_or_info_question` | pricing, ride pass, availability, requirements | nothing, answer from help centre | yes |
| `other` | rant, compliment, spam, too short | nothing (canned reply) | yes |

The taxonomy is read at prompt time, so revising it needs no retraining, only a golden-set re-run (which invalidates the enrich cache because the prompt text changes).

---

## 8. Storage — `pipeline/db.py`

SQLite via the standard library, file `data/app.db` (path from `DB_PATH`), no ORM.

| Table | One row per | Key columns |
|---|---|---|
| `tickets` | request | conversation_id, message, intent, confidence, sentiment, urgency, entities_json, reply, action, reason, retrieved_ids, latency_ms, **trace** (JSON) |
| `escalations` | escalated request | ticket_id, message, intent, urgency, reason, draft (nullable), retrieved_ids, status open/resolved |
| `llm_cache` | distinct LLM call | key (SHA-256), model, messages_json, response |

`init_db()` is idempotent and adds columns older databases lack (the `trace` column was added this way).

---

## 9. API — `api/main.py`

| Method, path | Purpose |
|---|---|
| `POST /chat` | Run the pipeline. Body `{message, history[≤2], conversation_id?}`. Header `X-Cache-Only: 1` answers from cache only (409 on a miss). |
| `GET /escalations?status=open` | Specialist queue |
| `POST /escalations/{id}/resolve` | Mark resolved |
| `GET /traces?limit=50` | Newest tickets with intent, action, reason, latency |
| `GET /traces/{ticket_id}` | Full trace JSON |
| `GET /metrics` | `eval/results/summary.json` if present, else `{"available": false}` |
| `GET /health` | Model ids, index sizes, cache row count |

No auth. CORS open to any localhost port. Interactive docs at `/docs`. Start with `make api`.

---

## 10. Frontend — `frontend/`

React 18 + Vite + Tailwind 4, no router, no state library, three tabs (`make ui`, port 5173):

- **Help** — the rider chat. Sends the message plus the last two turns. Auto replies are bubbles; escalations are a green card with the rule name and ticket reference.
- **Queue** — open escalations with intent, urgency, reason, the draft if one exists, and a Resolve button.
- **Trace** — ticket list on the left; the selected ticket as a vertical stepper on the right: enrich JSON with entity chips, the rules table with a green/grey dot per rule and its checked value, retrieved pairs and chunks with distance bars (dimmed above 0.6), draft raw vs final, checker verdict, the reply as the customer saw it, a "why not?" line (for auto: the two rules closest to firing; for escalate: the draft or "escalated before drafting"), a totals footer, a baselines comparison strip, an eval-metrics tile, and a "Replay (cached)" button.

The UI was built last, on purpose: it is not a graded deliverable and reads only the endpoints above.

---

## 11. Evaluation — `eval/`

### 11.1 What is measured

| Dimension | Metric | Where |
|---|---|---|
| Intent | accuracy, per-class P/R/F1, macro-F1, confusion-matrix PNG | `metrics.intent_metrics` |
| Escalation | precision / recall / F1 with escalate positive, breakdown per rule reason (true vs false escalations), auto-handle rate | `metrics.escalation_metrics` |
| Reply quality | Gemini judge, five rubric items scored 1–5 (acknowledges issue, asks right identifier, self-serve step correct or absent, no promises or links, tone), total 5–25 | `judge.py` |
| Judge validity | Cohen's kappa, judge vs 40 human-scored replies, on low/mid/high buckets fixed in advance | `kappa.judge_vs_human` |
| Label validity | Cohen's kappa, labeller A vs B on 50 rows, intent and escalate | `kappa.labeller_vs_labeller` |
| Latency | mean ms per system | `run_eval.summarise` |

The judge scores **auto-handled replies only**; the escalation template is not a drafted reply. The judge is a **different model family** (Gemini) from the drafter (Groq gpt-oss) so it is not grading its own style. Embedding similarity to Uber's real reply is reported as a tone signal only, because it rewards "DM us your email" for everything.

### 11.2 Baselines (same golden set, same metrics)

- **Trivial** (`baselines/trivial.py`): majority intent, always escalate, the single most common deflection template as the reply. The floor; if it scores well on a metric, that metric is misleading.
- **Simple** (`baselines/simple.py`): TF-IDF + logistic regression trained on 1,000 *silver* labels (the enricher's own predictions on corpus pairs, produced on the 20B model so it does not compete with the system for quota), keyword rules for escalation (police, lawyer, fraud, unsafe, stranded, "already dm", dollar amount over the limit), and the nearest historical Uber reply verbatim. What a week-one build without an LLM in the loop looks like. Inherits the enricher's errors; the report must say so.
- **Ours**: `pipeline.run`.

### 11.3 How to run

```
make eval        # baselines + system over the golden set → eval/results/{summary.json, summary.md, *_predictions.jsonl, *_confusion.png}
make reproduce   # the same with CACHE_ONLY=1: no keys, no network, must finish < 15 min
```

Until the golden set is labelled the harness prints **"NO LABELS — plumbing check only"** and intent/escalation metrics are empty. Any number from that phase must not appear in the report.

### 11.4 What is misleading about the headline number (write this section honestly)

- The golden set is small (150) and self-labelled; kappa on 50 rows is the only external check.
- Escalation recall is measured only on the conditions we enumerated.
- Judge and drafter are both LLMs; agreement with 40 human scores is thin evidence.
- Retrieval corpus and golden set come from the same period; distribution shift is untested.
- Uber's replies are mostly deflections, so a "similar to Uber" score rewards the wrong thing.
- Confidence is uncalibrated (it clusters at 0.93–0.99), so the 0.7 floor may never fire; the per-reason breakdown decides whether that rule stays.

---

## 12. Reproducibility, quota and cost

- **Free tier only.** Groq gpt-oss-120b allows roughly 200k tokens per day. One full golden-set pass (150 rows × up to 3 calls × ~1k tokens) is most of a day's allowance, so silver labelling, the golden run and prompt changes must land on different days, or the cache does the work.
- **Rate limiting.** Batch scripts call `llm.batch_mode()` which sets a 2 s sleep between calls; the API runs at 0. Transient 429/5xx errors retry three times with exponential backoff, then fall back to the 20B model once. A daily-quota 429 stops the batch immediately with a resume hint; progress is cached, rerun the same command tomorrow.
- **Cache invalidation is by prompt text.** Any change to a system prompt or to `taxonomy.py` wording changes every key for that stage. Budget a re-run before editing prompts.
- **Cost estimate in the trace** uses Groq list prices (120B $0.15/$0.60, 20B $0.075/$0.30 per million tokens in/out) with a chars/4 token estimate. It answers "what would this have cost", the free tier pays nothing.
- **Determinism.** Enrich at temperature 0; rules are pure; seeds fixed (42) in sampling, corpus capping and baselines.

---

## 13. Repository map

```
PROJECT_GUIDE.md          standing engineering instructions (architecture is fixed; hard rules)
Makefile                  data · index · eval · reproduce · api · test · ui
requirements.txt          pinned, Python 3.11, free-tier stack only
.env.example              GROQ_API_KEY, GEMINI_API_KEY, model ids, CACHE_ONLY, AMOUNT_LIMIT, DB_PATH, RATE_SLEEP

data/
  uber_tweets.csv, uber_pairs.jsonl, uber_threads.jsonl   extracted Uber subset
  pairs_informative.jsonl, pairs_deflection.jsonl          reply split
  golden_set.csv            150 rows, human labels + enricher suggestions
  help_center/*.md          12 hand-written articles with source_url front matter
  chroma/                   vector index (pairs, help)
  stats.json, validation.json                              counts and sanity report
  app.db                    SQLite (gitignored)

scripts/
  extract_uber_chunked.py   raw CSV → Uber subset (already run; output committed)
  split_replies.py          deflection vs informative
  sample_golden.py          stratified 150-row sample; --suggest fills suggested_* hint columns
  validate_data.py          sanity report + leakage check
  build_index.py            Chroma index
  peek_retrieval.py, eval_intents.py, smoke_chat.py, check_models.py, measure_prompt.py   dev tools

pipeline/
  models.py     Pydantic contracts (Entities, Enrichment, ChatRequest, ChatResponse, Context, ...)
  taxonomy.py   intents, definitions, examples, ask_for, AMOUNT_LIMIT, AUTO_ALLOW
  buckets.py    keyword buckets (coverage steering, retrieval hint)
  llm.py        the only module that talks to Groq: retries, fallback, daily-quota stop
  cache.py      content-addressed LLM cache, CACHE_ONLY, per-request call log
  db.py         SQLite schema and queries
  enrich.py     classify + extract (1 LLM call)
  rules.py      escalation rules + explain()
  retrieve.py   MiniLM + Chroma
  draft.py      reply drafting + post-processing
  check.py      safety check
  trace.py      trace builders
  pricing.py    list prices + token estimate
  run.py        the pipeline entry point

api/main.py     FastAPI surface
baselines/      trivial.py, simple.py
eval/           run_eval.py, metrics.py, judge.py, kappa.py, results/
frontend/src/   App.jsx, Help.jsx, Queue.jsx, Trace.jsx, TraceSteps.jsx
tests/          100 pytest tests, fake LLM, temp DB, no network, ~2 s
docs/           PROJECT_PLAN.md, DECISIONS.md, TRACE_SPEC.md, BORROWED.md, HANDBOOK.md (this file)
```

---

## 14. Decisions worth knowing (digest of `docs/DECISIONS.md`, 47 entries)

**Shape**
- Fixed pipeline over agent loop; rules over LLM for escalation; rules before retrieval; one enrich call (#2–#5 in the plan).
- `other` intent is auto-allowed with a canned reply and no draft (#28).
- One identifier per intent; two-part `ask_for` strings produced two-question replies (#30).

**Data**
- Golden-set unit is a customer turn, not a thread root; no root tweet in the data replies to anything, so history had to come from second turns (#6).
- Both the sampled turn and its history turn are removed from the corpus, stricter than the leakage rule requires (#7).
- Deflection regex widened after eyeballing; informative share fell from 19.5% to 13.6% (#11).
- Sampler refuses to overwrite a golden set with any label in it (#10).

**Models and prompts**
- Switched to `gpt-oss-120b/20b` when Groq decommissioned Llama 3.x on 16 Aug 2026 (#16). Judge moved to Gemini 3.5 Flash-Lite when 2.5 was retired for new users (#42).
- Draft needed `reasoning_effort=low` and 400 tokens; the first live run returned three empty drafts and the checker passed two of them, which led to the "empty draft is unsafe" guard and "never cache an empty completion" (#27).
- Enrich prompt compacted 986 → 621 tokens; the last 70 would have cost classification behaviour (#41).
- Repeat-contact definition tightened: waiting on a decision is not repeat contact (#29).
- `Entities.amount` accepts "£4.25"-style strings via a validator rather than a prompt change, to keep cache keys stable (#47).

**Cache and quota**
- Key = SHA-256(model + sorted-key JSON of messages); sampling params excluded (#2).
- Fallback responses are cached under the primary model's key; reproduction replays whatever answered (#13).
- Daily-quota 429 stops the batch, never retried or downgraded silently (#39).
- Silver labels for the simple baseline come from the 20B model to protect the 120B quota (#40).

**Eval**
- Judge scores auto-handled replies only (#35); reason breakdown counts true and false escalations per rule (#36); kappa buckets fixed before any scores existed (#37).
- `retrieved_ids` carry `kind:id:distance` so the judge sees exactly the facts the drafter saw (#34).

**Trace**
- Per-request call log in `cache.LAST_CALLS`; `explain()` evaluates all rules; `draft_raw` + `postprocess_with_notes`; ContextVar for per-request cache-only; tokens are a labelled chars/4 estimate because tiktoken needs a download and the cache stores no usage (#44, #45).

---

## 15. Testing

`make test` runs 100 pytest tests in about two seconds, all offline:

- A `FakeGroq` that dispatches on the system prompt serves enrich, draft and check from one object and records which stages ran.
- `DB_PATH` points at a temp file per test; retrieval is monkey-patched with a fixed `Context`.
- Covered: cache key stability and `CACHE_ONLY` behaviour; every rule in order; enrich retry and fallback; draft post-processing; the full `run()` on the escalate, auto, unsafe-draft, empty-draft and `other` paths; trace shape (eight keys, one row per rule, cache hits on replay); DB migration; baselines; metrics; judge parsing; reply split.

---

## 16. Where the project stands (13 Sep 2026)

**Done**
- Extraction, reply split, golden sample, validation, Chroma index.
- Full pipeline, API, three-tab UI including the Trace page.
- Eval harness, both baselines, judge, kappa tooling; harness proven end to end on unlabelled data.
- Golden set: 150 rows labelled by hand (35% intent override vs the enricher's suggestion); all five `suggested_*` hint columns filled.
- 100 tests passing.
- First `make eval` on real labels: trivial baseline scored on 150 rows (escalation F1 0.90 from "always escalate", the expected trap); the simple baseline stopped at silver label 590/1000 on the 20B daily quota, so the system run is still pending.

**Not done, in order**
1. **Fix the invalid label values** in the golden set (junk intents, blank or non-y/n escalate, free-text reasons, empty labeller). Second labeller on 50 rows.
2. **Replace the 12 placeholder help articles** with hand-checked paraphrases, set `date_checked`, `make index`.
3. **Run `make eval` after the Groq daily reset**: fits the simple baseline from the cached silver labels, runs the system and the judge; then hand-score 40 replies and set `AMOUNT_LIMIT` and `AUTO_ALLOW` from the per-reason precision.
4. **Write `docs/REPORT.md`**: problem framing, baselines, failure analysis, what's misleading about the headline number, what's next.
5. Optional: revisit the 0.6 retrieval distance cutoff with `peek_retrieval.py` output; consider storing real token usage in the cache.

---

## 17. Questions you will be asked (and the short answers)

- **Why not an agent loop?** No side-effect tools to choose between; a fixed order is measurable per stage and explainable line by line.
- **Why rules instead of asking the LLM whether to escalate?** Auditable one-word reasons, guaranteed recall on enumerated cases, thresholds tunable without retraining, cannot be talked out of it. The LLM still does the reading.
- **Why one enrich call?** Same read, no extra latency or quota, one JSON to validate.
- **Why rules before retrieval?** Escalations should not spend two calls on a draft nobody sends.
- **Why SQLite and Chroma on disk?** Fifteen-minute reproduction from a clone, no accounts, cannot silently pause.
- **Why a different model family for the judge?** So it is not grading its own writing style.
- **Why are tweets not enough for replies?** They are 86% deflections; the actual procedures come from the hand-written help centre.
- **Why "never DM us"?** The channel is an in-app chat; the customer is already talking to us.
- **What is the biggest risk to the numbers?** Small, self-labelled golden set; escalation recall only on enumerated conditions; judge is an LLM too.

---

## 18. Suggested slide outline

1. Title + one-sentence pitch
2. The assignment and the grading emphasis (proof > system)
3. The data in numbers (§2.1) and the triage-desk finding (§2.2)
4. What "good" means (§2.3)
5. Design principles + Hiver mapping (§3)
6. Architecture diagram (§4.1)
7. One request end to end (§4.2)
8. Enrich: prompt, output, failure handling (§5.2)
9. Rules table (§5.3)
10. Retrieve + draft: fixed reply shape, hard rules (§5.4–5.5)
11. Safety check and fail-closed (§5.6)
12. Storage, API, UI screenshots (§8–10), Trace page demo
13. Evaluation design: metrics, judge, kappa, baselines (§11)
14. Reproducibility and quota discipline (§12)
15. What is misleading about the headline number (§11.4)
16. Decisions that changed the design (§14 highlights)
17. Status and next steps (§16)
18. Demo script: send "left my wallet…" (auto with help step), "driver was drunk…" (safety escalation), "the app won't send a code…" (auto), then open Trace and Replay (cached).

---

## 19. Glossary

- **Pair** — one customer tweet and Uber's reply. **Thread** — the full conversation.
- **Deflection / informative** — a reply that only redirects ("DM us") vs one that answers.
- **Enrichment** — the structured output of the classify-and-extract LLM call.
- **Reason** — the name of the rule that escalated, `unsafe_draft`, or `auto`.
- **Golden set** — the 150 hand-labelled rows used for every metric.
- **Silver labels** — the enricher's own predictions used to train the simple baseline; not truth.
- **Judge** — Gemini scoring replies on a five-item rubric.
- **Trace** — per-ticket record of what every step saw and produced.
- **CACHE_ONLY** — mode in which any LLM cache miss is an error; how `make reproduce` runs without keys.
- **Plumbing check** — a run on unlabelled or placeholder data; proves the code path, never a reported number.
