# Paste this into Claude Code as the first message (after CLAUDE.md is at repo root)

Read CLAUDE.md and docs/PROJECT_PLAN.md fully.

Task for this session: scaffold the repo only. No LLM calls yet.

1. Create the directory layout from CLAUDE.md. `data/` already contains uber_pairs.jsonl,
   uber_threads.jsonl, uber_tweets.csv, stats.json, and help_center/*.md; `pipeline/taxonomy.py`
   already exists — do not modify either.
2. `requirements.txt` pinned: fastapi, uvicorn[standard], pydantic>=2, groq, google-generativeai,
   sentence-transformers, chromadb, tenacity, python-dotenv, scikit-learn, pandas, matplotlib, pytest.
3. `.env.example` with GROQ_API_KEY, GEMINI_API_KEY, PRIMARY_MODEL=llama-3.3-70b-versatile,
   FALLBACK_MODEL=llama-3.1-8b-instant, JUDGE_MODEL=gemini-2.5-flash-lite, CACHE_ONLY=0, AMOUNT_LIMIT=10.
4. `pipeline/models.py` with the Pydantic models exactly as in CLAUDE.md (Entities, Enrichment,
   Turn, ChatRequest, ChatResponse, Context, RetrievedPair, RetrievedChunk). Import Intent from taxonomy.
5. `pipeline/db.py`: `get_conn()`, `init_db()` creating tables tickets, escalations, llm_cache.
6. `pipeline/cache.py`: `cached_llm(model, messages, call_fn)` with SHA-256 key, SQLite storage,
   CACHE_ONLY behaviour per CLAUDE.md rule 1.
7. `tests/test_models.py` and `tests/test_cache.py` (cache hit, cache miss, CACHE_ONLY raises).
8. `Makefile` with targets data, index, eval, reproduce, api, test (bodies can call scripts that
   don't exist yet; use `@echo TODO` where needed).
9. `.gitignore`: data/app.db, .env, __pycache__, data/chroma/*.lock, frontend/node_modules.

Before writing, list the files you will create. After writing, run `pytest tests/` and paste the output.
