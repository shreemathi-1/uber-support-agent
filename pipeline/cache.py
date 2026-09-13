"""
Content-addressed cache for every LLM call (CLAUDE.md rule 1).
Key = SHA-256 of model + JSON-dumped messages; CACHE_ONLY=1 turns a miss into an error instead of a network call.
"""
import hashlib
import json
import os
from collections.abc import Callable
from contextvars import ContextVar

from pipeline.db import get_conn, init_db

Messages = list[dict[str, str]]
CallFn = Callable[[str, Messages], str]


STATS = {"hits": 0, "misses": 0}  # process-wide counters; run.py logs the per-request delta
# Every call since run.py last cleared it: {model, hit, messages, response}. The trace reads it; bounded per request.
LAST_CALLS: list[dict] = []
# Per-request cache-only switch for the API's X-Cache-Only header. A ContextVar, not os.environ, so one replay
# request cannot flip a concurrent normal request into cache-only mode (sync endpoints run in a thread pool).
FORCE_CACHE_ONLY: ContextVar[bool] = ContextVar("force_cache_only", default=False)


class CacheMissError(RuntimeError):
    """Raised on a cache miss when CACHE_ONLY=1. Means reproduction would need the network."""


def cache_only() -> bool:
    return FORCE_CACHE_ONLY.get() or os.environ.get("CACHE_ONLY", "0") == "1"


def cache_key(model: str, messages: Messages) -> str:
    payload = model + json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_llm(model: str, messages: Messages, call_fn: CallFn) -> str:
    """Return the cached response for (model, messages), or call `call_fn(model, messages)` and store it."""
    init_db()
    key = cache_key(model, messages)
    with get_conn() as conn:
        row = conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
    if row is not None:
        STATS["hits"] += 1
        LAST_CALLS.append({"model": model, "hit": True, "messages": messages, "response": row["response"]})
        return row["response"]
    STATS["misses"] += 1
    if cache_only():
        raise CacheMissError(f"CACHE_ONLY=1 and no cached response for model={model} key={key[:12]}")
    response = call_fn(model, messages)
    LAST_CALLS.append({"model": model, "hit": False, "messages": messages, "response": response})
    if not response.strip():  # never cache an empty completion (token budget eaten by reasoning, transient error)
        return response
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, model, messages_json, response) VALUES (?, ?, ?, ?)",
            (key, model, json.dumps(messages, sort_keys=True, ensure_ascii=False), response),
        )
    return response
