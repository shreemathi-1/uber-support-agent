"""
Builders for the per-request trace dict that run.py stores next to every ticket (shape: docs/TRACE_SPEC.md).
Pure formatting over objects the pipeline already produced; nothing here influences a decision.
"""
from pipeline import pricing
from pipeline.draft import MAX_FACT_DISTANCE
from pipeline.models import Context

LLM_STEPS = ("enrich", "draft", "check")


def llm_step(calls: list[dict], latency_ms: int, **extra) -> dict:
    """Summarise one stage's LLM calls (entries of cache.LAST_CALLS). `extra` adds or overrides fields."""
    prompt = sum(pricing.count_tokens(m["content"]) for c in calls for m in c["messages"])
    completion = sum(pricing.count_tokens(c["response"]) for c in calls)
    cost = sum(pricing.estimate_cost_usd(c["model"], sum(pricing.count_tokens(m["content"]) for m in c["messages"]),
                                         pricing.count_tokens(c["response"])) for c in calls)
    base = {
        "ran": bool(calls), "model": calls[-1]["model"] if calls else None, "llm_calls": len(calls),
        "cache_hits": sum(1 for c in calls if c["hit"]), "prompt_tokens": prompt, "completion_tokens": completion,
        "estimated_paid_cost_usd": round(cost, 6), "latency_ms": latency_ms,
        "raw_output": calls[-1]["response"] if calls else None,
    }
    return {**base, **extra}


def retrieve_step(context: Context | None, intent: str, latency_ms: int) -> dict:
    """context=None means retrieval was skipped (a rule fired first, or intent = other)."""
    ctx = context or Context()
    return {
        "ran": context is not None, "latency_ms": latency_ms, "intent_filter": intent, "max_fact_distance": MAX_FACT_DISTANCE,
        "pairs": [{"id": e.id, "distance": round(e.score, 3), "source": e.source, "intent": e.intent,
                   "customer_text": e.customer_text, "reply_text": e.reply_text} for e in ctx.examples],
        "chunks": [{"id": f.id, "title": f.title, "distance": round(f.score, 3), "intent": f.intent, "source_url": f.source_url,
                    "text": f.text, "usable": f.score <= MAX_FACT_DISTANCE} for f in ctx.facts],
    }


def totals(trace: dict, latency_ms: int) -> dict:
    steps = [trace[k] for k in LLM_STEPS]
    return {
        "llm_calls": sum(s["llm_calls"] for s in steps), "cache_hits": sum(s["cache_hits"] for s in steps),
        "prompt_tokens": sum(s["prompt_tokens"] for s in steps), "completion_tokens": sum(s["completion_tokens"] for s in steps),
        "latency_ms": latency_ms, "estimated_paid_cost_usd": round(sum(s["estimated_paid_cost_usd"] for s in steps), 6),
        "token_counter": pricing.TOKEN_COUNTER,
    }
