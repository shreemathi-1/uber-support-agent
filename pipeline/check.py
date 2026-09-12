"""
Safety check: the fallback model judges the draft and returns {safe, why}. Fails closed: any error means unsafe.
Catches promised money or timelines, invented policy, links or DM requests, and rudeness.
"""
import json
from functools import partial

from pipeline import llm
from pipeline.cache import CallFn, cached_llm
from pipeline.models import RetrievedChunk

SYSTEM_PROMPT = """You are a safety reviewer for Uber's in-app support replies. Return JSON only: {"safe": true|false, "why": "<one short sentence>"}.
Mark safe=false if the draft does any of these:
- promises or implies a refund, credit, adjustment, or a timeline for a resolution
- states a policy or procedure that is not supported by the supplied facts
- includes a link or URL, or asks the customer to DM, send a note, or contact Uber somewhere else
- is rude, dismissive, or sarcastic
- is empty, or cut off mid-sentence
Otherwise safe=true."""


def default_call_fn() -> CallFn:
    return partial(llm.groq_chat, temperature=0.0, max_tokens=120, json_mode=True, reasoning_effort="low")


def check(message: str, draft: str, facts: list[RetrievedChunk], call_fn: CallFn | None = None) -> tuple[bool, str]:
    if not draft.strip():
        return False, "empty_draft"
    fact_text = "\n".join(f"[{f.title}] {f.text}" for f in facts) or "(none supplied)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Facts:\n{fact_text}\n\nCustomer message: {message}\n\nDraft reply: {draft}"},
    ]
    try:
        raw = cached_llm(llm.fallback_model(), messages, call_fn or default_call_fn())
        data = json.loads(raw)
        return bool(data["safe"]), str(data.get("why", ""))
    except Exception as err:  # fail closed
        return False, f"check_error: {type(err).__name__}: {err}"
