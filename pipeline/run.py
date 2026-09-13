"""
The single pipeline entry point used by the API and the eval harness: enrich -> rules -> retrieve -> draft -> check -> persist.
Escalations stop after the rules so they cost one LLM call; every request writes a tickets row plus its trace (docs/TRACE_SPEC.md).
"""
import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar

from baselines import simple, trivial
from pipeline import cache, db, rules
from pipeline import trace as trace_mod
from pipeline.check import check
from pipeline.draft import draft_raw, postprocess_with_notes, usable_facts
from pipeline.enrich import enrich
from pipeline.models import ChatRequest, ChatResponse, Context
from pipeline.retrieve import retrieve
from pipeline.taxonomy import Intent

log = logging.getLogger(__name__)
T = TypeVar("T")

ESCALATION_REPLY = "Thanks for flagging this. A support specialist will review it and follow up with you here."
OTHER_REPLY = ("Thanks for the message. If there's something we can help with on a trip or your account, "
               "just tell us what happened.")


def _stage(fn: Callable[[], T]) -> tuple[T, list[dict], int]:
    """Run one step; return (result, the LLM calls it made, wall-clock ms). Feeds the trace, changes nothing."""
    n, t = len(cache.LAST_CALLS), time.perf_counter()
    result = fn()
    return result, cache.LAST_CALLS[n:], int((time.perf_counter() - t) * 1000)


def baseline_predictions(message: str) -> dict:
    """What the two baselines would have done with the same message. Outside the pipeline; not in latency_ms."""
    t = time.perf_counter()
    out = {"trivial": trivial.predict_one(message), "simple": simple.predict_one(message)}
    return {**out, "latency_ms": int((time.perf_counter() - t) * 1000)}


def run(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    db.init_db()
    cache.LAST_CALLS.clear()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    trace: dict = {"request": {"message": request.message, "history": [t.model_dump() for t in request.history],
                               "conversation_id": conversation_id,
                               "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}

    enrichment, calls, ms = _stage(lambda: enrich(request.message, request.history))
    trace["enrich"] = trace_mod.llm_step(calls, ms, parsed=enrichment.model_dump(mode="json"), retried=len(calls) > 1)

    reason = rules.evaluate(enrichment)
    trace["rules"] = {"reason": reason, "rows": rules.explain(enrichment)}

    context, draft_text, check_why = Context(), None, ""
    trace["retrieve"] = trace_mod.retrieve_step(None, enrichment.intent.value, 0)
    trace["draft"] = trace_mod.llm_step([], 0, final_reply=None, post_process_notes=[], canned=False)
    trace["check"] = trace_mod.llm_step([], 0, safe=None, why=None)
    if reason is None and enrichment.intent is Intent.OTHER:
        draft_text = OTHER_REPLY  # canned: nothing to ground, nothing to ask for (DECISIONS.md #28)
        trace["draft"].update(ran=True, final_reply=draft_text, canned=True)
    elif reason is None:
        context, _, ms = _stage(lambda: retrieve(request.message, enrichment.intent.value))
        trace["retrieve"] = trace_mod.retrieve_step(context, enrichment.intent.value, ms)
        raw, calls, ms = _stage(lambda: draft_raw(request.message, request.history, enrichment, context))
        draft_text, notes = postprocess_with_notes(raw, enrichment.entities.is_repeat_contact)
        trace["draft"] = trace_mod.llm_step(calls, ms, ran=True, raw_output=raw, final_reply=draft_text,
                                            post_process_notes=notes, canned=False)
        (safe, check_why), calls, ms = _stage(lambda: check(request.message, draft_text, usable_facts(context)))
        trace["check"] = trace_mod.llm_step(calls, ms, ran=True, safe=safe, why=check_why)
        if not safe:
            reason = "unsafe_draft"

    action = "escalate" if reason else "auto"
    # "kind:id:distance" so the eval judge can re-apply the drafter's distance filter without widening ChatResponse
    retrieved_ids = [f"pair:{e.id}:{e.score:.2f}" for e in context.examples] + [f"help:{f.id}:{f.score:.2f}" for f in context.facts]
    latency_ms = int((time.perf_counter() - t0) * 1000)
    entities_json = enrichment.entities.model_dump_json()

    ticket_id = db.insert_ticket({
        "conversation_id": conversation_id, "message": request.message, "intent": enrichment.intent.value,
        "confidence": enrichment.confidence, "sentiment": enrichment.sentiment, "urgency": enrichment.urgency,
        "entities_json": entities_json, "reply": draft_text or "", "action": action, "reason": reason or "auto",
        "retrieved_ids": json.dumps(retrieved_ids), "latency_ms": latency_ms,
    })
    escalation_id = None
    if action == "escalate":
        escalation_id = db.insert_escalation({
            "ticket_id": ticket_id, "conversation_id": conversation_id, "message": request.message,
            "intent": enrichment.intent.value, "urgency": enrichment.urgency, "reason": reason,
            "entities_json": entities_json, "draft": draft_text, "retrieved_ids": json.dumps(retrieved_ids),
        })
        reply = f"{ESCALATION_REPLY} Reference: ticket {ticket_id}."
        db.set_ticket_reply(ticket_id, reply)
    else:
        reply = draft_text or ""

    trace["action"] = {"action": action, "reason": reason or "auto", "reply": reply, "ticket_id": ticket_id,
                       "escalation_id": escalation_id}
    trace["totals"] = trace_mod.totals(trace, latency_ms)
    trace["baselines"] = baseline_predictions(request.message)
    db.set_ticket_trace(ticket_id, json.dumps(trace, ensure_ascii=False))

    log.info(json.dumps({
        "ticket_id": ticket_id, "conversation_id": conversation_id, "intent": enrichment.intent.value,
        "confidence": enrichment.confidence, "action": action, "reason": reason or "auto", "check_why": check_why,
        "models": {k: trace[k]["model"] for k in trace_mod.LLM_STEPS}, "totals": trace["totals"],
    }))
    return ChatResponse(
        conversation_id=conversation_id, intent=enrichment.intent, confidence=enrichment.confidence,
        sentiment=enrichment.sentiment, urgency=enrichment.urgency, entities=enrichment.entities,
        reply=reply, action=action, reason=reason or "auto", retrieved_ids=retrieved_ids, latency_ms=latency_ms,
    )
