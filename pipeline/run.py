"""
The single pipeline entry point used by the API and the eval harness: enrich -> rules -> retrieve -> draft -> check -> persist.
Escalations stop after the rules so they cost one LLM call; every request writes a tickets row.
"""
import json
import logging
import time
import uuid

from pipeline import cache, db, llm, rules
from pipeline.check import check
from pipeline.draft import draft, usable_facts
from pipeline.enrich import enrich
from pipeline.models import ChatRequest, ChatResponse, Context
from pipeline.retrieve import retrieve
from pipeline.taxonomy import Intent

log = logging.getLogger(__name__)

ESCALATION_REPLY = "Thanks for flagging this. A support specialist will review it and follow up with you here."
OTHER_REPLY = ("Thanks for the message. If there's something we can help with on a trip or your account, "
               "just tell us what happened.")


def run(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    stats0 = dict(cache.STATS)
    db.init_db()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    enrichment = enrich(request.message, request.history)
    reason = rules.evaluate(enrichment)
    context, draft_text, check_why = Context(), None, ""
    if reason is None and enrichment.intent is Intent.OTHER:
        draft_text = OTHER_REPLY  # canned: nothing to ground, nothing to ask for (DECISIONS.md #28)
    elif reason is None:
        context = retrieve(request.message, enrichment.intent.value)
        draft_text = draft(request.message, request.history, enrichment, context)
        safe, check_why = check(request.message, draft_text, usable_facts(context))
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
    if action == "escalate":
        db.insert_escalation({
            "ticket_id": ticket_id, "conversation_id": conversation_id, "message": request.message,
            "intent": enrichment.intent.value, "urgency": enrichment.urgency, "reason": reason,
            "entities_json": entities_json, "draft": draft_text, "retrieved_ids": json.dumps(retrieved_ids),
        })
        reply = f"{ESCALATION_REPLY} Reference: ticket {ticket_id}."
        db.set_ticket_reply(ticket_id, reply)
    else:
        reply = draft_text or ""

    log.info(json.dumps({
        "ticket_id": ticket_id, "conversation_id": conversation_id, "intent": enrichment.intent.value,
        "confidence": enrichment.confidence, "action": action, "reason": reason or "auto", "check_why": check_why,
        "models": {"enrich": llm.primary_model(), "draft": llm.primary_model() if context.examples else None,
                   "check": llm.fallback_model() if context.examples else None},
        "cache": {k: cache.STATS[k] - stats0[k] for k in cache.STATS}, "latency_ms": latency_ms,
    }))
    return ChatResponse(
        conversation_id=conversation_id, intent=enrichment.intent, confidence=enrichment.confidence,
        sentiment=enrichment.sentiment, urgency=enrichment.urgency, entities=enrichment.entities,
        reply=reply, action=action, reason=reason or "auto", retrieved_ids=retrieved_ids, latency_ms=latency_ms,
    )
