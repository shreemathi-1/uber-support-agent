"""
Enrich step: one LLM call turns a customer message (plus up to two history turns) into a validated Enrichment.
The system prompt is generated from taxonomy.INTENT_DEFINITIONS, so changing the taxonomy needs no prompt edit.
"""
import json
import logging
from functools import partial

from pydantic import ValidationError

from pipeline import llm
from pipeline.cache import CallFn, cached_llm
from pipeline.models import Enrichment, Entities, Turn
from pipeline.taxonomy import INTENT_DEFINITIONS, Intent

log = logging.getLogger(__name__)

EXAMPLE_OUTPUT = json.dumps(
    {
        "intent": "fare_dispute",
        "confidence": 0.85,
        "sentiment": "negative",
        "urgency": "medium",
        "entities": {
            "email": None, "trip_date": "Saturday", "city": "Lexington KY", "amount": 13.0,
            "mentions_safety": False, "mentions_legal": False, "is_repeat_contact": False,
        },
    },
    indent=2,
)

INSTRUCTIONS = """Rules:
- Respond with JSON only, no prose, exactly the keys shown above.
- intent must be one of the intent names listed. confidence is your own probability (0 to 1) that the intent is correct.
- sentiment is the customer's tone: negative, neutral, or positive.
- urgency is high when the customer is stranded, mentions a safety issue, the matter is time-critical, or they are an angry repeat contact; medium for an unresolved problem; low for questions and feedback.
- entities.is_repeat_contact is true if the message says they already contacted, DM'd, or emailed Uber and got no reply, or the earlier turn shows a prior message on the same issue.
- entities.mentions_safety is true only for physical danger, harassment, assault, intoxication, reckless driving, or feeling unsafe. Vehicle smell or condition alone is false.
- entities.mentions_legal is true only if the customer mentions police, a lawyer, legal action, a lawsuit, theft, fraud, a scam, or a regulator/consumer body. Anger, sarcasm, or "unacceptable" alone is false.
- entities.amount is the numeric value if a currency amount is mentioned, else null. email, trip_date, city: copy from the message if present, else null."""

FALLBACK = Enrichment(intent=Intent.OTHER, confidence=0.0, sentiment="neutral", urgency="low", entities=Entities())


def build_system_prompt() -> str:
    lines = [
        "You classify messages sent to Uber customer support and extract structured fields for routing.",
        "",
        "Intents (choose exactly one):",
    ]
    for intent, spec in INTENT_DEFINITIONS.items():
        lines.append(f"- {intent.value}: {spec['definition']}")
        for example in spec["examples"]:
            lines.append(f'    example: "{example}"')
    lines += ["", "Output JSON in exactly this shape:", EXAMPLE_OUTPUT, "", INSTRUCTIONS]
    return "\n".join(lines)


def build_user_prompt(message: str, history: list[Turn]) -> str:
    parts = [f"Earlier ({t.role}): {t.content}" for t in history]
    parts.append(f"Message: {message}")
    return "\n".join(parts)


def default_call_fn() -> CallFn:
    return partial(llm.groq_chat, temperature=0.0, max_tokens=300, json_mode=True, reasoning_effort="low")


def enrich(message: str, history: list[Turn] | None = None, call_fn: CallFn | None = None) -> Enrichment:
    """Classify one message. One retry with the validation error appended; then a safe OTHER/0.0 fallback."""
    call_fn = call_fn or default_call_fn()
    system = {"role": "system", "content": build_system_prompt()}
    user_text = build_user_prompt(message, history or [])
    messages = [system, {"role": "user", "content": user_text}]
    for attempt in (1, 2):
        raw = cached_llm(llm.primary_model(), messages, call_fn)
        try:
            return Enrichment.model_validate_json(raw)
        except ValidationError as err:
            log.warning("enrich attempt %d returned invalid JSON for %r: %s", attempt, message[:60], err)
            messages = [system, {"role": "user", "content": f"{user_text}\n\nYour previous answer was invalid:\n{err}\nRespond again with valid JSON only."}]
    log.warning("enrich giving up on %r; returning intent=other, confidence=0", message[:60])
    return FALLBACK.model_copy()
