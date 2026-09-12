"""
Enrich step: one LLM call turns a customer message (plus up to two history turns) into a validated Enrichment.
The system prompt is generated from taxonomy.INTENT_DEFINITIONS, so changing the taxonomy needs no prompt edit.
"""
import logging
from functools import partial

from pydantic import ValidationError

from pipeline import llm
from pipeline.cache import CallFn, cached_llm
from pipeline.models import Enrichment, Entities, Turn
from pipeline.taxonomy import INTENT_DEFINITIONS, Intent

log = logging.getLogger(__name__)

# One line, defaults as literals: every token here is paid on every call (scripts/measure_prompt.py; DECISIONS.md #41)
SCHEMA = ('{"intent":"<intent>","confidence":0.0,"sentiment":"negative|neutral|positive","urgency":"low|medium|high",'
          '"entities":{"email":null,"trip_date":null,"city":null,"amount":null,"mentions_safety":false,'
          '"mentions_legal":false,"is_repeat_contact":false}}')

INSTRUCTIONS = """Rules: confidence = P(intent correct). urgency: high = stranded, safety, time-critical, or angry repeat contact; medium = unresolved problem; low = question or feedback. mentions_safety: physical danger, harassment, assault, intoxication, reckless driving, feeling unsafe; not vehicle smell or condition alone. mentions_legal: police, lawyer, lawsuit, legal action, theft, fraud, scam, regulator; not anger or "unacceptable" alone. is_repeat_contact: says they already contacted, messaged, emailed, or DM'd Uber with no reply, or history shows a prior message on the same issue; not merely waiting on an application or decision. amount: number if a currency amount is mentioned, else null. email, trip_date, city: from the message, else null."""

FALLBACK = Enrichment(intent=Intent.OTHER, confidence=0.0, sentiment="neutral", urgency="low", entities=Entities())


def build_system_prompt() -> str:
    lines = ["Classify a message to Uber support. Reply with JSON only, exactly this shape:", SCHEMA, "", "Intents (one), with an example each:"]
    for intent, spec in INTENT_DEFINITIONS.items():
        lines.append(f"- {intent.value}: {spec['definition']} e.g. \"{min(spec['examples'], key=len)}\"")
    lines += ["", INSTRUCTIONS]
    return "\n".join(lines)


def build_user_prompt(message: str, history: list[Turn]) -> str:
    parts = [f"Earlier ({t.role}): {t.content}" for t in history]
    parts.append(f"Message: {message}")
    return "\n".join(parts)


def default_call_fn() -> CallFn:
    return partial(llm.groq_chat, temperature=0.0, max_tokens=300, json_mode=True, reasoning_effort="low")


def enrich(message: str, history: list[Turn] | None = None, call_fn: CallFn | None = None,
           model: str | None = None) -> Enrichment:
    """Classify one message. One retry with the validation error appended; then a safe OTHER/0.0 fallback."""
    call_fn = call_fn or default_call_fn()
    model = model or llm.primary_model()
    system = {"role": "system", "content": build_system_prompt()}
    user_text = build_user_prompt(message, history or [])
    messages = [system, {"role": "user", "content": user_text}]
    for attempt in (1, 2):
        raw = cached_llm(model, messages, call_fn)
        try:
            return Enrichment.model_validate_json(raw)
        except ValidationError as err:
            log.warning("enrich attempt %d returned invalid JSON for %r: %s", attempt, message[:60], err)
            messages = [system, {"role": "user", "content": f"{user_text}\n\nYour previous answer was invalid:\n{err}\nRespond again with valid JSON only."}]
    log.warning("enrich giving up on %r; returning intent=other, confidence=0", message[:60])
    return FALLBACK.model_copy()
