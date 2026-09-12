"""
Escalation rules: an ordered list of (name, predicate) over an Enrichment. First match wins and its name is the reason.
Pure functions, no LLM: the model reads the message and sets flags; these decide where the ticket goes.
"""
import os
from collections.abc import Callable

from pipeline.models import Enrichment
from pipeline.taxonomy import AMOUNT_LIMIT, AUTO_ALLOW

# PROVISIONAL: confidence is uncalibrated (smoke runs cluster at 0.93-0.97). Set from golden-set results.
CONFIDENCE_FLOOR = 0.7


def amount_limit() -> float:
    return float(os.environ.get("AMOUNT_LIMIT", AMOUNT_LIMIT))


RULES: list[tuple[str, Callable[[Enrichment], bool]]] = [
    ("safety", lambda e: e.entities.mentions_safety),
    ("legal", lambda e: e.entities.mentions_legal),
    ("amount_over", lambda e: e.entities.amount is not None and e.entities.amount > amount_limit()),
    ("urgent", lambda e: e.urgency == "high"),
    ("repeat_contact", lambda e: e.entities.is_repeat_contact),
    ("low_confidence", lambda e: e.confidence < CONFIDENCE_FLOOR),
    ("not_allowed", lambda e: e.intent not in AUTO_ALLOW),
]


def evaluate(enrichment: Enrichment) -> str | None:
    """Name of the first rule that fires, or None when the ticket may be auto-handled."""
    return next((name for name, fires in RULES if fires(enrichment)), None)
