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


# --- explanation only: what each rule looked at, for the trace. Never consulted by evaluate(). ---

URGENCY_ORDER = {"low": 0, "medium": 1, "high": 2}


def _clip(x: float) -> float:
    return round(min(1.0, max(0.0, x)), 3)


def _amount_margin(amount: float | None) -> float:
    limit = amount_limit()
    return 1.0 if amount is None or limit <= 0 else _clip((limit - amount) / limit)


# rule -> (value checked, threshold as text, margin 0..1 = how far the value is from firing; booleans are 0 or 1)
CHECKS: dict[str, Callable[[Enrichment], tuple[object, str, float]]] = {
    "safety": lambda e: (e.entities.mentions_safety, "is true", 1.0),
    "legal": lambda e: (e.entities.mentions_legal, "is true", 1.0),
    "amount_over": lambda e: (e.entities.amount, f"> {amount_limit():g}", _amount_margin(e.entities.amount)),
    "urgent": lambda e: (e.urgency, "== high", _clip((2 - URGENCY_ORDER[e.urgency]) / 2)),
    "repeat_contact": lambda e: (e.entities.is_repeat_contact, "is true", 1.0),
    "low_confidence": lambda e: (e.confidence, f"< {CONFIDENCE_FLOOR}", _clip((e.confidence - CONFIDENCE_FLOOR) / (1 - CONFIDENCE_FLOOR))),
    "not_allowed": lambda e: (e.intent.value, "not in AUTO_ALLOW", 1.0),
}


def explain(enrichment: Enrichment) -> list[dict]:
    """One row per rule, in RULES order, evaluated even after an earlier rule fired. `decisive` marks evaluate()'s answer."""
    reason = evaluate(enrichment)
    rows = []
    for name, fires in RULES:
        value, threshold, margin = CHECKS[name](enrichment)
        fired = bool(fires(enrichment))
        rows.append({"rule": name, "fired": fired, "value": value, "threshold": threshold,
                     "margin": 0.0 if fired else margin, "decisive": name == reason})
    return rows
