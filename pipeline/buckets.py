"""
Cheap keyword buckets for customer text. Used to steer golden-set coverage and as a retrieval hint on indexed pairs.
Never a label: the enrich step decides intent; these are regexes.
"""
import re

# First match wins. Short keys keep the golden-set sampling order stable (sample_golden sorts by bucket name).
BUCKETS: list[tuple[str, re.Pattern]] = [
    ("fare", re.compile(r"fare|charge|refund", re.I)),
    ("trip", re.compile(r"driver|trip|ride", re.I)),
    ("app", re.compile(r"\bapp\b|log ?in|\bcode\b|account", re.I)),
    ("driver_side", re.compile(r"apply|background|earnings|bank", re.I)),
    ("eats", re.compile(r"eats|order|food", re.I)),
    ("lost", re.compile(r"lost|left|wallet|phone", re.I)),
]

BUCKET_TO_INTENT: dict[str, str] = {
    "fare": "fare_dispute",
    "trip": "trip_or_driver_issue",
    "app": "app_or_account_issue",
    "driver_side": "driver_onboarding_or_earnings",
    "eats": "uber_eats",
    "lost": "lost_item",
    "other": "other",
}


def bucket_of(text: str) -> str:
    return next((name for name, pat in BUCKETS if pat.search(text)), "other")


def bucket_intent(text: str) -> str:
    """Bucket mapped onto an Intent value, for the `intent` metadata on indexed pairs. No bucket covers policy questions."""
    return BUCKET_TO_INTENT[bucket_of(text)]
