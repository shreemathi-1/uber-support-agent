"""
Trivial baseline: majority intent, always escalate, and the single most common deflection template as the reply.
The floor every other system must beat; if it scores well on a metric, that metric is misleading.
"""
import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

DEFLECTION_PATH = Path("data/pairs_deflection.jsonl")
GOLDEN_PATH = Path("data/golden_set.csv")  # read-only here: the majority intent comes from its human labels


def most_common_reply(path: Path = DEFLECTION_PATH) -> str:
    counts = Counter(json.loads(line)["brand_reply"] for line in path.open())
    return counts.most_common(1)[0][0] if counts else ""


class TrivialBaseline:
    def __init__(self, labelled_intents: list[str], deflection_path: Path = DEFLECTION_PATH) -> None:
        self.intent = Counter(labelled_intents).most_common(1)[0][0] if labelled_intents else "other"
        self.reply = most_common_reply(deflection_path)

    def predict(self, row: dict) -> dict:
        return {"intent": self.intent, "escalate": True, "reason": "trivial_always", "reply": self.reply}


@lru_cache(maxsize=1)
def _default() -> TrivialBaseline | None:
    """Built once per process from the data files; None when they are missing (tests, fresh checkout)."""
    if not DEFLECTION_PATH.exists():
        return None
    labels = []
    if GOLDEN_PATH.exists():
        with GOLDEN_PATH.open(newline="") as f:
            labels = [r["intent"].strip() for r in csv.DictReader(f) if r["intent"].strip()]
    return TrivialBaseline(labels, DEFLECTION_PATH)


def predict_one(message: str) -> dict:
    """The trivial baseline's answer for one message, for the trace. Majority intent is "other" until the golden set is labelled."""
    baseline = _default()
    if baseline is None:
        return {"available": False, "why": f"{DEFLECTION_PATH} missing; run make data"}
    return {"available": True, **baseline.predict({"text": message})}
