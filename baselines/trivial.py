"""
Trivial baseline: majority intent, always escalate, and the single most common deflection template as the reply.
The floor every other system must beat; if it scores well on a metric, that metric is misleading.
"""
import json
from collections import Counter
from pathlib import Path

DEFLECTION_PATH = Path("data/pairs_deflection.jsonl")


def most_common_reply(path: Path = DEFLECTION_PATH) -> str:
    counts = Counter(json.loads(line)["brand_reply"] for line in path.open())
    return counts.most_common(1)[0][0] if counts else ""


class TrivialBaseline:
    def __init__(self, labelled_intents: list[str], deflection_path: Path = DEFLECTION_PATH) -> None:
        self.intent = Counter(labelled_intents).most_common(1)[0][0] if labelled_intents else "other"
        self.reply = most_common_reply(deflection_path)

    def predict(self, row: dict) -> dict:
        return {"intent": self.intent, "escalate": True, "reason": "trivial_always", "reply": self.reply}
