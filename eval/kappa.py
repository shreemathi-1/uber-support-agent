"""
Inter-rater agreement: judge vs human on reply totals, and labeller A vs labeller B on intent and escalate.
Each entry point returns None when its input file is absent, so the harness runs before those files exist.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

GOLDEN = Path("data/golden_set.csv")
GOLDEN_B = Path("data/golden_set_labeller_b.csv")
HUMAN_SCORES = Path("eval/human_scores.csv")           # columns: id, human_total
OURS_PREDICTIONS = Path("eval/results/ours_predictions.jsonl")


def cohen_kappa(a: list, b: list) -> float:
    return round(float(cohen_kappa_score(a, b)), 4)


def bucket(total: int) -> str:
    """Totals run 5-25. low <= 12, mid 13-19, high >= 20 (decision #37)."""
    return "low" if total <= 12 else "mid" if total <= 19 else "high"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def judge_vs_human(predictions: Path = OURS_PREDICTIONS, human: Path = HUMAN_SCORES) -> dict | None:
    if not (predictions.exists() and human.exists()):
        return None
    judged = {p["id"]: p["judge"]["total"] for p in map(json.loads, predictions.open()) if p.get("judge")}
    pairs = [(bucket(judged[r["id"]]), bucket(int(r["human_total"]))) for r in read_csv(human) if r["id"] in judged]
    if not pairs:
        return None
    a, b = zip(*pairs)
    return {"n": len(pairs), "kappa_bucketed": cohen_kappa(list(a), list(b))}


def labeller_vs_labeller(golden: Path = GOLDEN, other: Path = GOLDEN_B) -> dict | None:
    if not (golden.exists() and other.exists()):
        return None
    a_rows = {r["id"]: r for r in read_csv(golden)}
    shared = [(a_rows[r["id"]], r) for r in read_csv(other) if r["id"] in a_rows and r["intent"] and a_rows[r["id"]]["intent"]]
    if not shared:
        return None
    a, b = zip(*shared)
    return {"n": len(shared),
            "kappa_intent": cohen_kappa([x["intent"] for x in a], [y["intent"] for y in b]),
            "kappa_escalate": cohen_kappa([x["escalate"].lower() for x in a], [y["escalate"].lower() for y in b])}


def main() -> None:
    print("judge vs human:", judge_vs_human() or "eval/human_scores.csv or ours_predictions.jsonl missing")
    print("labeller A vs B:", labeller_vs_labeller() or "data/golden_set_labeller_b.csv missing or unlabelled")


if __name__ == "__main__":
    main()
