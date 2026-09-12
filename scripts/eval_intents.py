"""
Intent metrics for the enrich step on data/golden_set.csv, or a smoke test on the first N rows if nothing is labelled yet.
Labelled path: accuracy, per-class P/R/F1, macro-F1, confusion table + eval/results/intent_confusion.png.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python scripts/eval_intents.py` finds `pipeline`

from pipeline.cache import CacheMissError  # noqa: E402
from pipeline.enrich import enrich
from pipeline.models import Enrichment, Turn
from pipeline.taxonomy import Intent

GOLDEN = Path("data/golden_set.csv")
RESULTS = Path("eval/results")


def history_of(row: dict) -> list[Turn]:
    return [Turn(role="user", content=row["history"])] if row["history"] else []


def run_rows(rows: list[dict]) -> list[Enrichment]:
    out: list[Enrichment] = []
    for i, row in enumerate(rows, 1):
        try:
            out.append(enrich(row["text"], history_of(row)))
        except CacheMissError as err:
            sys.exit(f"CACHE_ONLY=1 and row {row['id']} is not cached: {err}")
        print(f"\r{i}/{len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return out


def smoke(rows: list[dict]) -> None:
    print("NO LABELS: this is a smoke test, not a result.\n")
    print(f"{'id':5} {'message':60} {'intent':30} {'conf':4} {'urg':6} entities")
    for row, enr in zip(rows, run_rows(rows)):
        ents = {k: v for k, v in enr.entities.model_dump().items() if v not in (None, False)}
        print(f"{row['id']:5} {row['text'][:58]:60} {enr.intent.value:30} {enr.confidence:.2f} {enr.urgency:6} {ents}")


def evaluate(rows: list[dict]) -> None:
    # Heavy deps imported here so the smoke path works before scikit-learn/matplotlib are installed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score

    labels = [i.value for i in Intent]
    y_true = [row["intent"].strip() for row in rows]
    y_pred = [enr.intent.value for enr in run_rows(rows)]

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(f"accuracy={accuracy_score(y_true, y_pred):.3f}  macro_f1={f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0):.3f}  n={len(rows)}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    short = [l[:12] for l in labels]
    print("\nconfusion (rows=true, cols=pred):")
    print(f"{'':14}" + "".join(f"{s:>13}" for s in short))
    for name, counts in zip(short, cm):
        print(f"{name:14}" + "".join(f"{c:>13}" for c in counts))

    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=short).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "intent_confusion.png", dpi=120)
    (RESULTS / "intent_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {RESULTS / 'intent_confusion.png'} and {RESULTS / 'intent_metrics.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="rows for the smoke test when nothing is labelled")
    args = ap.parse_args()

    with GOLDEN.open(newline="") as f:
        rows = list(csv.DictReader(f))
    labelled = [r for r in rows if r["intent"].strip()]
    if labelled:
        evaluate(labelled)
    else:
        smoke(rows[: args.n])


if __name__ == "__main__":
    main()
