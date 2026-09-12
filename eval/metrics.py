"""
Plain-dict metrics for intent classification and escalation decisions, plus the confusion matrix PNG.
Escalation is the positive class throughout; the per-reason breakdown shows which rule over- or under-fires.
"""
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def intent_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None,
                   png_path: Path | None = None) -> dict:
    labels = labels or sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if png_path:
        save_confusion_png(cm, labels, png_path)
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(float(f.mean()), 4),
        "per_class": {lab: {"precision": round(float(pi), 4), "recall": round(float(ri), 4),
                            "f1": round(float(fi), 4), "support": int(si)}
                      for lab, pi, ri, fi, si in zip(labels, p, r, f, s)},
        "confusion": {"labels": labels, "matrix": cm.tolist()},
    }


def save_confusion_png(cm, labels: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=[l[:14] for l in labels]).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def escalation_metrics(y_true: list[bool], y_pred: list[bool], reasons: list[str]) -> dict:
    """P/R/F1 with escalate positive, and per reason: how many fired, how many were true and false escalations."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=True, zero_division=0)
    by_reason: dict[str, dict] = defaultdict(lambda: {"count": 0, "true_escalations": 0, "false_escalations": 0})
    for truth, pred, reason in zip(y_true, y_pred, reasons):
        if pred:
            by_reason[reason]["count"] += 1
            by_reason[reason]["true_escalations" if truth else "false_escalations"] += 1
    return {"n": len(y_true), "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
            "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1]["count"]))}


def auto_handle_rate(preds: list[dict]) -> float:
    return round(sum(1 for p in preds if not p["escalate"]) / len(preds), 4) if preds else 0.0
