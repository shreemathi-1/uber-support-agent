"""eval/metrics.py on tiny hand-built arrays with answers worked out by hand."""
from eval.metrics import auto_handle_rate, escalation_metrics, intent_metrics


def test_intent_metrics_by_hand() -> None:
    m = intent_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], labels=["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert m["per_class"]["b"] == {"precision": 0.6667, "recall": 1.0, "f1": 0.8, "support": 2}
    assert m["macro_f1"] == 0.7333
    assert m["confusion"]["matrix"] == [[1, 1], [0, 2]]


def test_intent_metrics_handles_unseen_label_without_error() -> None:
    m = intent_metrics(["a", "b"], ["a", "c"], labels=["a", "b", "c"])
    assert m["per_class"]["b"]["recall"] == 0.0 and m["per_class"]["c"]["support"] == 0


def test_escalation_metrics_and_reason_breakdown() -> None:
    m = escalation_metrics([True, True, False, False], [True, False, True, False], ["safety", "auto", "low_confidence", "auto"])
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)
    assert m["by_reason"]["safety"] == {"count": 1, "true_escalations": 1, "false_escalations": 0}
    assert m["by_reason"]["low_confidence"] == {"count": 1, "true_escalations": 0, "false_escalations": 1}
    assert "auto" not in m["by_reason"]


def test_escalation_metrics_all_negative_does_not_crash() -> None:
    m = escalation_metrics([False, False], [False, False], ["auto", "auto"])
    assert m["precision"] == 0.0 and m["by_reason"] == {}


def test_auto_handle_rate() -> None:
    assert auto_handle_rate([{"escalate": True}, {"escalate": False}, {"escalate": False}, {"escalate": False}]) == 0.75
    assert auto_handle_rate([]) == 0.0
