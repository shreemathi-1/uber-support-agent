"""Trivial baseline shape and the simple baseline's keyword escalation rules. No network, no model fitting."""
import json
from pathlib import Path

import pytest

from baselines.simple import keyword_escalate
from baselines.trivial import TrivialBaseline


@pytest.fixture
def deflection_file(tmp_path: Path) -> Path:
    p = tmp_path / "defl.jsonl"
    rows = [{"brand_reply": "Send us a DM"}] * 3 + [{"brand_reply": "Here to help"}] * 2
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_trivial_uses_majority_intent_and_most_common_reply(deflection_file: Path) -> None:
    b = TrivialBaseline(["fare_dispute", "lost_item", "fare_dispute"], deflection_file)
    pred = b.predict({"text": "anything"})
    assert pred == {"intent": "fare_dispute", "escalate": True, "reason": "trivial_always", "reply": "Send us a DM"}


def test_trivial_defaults_to_other_without_labels(deflection_file: Path) -> None:
    assert TrivialBaseline([], deflection_file).predict({"text": "x"})["intent"] == "other"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I'm calling the police about this driver", "police"),
        ("already DM'd you twice", "already dm"),
        ("No response from anyone in 3 days", "no response"),
        ("you charged me $25 for nothing", "amount_over"),
        ("a $5 cancellation fee?", None),
        ("my driver was lovely", None),
    ],
)
def test_keyword_escalate(text: str, expected: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMOUNT_LIMIT", "10")
    assert keyword_escalate(text) == expected


def test_judge_parses_and_totals_with_fake_call_fn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eval.judge import judge
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.delenv("CACHE_ONLY", raising=False)
    fake = lambda m, ms: json.dumps({"scores": {"acknowledges_issue": 5, "asks_right_identifier": 4,
                                                "self_serve_step_correct_or_absent": 7, "no_promises_or_links": 5, "tone": 4},
                                     "rationale": "ok"})
    out = judge("lost phone", "Sorry about the phone. What was the trip date?", [], call_fn=fake)
    assert out["scores"]["self_serve_step_correct_or_absent"] == 5  # clamped
    assert out["total"] == 23 and out["rationale"] == "ok"


def test_silver_label_reuses_primary_cache_else_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from baselines.simple import silver_label
    from pipeline import llm
    from pipeline.enrich import enrich
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.delenv("CACHE_ONLY", raising=False)
    good = lambda m, ms: '{"intent": "lost_item", "confidence": 0.9, "sentiment": "neutral", "urgency": "low", "entities": {}}'
    other = lambda m, ms, **kw: '{"intent": "other", "confidence": 0.5, "sentiment": "neutral", "urgency": "low", "entities": {}}'
    enrich("cached on primary", [], call_fn=good, model=llm.primary_model())      # warm the primary cache
    monkeypatch.setattr(llm, "groq_chat", other)                                  # fallback path would answer "other"
    assert silver_label("cached on primary") == ("lost_item", "primary_cached")
    assert silver_label("never seen") == ("other", "fallback")
