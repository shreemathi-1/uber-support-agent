"""Rule order, first-match-wins, and the env-driven amount limit for pipeline/rules.py."""
import pytest

from pipeline import rules
from pipeline.models import Enrichment, Entities
from pipeline.taxonomy import Intent


def enr(**overrides) -> Enrichment:
    base = dict(intent=Intent.LOST_ITEM, confidence=0.9, sentiment="negative", urgency="medium", entities=Entities())
    base.update(overrides)
    return Enrichment(**base)


def test_allowed_intent_with_no_flags_is_auto() -> None:
    assert rules.evaluate(enr()) is None


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"entities": Entities(mentions_safety=True)}, "safety"),
        ({"entities": Entities(mentions_legal=True)}, "legal"),
        ({"entities": Entities(amount=10.01)}, "amount_over"),
        ({"urgency": "high"}, "urgent"),
        ({"entities": Entities(is_repeat_contact=True)}, "repeat_contact"),
        ({"confidence": 0.69}, "low_confidence"),
        ({"intent": Intent.FARE_DISPUTE}, "not_allowed"),
    ],
)
def test_each_rule_fires_alone(overrides: dict, expected: str) -> None:
    assert rules.evaluate(enr(**overrides)) == expected


def test_first_match_wins_in_table_order() -> None:
    everything = enr(intent=Intent.FARE_DISPUTE, confidence=0.1, urgency="high",
                     entities=Entities(mentions_safety=True, mentions_legal=True, amount=500, is_repeat_contact=True))
    assert rules.evaluate(everything) == "safety"
    everything.entities.mentions_safety = False
    assert rules.evaluate(everything) == "legal"
    everything.entities.mentions_legal = False
    assert rules.evaluate(everything) == "amount_over"


def test_amount_at_limit_does_not_fire_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    assert rules.evaluate(enr(entities=Entities(amount=10.0))) is None
    monkeypatch.setenv("AMOUNT_LIMIT", "5")
    assert rules.evaluate(enr(entities=Entities(amount=6.0))) == "amount_over"


def test_rule_names_and_order_match_claude_md() -> None:
    assert [name for name, _ in rules.RULES] == [
        "safety", "legal", "amount_over", "urgent", "repeat_contact", "low_confidence", "not_allowed"
    ]


def test_other_is_auto_allowed_but_still_subject_to_earlier_rules() -> None:
    assert rules.evaluate(enr(intent=Intent.OTHER)) is None
    assert rules.evaluate(enr(intent=Intent.OTHER, urgency="high")) == "urgent"
