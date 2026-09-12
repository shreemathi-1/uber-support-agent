"""Tests for pipeline/enrich.py with a fake call_fn: happy path, one retry on bad JSON, fallback to OTHER. No network."""
import json
from pathlib import Path

import pytest

from pipeline.enrich import build_system_prompt, build_user_prompt, enrich
from pipeline.models import Turn
from pipeline.taxonomy import INTENT_DEFINITIONS, Intent

GOOD = json.dumps({
    "intent": "lost_item", "confidence": 0.9, "sentiment": "negative", "urgency": "medium",
    "entities": {"trip_date": "20 minutes ago", "is_repeat_contact": True},
})


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("CACHE_ONLY", raising=False)


def make_fake(responses: list[str]) -> tuple:
    """call_fn that returns the given responses in order and records every call."""
    calls: list[list[dict]] = []

    def fake(model: str, messages: list[dict]) -> str:
        calls.append(messages)
        return responses[len(calls) - 1]

    return fake, calls


def test_system_prompt_lists_every_intent_and_example() -> None:
    prompt = build_system_prompt()
    for intent, spec in INTENT_DEFINITIONS.items():
        assert f"- {intent.value}:" in prompt
        assert min(spec["examples"], key=len) in prompt  # one example per intent, the shortest (decision #41)
    assert "JSON only" in prompt


def test_user_prompt_includes_history_then_message() -> None:
    text = build_user_prompt("still nothing", [Turn(role="user", content="left my phone")])
    assert text.startswith("Earlier (user): left my phone")
    assert text.endswith("Message: still nothing")


def test_enrich_parses_valid_json() -> None:
    fake, calls = make_fake([GOOD])
    enr = enrich("left my phone in the car, already DM'd", call_fn=fake)
    assert enr.intent is Intent.LOST_ITEM
    assert enr.entities.is_repeat_contact is True
    assert enr.entities.amount is None
    assert len(calls) == 1


def test_enrich_retries_once_with_error_appended() -> None:
    fake, calls = make_fake(['{"intent": "refund"}', GOOD])
    enr = enrich("charged twice", call_fn=fake)
    assert enr.intent is Intent.LOST_ITEM
    assert len(calls) == 2
    retry_user_turn = calls[1][-1]["content"]
    assert "previous answer was invalid" in retry_user_turn
    assert "Message: charged twice" in retry_user_turn


def test_enrich_falls_back_to_other_after_two_failures() -> None:
    fake, calls = make_fake(["not json at all", '{"intent": "fare_dispute", "confidence": 7}'])
    enr = enrich("???", call_fn=fake)
    assert enr.intent is Intent.OTHER
    assert enr.confidence == 0.0
    assert enr.urgency == "low"
    assert len(calls) == 2


def test_enrich_second_call_is_a_cache_hit() -> None:
    fake, calls = make_fake([GOOD])
    enrich("same message", call_fn=fake)
    enrich("same message", call_fn=fake)
    assert len(calls) == 1


def test_daily_quota_429_is_not_transient_but_minute_429_is() -> None:
    import httpx
    from groq import RateLimitError
    from pipeline.llm import _is_daily_quota, _is_transient
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    daily = RateLimitError("Rate limit reached ... on tokens per day (TPD): Limit 200000", response=resp, body=None)
    minute = RateLimitError("Rate limit reached ... on requests per minute (RPM)", response=resp, body=None)
    assert _is_daily_quota(daily) and not _is_transient(daily)
    assert not _is_daily_quota(minute) and _is_transient(minute)


def test_enrich_model_param_changes_cache_key() -> None:
    fake, calls = make_fake([GOOD, GOOD])
    enrich("same message", call_fn=fake, model="model-a")
    enrich("same message", call_fn=fake, model="model-b")
    assert len(calls) == 2
