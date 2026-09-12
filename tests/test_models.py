"""Contract tests for pipeline/models.py: defaults, validation bounds, and the history cap."""
import pytest
from pydantic import ValidationError

from pipeline.models import ChatRequest, ChatResponse, Enrichment, Entities, Turn
from pipeline.taxonomy import Intent

ENRICH_JSON = {
    "intent": "fare_dispute",
    "confidence": 0.92,
    "sentiment": "negative",
    "urgency": "medium",
    "entities": {"amount": 13.0, "city": "Lexington KY"},
}


def test_entities_defaults_are_empty_and_false() -> None:
    e = Entities()
    assert e.email is None and e.amount is None
    assert e.mentions_safety is False and e.is_repeat_contact is False


def test_enrichment_parses_llm_json_and_coerces_intent_enum() -> None:
    enr = Enrichment.model_validate(ENRICH_JSON)
    assert enr.intent is Intent.FARE_DISPUTE
    assert enr.entities.amount == 13.0
    assert enr.entities.mentions_legal is False


@pytest.mark.parametrize(
    "bad",
    [
        {**ENRICH_JSON, "intent": "refund"},           # not in taxonomy
        {**ENRICH_JSON, "confidence": 1.5},            # out of range
        {**ENRICH_JSON, "sentiment": "angry"},         # not in literal
        {**ENRICH_JSON, "urgency": "critical"},        # not in literal
    ],
)
def test_enrichment_rejects_invalid_values(bad: dict) -> None:
    with pytest.raises(ValidationError):
        Enrichment.model_validate(bad)


def test_chat_request_allows_up_to_two_history_turns() -> None:
    turns = [Turn(role="user", content="hi"), Turn(role="agent", content="hello")]
    req = ChatRequest(message="still broken", history=turns)
    assert len(req.history) == 2
    assert req.conversation_id is None


def test_chat_request_rejects_three_history_turns() -> None:
    turns = [Turn(role="user", content=str(i)) for i in range(3)]
    with pytest.raises(ValidationError):
        ChatRequest(message="x", history=turns)


def test_chat_response_round_trips_through_json() -> None:
    resp = ChatResponse(
        conversation_id="c1",
        intent=Intent.LOST_ITEM,
        confidence=0.8,
        sentiment="neutral",
        urgency="low",
        entities=Entities(),
        reply="Sorry to hear that.",
        action="auto",
        reason="auto",
        retrieved_ids=["p1", "h1"],
        latency_ms=120,
    )
    again = ChatResponse.model_validate_json(resp.model_dump_json())
    assert again == resp
    assert again.intent.value == "lost_item"


def test_chat_response_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(
            conversation_id="c1", intent=Intent.OTHER, confidence=0.1, sentiment="neutral",
            urgency="low", entities=Entities(), reply="", action="defer", reason="auto",
            retrieved_ids=[], latency_ms=0,
        )
