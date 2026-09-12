"""pipeline.run end to end with a fake Groq and a fixed retrieval: escalation before drafting, auto path, unsafe draft."""
import json
from pathlib import Path

import pytest

from pipeline import db, llm, run as run_mod
from pipeline.models import ChatRequest, Context, RetrievedChunk, RetrievedPair


def enrich_json(**overrides) -> str:
    base = {"intent": "lost_item", "confidence": 0.9, "sentiment": "negative", "urgency": "medium", "entities": {}}
    base.update(overrides)
    return json.dumps(base)


class FakeGroq:
    """Dispatches on the system prompt so one fake serves enrich, draft and check. Records which stages ran."""

    def __init__(self, enrich: str, draft: str = "Sorry about that. What was the trip date? We'll take it from there.",
                 check: str = '{"safe": true, "why": "fine"}') -> None:
        self.responses = {"classify": enrich, "in-app support chat": draft, "safety reviewer": check}
        self.stages: list[str] = []

    def __call__(self, model: str, messages: list[dict], **kwargs) -> str:
        system = messages[0]["content"]
        stage = next(k for k in self.responses if k in system)
        self.stages.append(stage)
        return self.responses[stage]


CTX = Context(
    examples=[RetrievedPair(id="p1", customer_text="c", reply_text="r", intent="lost_item", source="informative", score=0.2)],
    facts=[RetrievedChunk(id="h1", title="t", source_url="u", intent="lost_item", text="f", score=0.4)],
)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("CACHE_ONLY", raising=False)
    monkeypatch.setattr(run_mod, "retrieve", lambda message, intent: CTX)


def test_rule_escalates_before_any_drafting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGroq(enrich_json(entities={"mentions_safety": True}))
    monkeypatch.setattr(llm, "groq_chat", fake)
    resp = run_mod.run(ChatRequest(message="driver was drunk"))
    assert resp.action == "escalate" and resp.reason == "safety"
    assert fake.stages == ["classify"]
    assert resp.retrieved_ids == []
    assert "ticket 1" in resp.reply
    esc = db.list_escalations("open")
    assert len(esc) == 1 and esc[0]["draft"] is None and esc[0]["reason"] == "safety"
    assert db.count("tickets") == 1


def test_auto_path_runs_all_three_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGroq(enrich_json())
    monkeypatch.setattr(llm, "groq_chat", fake)
    resp = run_mod.run(ChatRequest(message="lost my wallet", conversation_id="conv-1"))
    assert resp.action == "auto" and resp.reason == "auto"
    assert fake.stages == ["classify", "in-app support chat", "safety reviewer"]
    assert resp.reply.startswith("Sorry about that.")
    assert resp.retrieved_ids == ["pair:p1:0.20", "help:h1:0.40"]
    assert resp.conversation_id == "conv-1"
    assert db.count("escalations") == 0 and db.count("tickets") == 1


def test_unsafe_draft_escalates_and_keeps_the_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGroq(enrich_json(), check='{"safe": false, "why": "promises refund"}')
    monkeypatch.setattr(llm, "groq_chat", fake)
    resp = run_mod.run(ChatRequest(message="lost my wallet"))
    assert resp.action == "escalate" and resp.reason == "unsafe_draft"
    assert "specialist" in resp.reply
    esc = db.list_escalations("open")[0]
    assert esc["draft"].startswith("Sorry about that.") and esc["retrieved_ids"] == ["pair:p1:0.20", "help:h1:0.40"]


def test_check_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGroq(enrich_json(), check="this is not json")
    monkeypatch.setattr(llm, "groq_chat", fake)
    resp = run_mod.run(ChatRequest(message="lost my wallet"))
    assert resp.reason == "unsafe_draft"


def test_resolve_escalation_changes_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "groq_chat", FakeGroq(enrich_json(urgency="high")))
    run_mod.run(ChatRequest(message="stranded"))
    esc_id = db.list_escalations("open")[0]["id"]
    assert db.resolve_escalation(esc_id) is True
    assert db.resolve_escalation(esc_id) is False
    assert db.list_escalations("open") == [] and len(db.list_escalations("resolved")) == 1


def test_empty_draft_is_unsafe_without_calling_the_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGroq(enrich_json(), draft="")
    monkeypatch.setattr(llm, "groq_chat", fake)
    resp = run_mod.run(ChatRequest(message="lost my wallet"))
    assert resp.reason == "unsafe_draft"
    assert fake.stages == ["classify", "in-app support chat"]


def test_other_intent_gets_canned_reply_without_drafting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGroq(enrich_json(intent="other", sentiment="positive", urgency="low"))
    monkeypatch.setattr(llm, "groq_chat", fake)
    resp = run_mod.run(ChatRequest(message="thanks, that sorted it"))
    assert resp.action == "auto" and resp.reason == "auto"
    assert resp.reply == run_mod.OTHER_REPLY
    assert fake.stages == ["classify"]
    assert resp.retrieved_ids == [] and db.count("escalations") == 0
