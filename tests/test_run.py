"""pipeline.run end to end with a fake Groq and a fixed retrieval: escalation before drafting, auto path, unsafe draft."""
import json
import re
from pathlib import Path

import pytest

from baselines import simple, trivial
from pipeline import db, llm, rules, run as run_mod
from pipeline.models import ChatRequest, Context, RetrievedChunk, RetrievedPair

TRACE_KEYS = ["request", "enrich", "rules", "retrieve", "draft", "check", "action", "totals"]


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
    # baselines answer from files that do not exist here: no 12k-line read, no pickle, no Chroma
    monkeypatch.setattr(trivial, "DEFLECTION_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(simple, "MODEL_PATH", tmp_path / "missing.pkl")
    trivial._default.cache_clear()


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


def test_trace_has_all_eight_steps_in_order_and_one_row_per_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGroq(enrich_json())
    monkeypatch.setattr(llm, "groq_chat", fake)
    resp = run_mod.run(ChatRequest(message="lost my wallet"))
    trace = db.get_trace(1)["trace"]
    assert list(trace)[:8] == TRACE_KEYS
    assert [r["rule"] for r in trace["rules"]["rows"]] == [name for name, _ in rules.RULES]
    assert all({"rule", "fired", "value", "threshold", "margin", "decisive"} <= set(r) for r in trace["rules"]["rows"])
    assert trace["rules"]["reason"] is None and not any(r["fired"] for r in trace["rules"]["rows"])
    assert trace["enrich"]["parsed"]["intent"] == "lost_item" and trace["enrich"]["llm_calls"] == 1
    assert trace["retrieve"]["ran"] and [p["id"] for p in trace["retrieve"]["pairs"]] == ["p1"]
    assert trace["retrieve"]["chunks"][0]["usable"] is True
    assert trace["draft"]["raw_output"] == fake.responses["in-app support chat"]
    assert trace["draft"]["final_reply"] == resp.reply and trace["draft"]["post_process_notes"] == []
    assert trace["check"] == {**trace["check"], "safe": True, "why": "fine"}
    assert trace["action"] == {"action": "auto", "reason": "auto", "reply": resp.reply, "ticket_id": 1, "escalation_id": None}
    assert trace["totals"]["llm_calls"] == 3 and trace["totals"]["cache_hits"] == 0
    assert trace["totals"]["prompt_tokens"] > 0 and trace["totals"]["estimated_paid_cost_usd"] > 0
    assert trace["baselines"]["trivial"]["available"] is False and trace["baselines"]["simple"]["available"] is False


def test_trace_of_a_rule_escalation_marks_decisive_rule_and_skipped_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "groq_chat", FakeGroq(enrich_json(entities={"mentions_safety": True}, urgency="high")))
    run_mod.run(ChatRequest(message="driver was drunk"))
    trace = db.get_trace(1)["trace"]
    rows = {r["rule"]: r for r in trace["rules"]["rows"]}
    assert rows["safety"] == {**rows["safety"], "fired": True, "decisive": True, "value": True, "margin": 0.0}
    assert rows["urgent"]["fired"] is True and rows["urgent"]["decisive"] is False  # evaluated even after safety fired
    assert not trace["retrieve"]["ran"] and not trace["draft"]["ran"] and trace["check"]["safe"] is None
    assert trace["draft"]["final_reply"] is None
    assert trace["action"]["escalation_id"] == 1 and "ticket 1" in trace["action"]["reply"]
    assert trace["totals"]["llm_calls"] == 1


def test_replaying_the_same_message_counts_cache_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "groq_chat", FakeGroq(enrich_json()))
    run_mod.run(ChatRequest(message="lost my wallet"))
    monkeypatch.setenv("CACHE_ONLY", "1")  # second pass must not need the fake at all
    run_mod.run(ChatRequest(message="lost my wallet"))
    assert db.get_trace(2)["trace"]["totals"] == {**db.get_trace(2)["trace"]["totals"], "llm_calls": 3, "cache_hits": 3}
    assert [t["id"] for t in db.list_traces(10)] == [2, 1] and db.list_traces(10)[0]["has_trace"] is True


def test_init_db_adds_trace_column_to_an_older_tickets_table() -> None:
    with db.get_conn() as conn:
        conn.executescript(re.sub(r",\s*trace\s+TEXT", "", db.SCHEMA))  # the schema as it was before the trace column
    assert "trace" not in {r["name"] for r in db.get_conn().execute("PRAGMA table_info(tickets)")}
    db.init_db()
    assert "trace" in {r["name"] for r in db.get_conn().execute("PRAGMA table_info(tickets)")}
    assert db.get_trace(999) is None
