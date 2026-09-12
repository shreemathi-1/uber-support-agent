"""Draft prompt assembly and post-processing (URL strip, 280 cut, repeat-contact prefix) with a fake call_fn."""
from pathlib import Path

import pytest

from pipeline.draft import MAX_CHARS, REPEAT_PREFIX, build_system_prompt, build_user_prompt, draft, postprocess
from pipeline.models import Context, Enrichment, Entities, RetrievedChunk, RetrievedPair
from pipeline.taxonomy import Intent


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("CACHE_ONLY", raising=False)


def pair(i: str, source: str, score: float) -> RetrievedPair:
    return RetrievedPair(id=i, customer_text=f"cust {i}", reply_text=f"uber {i}", intent="lost_item", source=source, score=score)


def chunk(i: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=i, title=f"title {i}", source_url="https://x", intent="lost_item", text=f"fact {i}", score=score)


ENR = Enrichment(intent=Intent.LOST_ITEM, confidence=0.9, sentiment="negative", urgency="medium", entities=Entities())


def test_postprocess_strips_urls_and_collapses_whitespace() -> None:
    out = postprocess("Sorry about that.  See https://help.uber.com/x or t.co/abc  for  more.", False)
    assert "http" not in out and "t.co" not in out
    assert "  " not in out


def test_postprocess_truncates_at_sentence_boundary() -> None:
    text = ("This is a sentence. " * 20).strip()
    out = postprocess(text, False)
    assert len(out) <= MAX_CHARS
    assert out.endswith(".")


def test_postprocess_repeat_prefix_only_when_not_already_acknowledged() -> None:
    assert postprocess("Sorry about the wallet.", True).startswith(REPEAT_PREFIX)
    already = "Sorry you've had to reach us again about the wallet."
    assert postprocess(already, True) == already
    assert not postprocess("Sorry about the wallet.", False).startswith(REPEAT_PREFIX)


def test_user_prompt_orders_informative_first_and_filters_far_facts() -> None:
    ctx = Context(examples=[pair("d1", "deflection", 0.1), pair("i1", "informative", 0.4), pair("i2", "informative", 0.3)],
                  facts=[chunk("near", 0.5), chunk("far", 0.9)])
    text = build_user_prompt("where is my wallet", [], ENR, ctx)
    assert text.index("cust i2") < text.index("cust i1") < text.index("cust d1")
    assert "fact near" in text and "fact far" not in text
    assert "Ask for: the trip date" in text


def test_user_prompt_says_skip_step_b_without_facts() -> None:
    text = build_user_prompt("hi", [], ENR, Context())
    assert "skip step (b)" in text


def test_system_prompt_has_channel_rules() -> None:
    p = build_system_prompt()
    assert "DM us" in p and "280" in p and "in-app chat" in p


def test_draft_calls_fn_and_postprocesses() -> None:
    fake = lambda model, messages: "Sorry about the wallet. Check https://x.y/z now.  What was the trip date?"
    out = draft("lost wallet", [], ENR, Context(), call_fn=fake)
    assert out == "Sorry about the wallet. Check now. What was the trip date?"
