"""Filter and fallback behaviour of pipeline/retrieve.py on a tiny temp index with a fake keyword embedder. No model download."""
from pathlib import Path

import pytest

from pipeline import retrieve as r

TOPICS = ["phone", "fare", "food", "app"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """One-hot over four topic words plus a tiny constant, so same-topic texts have cosine distance ~0 and others ~1."""
    return [[1.0 if t in text.lower() else 0.0 for t in TOPICS] + [0.01] for text in texts]


PAIRS = [
    ("p1", "lost my phone in the car", "lost_item"),
    ("p2", "phone left on back seat", "lost_item"),
    ("p3", "driver has my phone", "lost_item"),
    ("p4", "fare was double what I expected", "fare_dispute"),
    ("p5", "why is my fare so high", "fare_dispute"),
    ("p6", "food order never arrived", "uber_eats"),
]
HELP = [
    ("h1", "Use Your Trips to call the driver about the phone you left", "lost_item", "Recovering a lost item"),
    ("h2", "Request a fare review from the trip receipt", "fare_dispute", "Reviewing a fare"),
]


@pytest.fixture(autouse=True)
def tiny_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    r.get_client.cache_clear()
    monkeypatch.setattr(r, "embed", fake_embed)
    pairs = r.get_collection("pairs")
    pairs.add(ids=[i for i, _, _ in PAIRS], documents=[d for _, d, _ in PAIRS],
              metadatas=[{"brand_reply": f"reply to {i}", "source": "deflection", "tweet_id": i, "intent": it} for i, _, it in PAIRS],
              embeddings=fake_embed([d for _, d, _ in PAIRS]))
    help_ = r.get_collection("help")
    help_.add(ids=[i for i, _, _, _ in HELP], documents=[d for _, d, _, _ in HELP],
              metadatas=[{"title": t, "source_url": "https://example.test", "intent": it, "date_checked": "x", "chunk_index": 0} for i, d, it, t in HELP],
              embeddings=fake_embed([d for _, d, _, _ in HELP]))
    yield
    r.get_client.cache_clear()


def test_filtered_query_returns_only_that_intent() -> None:
    ctx = r.retrieve("where is my phone", "lost_item")
    assert [e.intent for e in ctx.examples] == ["lost_item"] * 3
    assert all(e.score < 0.1 for e in ctx.examples)
    assert ctx.examples[0].reply_text.startswith("reply to")
    assert [f.title for f in ctx.facts] == ["Recovering a lost item"]


def test_too_few_filtered_hits_falls_back_to_unfiltered() -> None:
    ctx = r.retrieve("my fare is wrong", "fare_dispute")  # only 2 fare pairs indexed, MIN_HITS is 3
    assert len(ctx.examples) == 5
    assert ctx.examples[0].intent == "fare_dispute"
    assert {e.intent for e in ctx.examples} > {"fare_dispute"}


def test_distant_filtered_hits_fall_back_to_unfiltered() -> None:
    ctx = r.retrieve("food never came", "lost_item")  # 3 lost_item pairs exist but are all far away
    assert ctx.examples[0].id == "p6"
    assert ctx.examples[0].score < 0.1
    assert ctx.facts, "help fallback should still return the nearest chunk"


def test_intent_with_no_indexed_docs_falls_back() -> None:
    ctx = r.retrieve("phone", "policy_or_info_question")
    assert len(ctx.examples) == 5
    assert ctx.examples[0].intent == "lost_item"


def test_empty_index_returns_empty_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "empty"))
    r.get_client.cache_clear()
    ctx = r.retrieve("anything", "other")
    assert ctx.examples == [] and ctx.facts == []
