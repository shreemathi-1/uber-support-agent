"""Tests for pipeline/cache.py: miss calls through and stores, hit skips the call, CACHE_ONLY=1 miss raises."""
from pathlib import Path

import pytest

from pipeline.cache import CacheMissError, cache_key, cached_llm

MSGS = [{"role": "user", "content": "left my phone in the car"}]


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("CACHE_ONLY", raising=False)


def test_cache_miss_calls_fn_and_stores() -> None:
    calls: list[tuple[str, list]] = []

    def fake(model: str, messages: list) -> str:
        calls.append((model, messages))
        return "reply-1"

    assert cached_llm("m", MSGS, fake) == "reply-1"
    assert calls == [("m", MSGS)]


def test_cache_hit_does_not_call_fn() -> None:
    cached_llm("m", MSGS, lambda m, ms: "reply-1")

    def boom(model: str, messages: list) -> str:
        raise AssertionError("network call on a cache hit")

    assert cached_llm("m", MSGS, boom) == "reply-1"


def test_key_depends_on_model_and_messages() -> None:
    assert cache_key("a", MSGS) != cache_key("b", MSGS)
    assert cache_key("a", MSGS) != cache_key("a", [{"role": "user", "content": "other"}])
    # Dict key order must not change the key.
    assert cache_key("a", [{"content": "x", "role": "user"}]) == cache_key("a", [{"role": "user", "content": "x"}])


def test_cache_only_miss_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_ONLY", "1")

    def boom(model: str, messages: list) -> str:
        raise AssertionError("network call under CACHE_ONLY=1")

    with pytest.raises(CacheMissError):
        cached_llm("m", MSGS, boom)


def test_cache_only_hit_is_served(monkeypatch: pytest.MonkeyPatch) -> None:
    cached_llm("m", MSGS, lambda m, ms: "reply-1")
    monkeypatch.setenv("CACHE_ONLY", "1")
    assert cached_llm("m", MSGS, lambda m, ms: "should-not-run") == "reply-1"


def test_empty_response_is_not_cached() -> None:
    answers = iter(["", "reply-2"])
    fn = lambda m, ms: next(answers)
    assert cached_llm("m", MSGS, fn) == ""
    assert cached_llm("m", MSGS, fn) == "reply-2"
