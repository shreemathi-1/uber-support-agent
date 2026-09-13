"""
Simple baseline: TF-IDF + logistic regression on LLM silver labels for intent, keyword rules for escalation,
and the nearest historical Uber reply verbatim. What a week-one implementation without an LLM in the loop looks like.
"""
import json
import pickle
import random
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from pipeline import llm
from pipeline.cache import CacheMissError
from pipeline.enrich import enrich
from pipeline.llm import DailyQuotaExceeded
from pipeline.retrieve import _query, embed
from pipeline.rules import amount_limit

DATA = Path("data")
MODEL_PATH = Path("eval/results/simple_intent.pkl")
SILVER_INFO_PATH = Path("eval/results/simple_silver.json")
SEED = 42
KEYWORDS = ["police", "lawyer", "sue", "theft", "fraud", "unsafe", "assault", "harass", "stranded", "urgent",
            "emergency", "no response", "already contacted", "already dm"]
AMOUNT_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)")


def keyword_escalate(text: str) -> str | None:
    """First matching keyword, or 'amount_over' for a dollar figure above AMOUNT_LIMIT, else None."""
    lowered = text.lower()
    for kw in KEYWORDS:
        if kw in lowered:
            return kw
    if any(float(m.group(1)) > amount_limit() for m in AMOUNT_RE.finditer(text)):
        return "amount_over"
    return None


def silver_pairs(n: int) -> list[dict]:
    pairs = [json.loads(l) for name in ("pairs_informative.jsonl", "pairs_deflection.jsonl") for l in (DATA / name).open()]
    return random.Random(SEED).sample(pairs, min(n, len(pairs)))


def _cache_only(model: str, messages: list) -> str:
    raise CacheMissError("not cached")


def silver_label(text: str) -> tuple[str, str]:
    """(intent, which) - a label already cached from PRIMARY_MODEL is reused; otherwise label on FALLBACK_MODEL (decision #40)."""
    try:
        return enrich(text, [], call_fn=_cache_only, model=llm.primary_model()).intent.value, "primary_cached"
    except CacheMissError:
        return enrich(text, [], model=llm.fallback_model()).intent.value, "fallback"


def fit(n_silver: int, model_path: Path = MODEL_PATH) -> Pipeline:
    """Label n_silver pairs with the enricher (cached), fit TF-IDF + LR, pickle it. Silver labels, not truth."""
    pairs = silver_pairs(n_silver)
    texts, labels = [], []
    split = {"primary_cached": 0, "fallback": 0}
    for i, p in enumerate(pairs, 1):
        try:
            label, which = silver_label(p["customer_text"])
        except CacheMissError as err:
            sys.exit(f"CACHE_ONLY=1: silver label for pair {p['customer_tweet_id']} is not cached: {err}")
        except DailyQuotaExceeded as err:
            sys.exit(f"stopped at silver pair {i}/{len(pairs)}: {err}")
        labels.append(label)
        split[which] += 1
        texts.append(p["customer_text"])
        if i % 50 == 0 or i == len(pairs):
            print(f"\r  silver labels: {i}/{len(pairs)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    # TF-IDF + LogisticRegression text pipeline per the scikit-learn text classification tutorial (docs/BORROWED.md)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000))
    model.fit(texts, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(pickle.dumps(model))
    SILVER_INFO_PATH.write_text(json.dumps({"n": len(pairs), **split, "primary_model": llm.primary_model(),
                                            "fallback_model": llm.fallback_model()}, indent=2) + "\n")
    return model


def load_or_fit(n_silver: int, model_path: Path = MODEL_PATH) -> Pipeline:
    if model_path.exists():
        return pickle.loads(model_path.read_bytes())
    return fit(n_silver, model_path)


def nearest_reply(text: str) -> str:
    rows = _query("pairs", embed([text])[0], 1, None)
    return rows[0][2]["brand_reply"] if rows else ""


class SimpleBaseline:
    def __init__(self, model: Pipeline) -> None:
        self.model = model

    def predict(self, row: dict) -> dict:
        reason = keyword_escalate(row["text"])
        return {"intent": str(self.model.predict([row["text"]])[0]), "escalate": reason is not None,
                "reason": reason or "auto", "reply": nearest_reply(row["text"])}


def predict_one(message: str) -> dict:
    """The simple baseline's answer for one message, for the trace. Loads the pickle each call (small); never fits."""
    if not MODEL_PATH.exists():
        return {"available": False, "why": f"{MODEL_PATH} missing; run make eval to fit it"}
    baseline = SimpleBaseline(pickle.loads(MODEL_PATH.read_bytes()))
    return {"available": True, **baseline.predict({"text": message})}
