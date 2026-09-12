"""
Retrieval step: embed the message with MiniLM and pull the nearest historical pairs and help chunks from Chroma.
Filters by intent first; falls back to an unfiltered query when the filter yields too few or too distant hits.
"""
import os
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

from pipeline.models import Context, RetrievedChunk, RetrievedPair

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PAIRS_N, HELP_N = 5, 2
MIN_HITS = {"pairs": 3, "help": 1}   # fewer filtered hits than this -> unfiltered requery
MAX_BEST_DISTANCE = 0.6              # best filtered hit farther than this (cosine distance) -> unfiltered requery


def chroma_path() -> str:
    return os.environ.get("CHROMA_PATH", "data/chroma")


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    Path(chroma_path()).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_path(), settings=Settings(anonymized_telemetry=False))


def get_collection(name: str) -> chromadb.Collection:
    """Cosine space so distances are 0 (identical) to 2 (opposite); embeddings are always supplied by us."""
    return get_client().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer  # heavy import, only when embedding is actually needed
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False).tolist()


def _query(name: str, vector: list[float], n: int, intent: str | None) -> list[tuple[str, str, dict, float]]:
    """Returns (id, document, metadata, distance) rows, nearest first. Empty list if the filter matches nothing."""
    coll = get_collection(name)
    if coll.count() == 0:
        return []
    kwargs = {"where": {"intent": intent}} if intent else {}
    res = coll.query(query_embeddings=[vector], n_results=n, include=["documents", "metadatas", "distances"], **kwargs)
    return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))


def _query_with_fallback(name: str, vector: list[float], n: int, intent: str) -> list[tuple[str, str, dict, float]]:
    rows = _query(name, vector, n, intent)
    if len(rows) < MIN_HITS[name] or rows[0][3] > MAX_BEST_DISTANCE:
        rows = _query(name, vector, n, None)
    return rows


def retrieve(message: str, intent: str) -> Context:
    vector = embed([message])[0]
    examples = [
        RetrievedPair(id=i, customer_text=doc, reply_text=m["brand_reply"], intent=m["intent"], source=m["source"], score=d)
        for i, doc, m, d in _query_with_fallback("pairs", vector, PAIRS_N, intent)
    ]
    facts = [
        RetrievedChunk(id=i, title=m["title"], source_url=m["source_url"], intent=m["intent"], text=doc, score=d)
        for i, doc, m, d in _query_with_fallback("help", vector, HELP_N, intent)
    ]
    return Context(examples=examples, facts=facts)
