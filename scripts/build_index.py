"""
Build data/chroma from the split pairs and the help-centre articles. Deletes and rebuilds both collections.
Pairs are capped at MAX_PAIRS: every informative pair first, then a seeded sample of deflection pairs.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.buckets import bucket_intent  # noqa: E402
from pipeline.retrieve import chroma_path, embed, get_client, get_collection  # noqa: E402

DATA = Path("data")
SEED = 42
MAX_PAIRS = 20_000
BATCH = 256
CHUNK_WORDS = 220  # ~300 tokens


def load_pairs() -> list[dict]:
    informative = [json.loads(l) for l in (DATA / "pairs_informative.jsonl").open()]
    deflection = [json.loads(l) for l in (DATA / "pairs_deflection.jsonl").open()]
    for p in informative:
        p["source"] = "informative"
    for p in deflection:
        p["source"] = "deflection"
    room = max(0, MAX_PAIRS - len(informative))
    sampled = random.Random(SEED).sample(deflection, min(room, len(deflection)))
    print(f"pairs: {len(informative)} informative + {len(sampled)} of {len(deflection)} deflection (cap {MAX_PAIRS})")
    return informative + sampled


def parse_article(path: Path) -> tuple[dict, str]:
    """Front matter between the first two '---' lines as key: value; the rest is the body."""
    text = path.read_text()
    _, front, body = text.split("---", 2)
    meta = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in front.strip().splitlines())}
    return meta, body.strip()


def chunk(body: str) -> list[str]:
    """Pack paragraphs into chunks of about CHUNK_WORDS words; a single long paragraph is split by words."""
    chunks: list[str] = []
    current: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        words = para.split()
        if len(current) + len(words) > CHUNK_WORDS and current:
            chunks.append(" ".join(current))
            current = []
        current.extend(words)
        while len(current) > CHUNK_WORDS:
            chunks.append(" ".join(current[:CHUNK_WORDS]))
            current = current[CHUNK_WORDS:]
    if current:
        chunks.append(" ".join(current))
    return chunks


def rebuild(name: str):
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:  # chromadb raises ValueError (0.6) or NotFoundError (1.x) when absent
        pass
    return get_collection(name)


def add_in_batches(coll, ids: list[str], docs: list[str], metas: list[dict]) -> None:
    for start in range(0, len(docs), BATCH):
        sl = slice(start, start + BATCH)
        coll.add(ids=ids[sl], documents=docs[sl], metadatas=metas[sl], embeddings=embed(docs[sl]))
        print(f"\r  {coll.name}: {min(start + BATCH, len(docs))}/{len(docs)}", end="", file=sys.stderr)
    print(file=sys.stderr)


def main() -> None:
    pairs = load_pairs()
    coll = rebuild("pairs")
    add_in_batches(
        coll,
        ids=[p["customer_tweet_id"] for p in pairs],
        docs=[p["customer_text"] for p in pairs],
        metas=[{"brand_reply": p["brand_reply"], "source": p["source"], "tweet_id": p["customer_tweet_id"],
                "intent": bucket_intent(p["customer_text"])} for p in pairs],
    )

    ids, docs, metas = [], [], []
    for path in sorted((DATA / "help_center").glob("*.md")):
        meta, body = parse_article(path)
        for i, text in enumerate(chunk(body)):
            ids.append(f"{path.stem}#{i}")
            docs.append(text)
            metas.append({"title": meta["title"], "source_url": meta["source_url"], "intent": meta["intent"],
                          "date_checked": meta["date_checked"], "chunk_index": i})
    help_coll = rebuild("help")
    add_in_batches(help_coll, ids, docs, metas)

    size_mb = sum(f.stat().st_size for f in Path(chroma_path()).rglob("*") if f.is_file()) / 1e6
    print(f"done: pairs={coll.count()} help={help_coll.count()} chunks from {len(list((DATA / 'help_center').glob('*.md')))} articles, "
          f"index size {size_mb:.1f} MB at {chroma_path()}")


if __name__ == "__main__":
    main()
