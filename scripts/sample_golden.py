"""
Draw the 200-row unlabelled golden set from data/uber_threads.jsonl and remove those tweets from the retrieval corpus.
Label columns are left empty on purpose: a human fills them (CLAUDE.md rule 2). Keyword buckets only steer coverage.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.buckets import bucket_of
from scripts.validate_data import is_english

DATA = Path("data")
SEED = 42
N_ROWS = 200
MIN_PER_BUCKET = 15
MIN_CHARS = 20
LABEL_COLUMNS = ["intent", "sentiment", "urgency", "escalate", "reason", "ideal_reply_notes", "labeller"]
COLUMNS = ["id", "customer_tweet_id", "text", "history", *LABEL_COLUMNS]

def customer_tweet_ids() -> dict[tuple[str, str], str]:
    """(created_at, clean_text) -> tweet_id for customer tweets; thread turns carry no ids, this recovers them."""
    csv.field_size_limit(10**9)
    seen: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    with (DATA / "uber_tweets.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["role"] == "customer":
                key = (r["created_at"], r["clean_text"])
                ambiguous.add(key) if key in seen else seen.setdefault(key, r["tweet_id"])
    return {k: v for k, v in seen.items() if k not in ambiguous}


def candidates() -> list[dict]:
    """One candidate per customer turn: the root (no history) and the second customer turn (root as history)."""
    ids = customer_tweet_ids()
    out: list[dict] = []
    for line in (DATA / "uber_threads.jsonl").open():
        thread = json.loads(line)
        turns = [t for t in thread["turns"] if t["role"] == "customer"][:2]
        for i, turn in enumerate(turns):
            text = turn["text"]
            tweet_id = ids.get((turn["ts"], text))
            if tweet_id is None or len(text) < MIN_CHARS or not is_english(text):
                continue
            history = turns[0]["text"] if i == 1 else ""
            out.append({"thread": thread["root_tweet_id"], "customer_tweet_id": tweet_id, "text": text,
                        "history": history, "bucket": bucket_of(text)})
    return out


def stratified_sample(pool: list[dict], rng: random.Random) -> list[dict]:
    """Guarantee MIN_PER_BUCKET from each bucket where available, then fill to N_ROWS at random. One row per thread."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by_bucket[c["bucket"]].append(c)
    chosen: list[dict] = []
    used_threads: set[str] = set()

    def take(items: list[dict], n: int) -> None:
        rng.shuffle(items)
        for c in items:
            if len(chosen) >= N_ROWS or n <= 0:
                return
            if c["thread"] not in used_threads:
                chosen.append(c)
                used_threads.add(c["thread"])
                n -= 1

    for name in sorted(by_bucket):
        take(by_bucket[name], MIN_PER_BUCKET)
    take(pool, N_ROWS - len(chosen))
    rng.shuffle(chosen)
    return chosen


def labelled_golden() -> list[dict] | None:
    """The existing golden_set.csv if a human has started filling labels, else None. Never overwrite labels (rule 2)."""
    path = DATA / "golden_set.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(newline="")))
    return rows if any(r[c] for r in rows for c in LABEL_COLUMNS) else None


def remove_from_corpus(drop_ids: set[str], drop_texts: set[str]) -> int:
    """Drop corpus pairs that are a golden row (by id) or a golden row's history turn (by text)."""
    removed = 0
    for name in ("pairs_informative.jsonl", "pairs_deflection.jsonl"):
        path = DATA / name
        rows = [json.loads(line) for line in path.open()]
        kept = [r for r in rows if r["customer_tweet_id"] not in drop_ids and r["customer_text"] not in drop_texts]
        removed += len(rows) - len(kept)
        with path.open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return removed


def write_golden(rows: list[dict]) -> None:
    with (DATA / "golden_set.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"id": f"g{i:03d}", "customer_tweet_id": r["customer_tweet_id"],
                        "text": r["text"], "history": r["history"]})


def suggest() -> None:
    """Write the enricher's predicted intent into `suggested_intent` (never `intent`). Labels stay untouched."""
    from pipeline import llm
    from pipeline.enrich import enrich
    from pipeline.models import Turn

    llm.batch_mode()
    path = DATA / "golden_set.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if "suggested_intent" not in fields:
        fields.append("suggested_intent")
    agree = labelled = 0
    for i, r in enumerate(rows, 1):
        history = [Turn(role="user", content=r["history"])] if r["history"] else []
        r["suggested_intent"] = enrich(r["text"], history).intent.value
        if r["intent"].strip():
            labelled += 1
            agree += r["intent"].strip() == r["suggested_intent"]
        print(f"\r{i}/{len(rows)}", end="", flush=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsuggested_intent written for {len(rows)} rows")
    if labelled:
        print(f"labelled rows: {labelled}, agreement with suggestion: {agree}/{labelled} "
              f"(override rate {1 - agree / labelled:.0%})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suggest", action="store_true", help="fill suggested_intent from the enricher; no resampling")
    if ap.parse_args().suggest:
        suggest()
        return
    rows = labelled_golden()
    if rows is not None:
        print("golden_set.csv already has labels: keeping it, only re-applying corpus removal (CLAUDE.md rule 2)")
    else:
        pool = candidates()
        rows = stratified_sample(pool, random.Random(SEED))
        write_golden(rows)
        print(f"candidates={len(pool)} sampled={len(rows)}")

    removed = remove_from_corpus({r["customer_tweet_id"] for r in rows}, {r["history"] for r in rows if r["history"]})

    stats = json.loads((DATA / "stats.json").read_text())
    stats["golden_set"] = {"rows": len(rows), "with_history": sum(1 for r in rows if r["history"]),
                           "seed": SEED, "removed_from_corpus": removed}
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print(f"with_history={stats['golden_set']['with_history']} removed_from_corpus={removed}")
    print("bucket counts (coverage only, not labels):")
    for name, n in sorted(Counter(bucket_of(r["text"]) for r in rows).items()):
        print(f"  {name:12s} {n}")


if __name__ == "__main__":
    main()
