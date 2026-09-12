"""
Data sanity report: counts, template share, text lengths, an English proxy, a golden-set leakage check, and 30 pairs to eyeball.
Prints everything and writes data/validation.json; exits 1 if the golden set shares a customer_tweet_id with the retrieval corpus.
"""
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

DATA = Path("data")
SEED = 42
TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"
BUCKET_WIDTH = 30  # 10 buckets of 30 chars covers a 280-char tweet


def is_english(text: str) -> bool:
    """Proxy: > 70% of characters are ASCII letters, spaces or basic punctuation. No language model needed."""
    if not text:
        return False
    ok = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isspace() or c in ".,!?'\"-:;()"))
    return ok / len(text) > 0.7


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", "", text.lower())).strip()


def date_range(rows: list[dict]) -> list[str]:
    dates = sorted(datetime.strptime(r["created_at"], TS_FORMAT) for r in rows)
    return [dates[0].date().isoformat(), dates[-1].date().isoformat()]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def leakage_ids(golden_path: Path) -> set[str]:
    if not golden_path.exists():
        return set()
    with golden_path.open(newline="") as f:
        return {row["customer_tweet_id"] for row in csv.DictReader(f)}


def main() -> None:
    csv.field_size_limit(10**9)
    with (DATA / "uber_tweets.csv").open(newline="") as f:
        tweets = list(csv.DictReader(f))
    pairs = read_jsonl(DATA / "uber_pairs.jsonl")
    threads = read_jsonl(DATA / "uber_threads.jsonl")
    stats = json.loads((DATA / "stats.json").read_text())

    report: dict = {
        "rows": {"tweets": len(tweets), "pairs": len(pairs), "threads": len(threads)},
        # Two ranges: hop-2 extraction pulls in older customer tweets, so "all" starts earlier than Uber's own replies.
        "date_range_all_uber_related_tweets": date_range(tweets),
        "date_range_uber_support_authored_tweets": date_range([t for t in tweets if t["author_id"] == "Uber_Support"]),
        "top15_template_share": stats["top15_template_share"],
    }

    forms = Counter(normalise(p["brand_reply"]) for p in pairs)
    top30 = sum(n for _, n in forms.most_common(30))
    report["near_duplicate_share_top30_normalised"] = round(top30 / len(pairs), 4)

    lengths = [len(p["customer_text"]) for p in pairs]
    hist = Counter(min(n // BUCKET_WIDTH, 9) for n in lengths)
    report["customer_length_histogram"] = {
        f"{b * BUCKET_WIDTH}-{(b + 1) * BUCKET_WIDTH - 1 if b < 9 else '+'}": hist.get(b, 0) for b in range(10)
    }
    report["share_under_8_chars"] = round(sum(1 for n in lengths if n < 8) / len(pairs), 4)
    report["english_share"] = round(sum(1 for p in pairs if is_english(p["customer_text"])) / len(pairs), 4)

    corpus_ids = {p["customer_tweet_id"] for p in read_jsonl(DATA / "pairs_informative.jsonl")}
    corpus_ids |= {p["customer_tweet_id"] for p in read_jsonl(DATA / "pairs_deflection.jsonl")}
    leaked = leakage_ids(DATA / "golden_set.csv") & corpus_ids
    report["golden_leakage_ids"] = sorted(leaked)

    (DATA / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    print("\n30 random pairs for manual review:")
    for p in random.Random(SEED).sample(pairs, 30):
        print(f"  CUSTOMER: {p['customer_text'][:120]}\n  UBER:     {p['brand_reply'][:120]}\n")

    if leaked:
        print(f"LEAKAGE: {len(leaked)} golden customer_tweet_ids are still in the retrieval corpus", file=sys.stderr)
        sys.exit(1)
    print("leakage check: OK")


if __name__ == "__main__":
    main()
