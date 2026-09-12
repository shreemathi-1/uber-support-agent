"""
Memory-safe extraction of the Uber_Support subset from the Kaggle
"Customer Support on Twitter" dataset (twcs.csv, ~2.8M rows).

Never loads the full file. Streams it in chunks, three passes:
  pass 1  find Uber_Support rows + the tweet ids they link to
  pass 2  pull those linked rows (hop 1) + the ids *they* link to
  pass 3  pull hop-2 rows
Then builds pairs, threads and stats from the small Uber-only subset.

Usage:
    pip install pandas
    python extract_uber_chunked.py --csv twcs.csv --out data/ [--chunksize 100000]

Outputs (in --out):
    uber_tweets.csv     every tweet in a thread that involves Uber_Support
    uber_pairs.jsonl    (customer message -> Uber reply) pairs   <- RAG corpus
    uber_threads.jsonl  full reconstructed threads               <- for labelling
    stats.json          counts for the report
"""
import argparse, json, re, html, time
from pathlib import Path
import pandas as pd

BRAND = "Uber_Support"
COLS  = ["tweet_id", "author_id", "inbound", "created_at", "text",
         "response_tweet_id", "in_response_to_tweet_id"]

URL_RE     = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE   = re.compile(r"\s+")


def clean(text: str) -> str:
    """Light normalisation. Keeps emoji, casing, punctuation."""
    t = html.unescape(str(text))
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    t = t.replace("^", " ")
    return SPACE_RE.sub(" ", t).strip(" .-:")


def split_ids(v):
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def chunks(csv_path, chunksize):
    return pd.read_csv(csv_path, usecols=COLS, dtype=str,
                       keep_default_na=False, chunksize=chunksize)


def count_rows(csv_path, chunksize):
    n = 0
    for c in pd.read_csv(csv_path, usecols=["tweet_id"], dtype=str,
                         keep_default_na=False, chunksize=chunksize):
        n += len(c)
    return n


def progress(i, total_chunks, start, chunksize):
    lo, hi = i * chunksize + 1, (i + 1) * chunksize
    print(f"[{i + 1}/{total_chunks}] Processing rows {lo:,}–{hi:,}")


def stream_pass(csv_path, chunksize, total_chunks, label, want_ids=None,
                want_author=None):
    """Return list of row-dicts matching want_ids and/or want_author."""
    print(f"\n=== {label} ===")
    found, t0 = [], time.time()
    for i, chunk in enumerate(chunks(csv_path, chunksize)):
        progress(i, total_chunks, t0, chunksize)
        mask = pd.Series(False, index=chunk.index)
        if want_author is not None:
            mask |= chunk["author_id"] == want_author
        if want_ids is not None:
            mask |= chunk["tweet_id"].isin(want_ids)
        found.extend(chunk[mask].to_dict("records"))
    print(f"    -> {len(found):,} rows in {time.time() - t0:.1f}s")
    return found


def neighbour_ids(rows):
    ids = set()
    for r in rows:
        ids.update(split_ids(r["response_tweet_id"]))
        ids.update(split_ids(r["in_response_to_tweet_id"]))
    return ids


def main(csv_path, out_dir, chunksize):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    print("Counting rows ...")
    total_rows = count_rows(csv_path, chunksize)
    total_chunks = (total_rows + chunksize - 1) // chunksize
    print(f"Total rows: {total_rows:,}  ({total_chunks} chunks of {chunksize:,})")

    # ---- pass 1: Uber rows ----
    uber_rows = stream_pass(csv_path, chunksize, total_chunks,
                            "Pass 1: Uber_Support rows", want_author=BRAND)
    print(f"Uber_Support rows found:  {len(uber_rows):,}")
    have = {r["tweet_id"] for r in uber_rows}
    hop1 = neighbour_ids(uber_rows) - have

    # ---- pass 2: hop 1 ----
    hop1_rows = stream_pass(csv_path, chunksize, total_chunks,
                            "Pass 2: tweets linked to Uber (hop 1)", want_ids=hop1)
    have |= {r["tweet_id"] for r in hop1_rows}
    hop2 = neighbour_ids(hop1_rows) - have

    # ---- pass 3: hop 2 ----
    hop2_rows = stream_pass(csv_path, chunksize, total_chunks,
                            "Pass 3: follow-ups (hop 2)", want_ids=hop2)

    # ---- assemble small subset ----
    sub = pd.DataFrame(uber_rows + hop1_rows + hop2_rows).drop_duplicates("tweet_id")
    sub["inbound"] = sub["inbound"].str.lower() == "true"
    sub["clean_text"] = sub["text"].map(clean)
    sub["role"] = sub["inbound"].map({True: "customer", False: "brand"})
    sub = sub[sub["clean_text"].str.len() > 0].reset_index(drop=True)
    sub.to_csv(out / "uber_tweets.csv", index=False)
    print(f"\nUber-related tweets kept: {len(sub):,}")

    by_id = sub.set_index("tweet_id")

    # ---- pairs ----
    pairs = []
    for _, r in sub[sub["author_id"] == BRAND].iterrows():
        parent = split_ids(r["in_response_to_tweet_id"])
        if not parent or parent[0] not in by_id.index:
            continue
        p = by_id.loc[parent[0]]
        if not p["inbound"]:
            continue
        pairs.append({
            "customer_tweet_id": parent[0],
            "customer_text": p["clean_text"],
            "customer_raw": p["text"],
            "brand_tweet_id": r["tweet_id"],
            "brand_reply": r["clean_text"],
            "created_at": r["created_at"],
        })
    seen, uniq = set(), []
    for p in pairs:
        k = p["customer_text"].lower()
        if k in seen or len(p["customer_text"]) < 8:
            continue
        seen.add(k); uniq.append(p)
    with open(out / "uber_pairs.jsonl", "w", encoding="utf-8") as f:
        for p in uniq:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Customer->Uber pairs:     {len(pairs):,}  (unique: {len(uniq):,})")

    # ---- threads ----
    roots = sub[(sub["in_response_to_tweet_id"] == "") & sub["inbound"]]
    threads = []

    def walk(tid, acc, depth=0):
        if tid not in by_id.index or depth > 20:
            return
        r = by_id.loc[tid]
        acc.append({"role": r["role"], "text": r["clean_text"], "ts": r["created_at"]})
        for c in split_ids(r["response_tweet_id"]):
            walk(c, acc, depth + 1)

    for tid in roots["tweet_id"]:
        acc = []
        walk(tid, acc)
        if any(m["role"] == "brand" for m in acc):
            threads.append({"root_tweet_id": tid, "turns": acc})
    with open(out / "uber_threads.jsonl", "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Threads reconstructed:    {len(threads):,}")

    # ---- stats ----
    uber_only = sub[sub["author_id"] == BRAND]
    top = uber_only["clean_text"].value_counts().head(15)
    stats = {
        "total_rows_in_csv": int(total_rows),
        "uber_related_tweets": int(len(sub)),
        "uber_replies": int(len(uber_only)),
        "customer_tweets": int(sub["inbound"].sum()),
        "pairs": len(pairs), "unique_pairs": len(uniq), "threads": len(threads),
        "date_range": [sub["created_at"].min(), sub["created_at"].max()],
        "top15_template_share": round(float(top.sum() / max(len(uber_only), 1)), 3),
        "top_15_uber_reply_templates": top.to_dict(),
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nTop-15 templates cover {stats['top15_template_share']:.0%} of Uber replies")
    print(f"Done. Files written to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--chunksize", type=int, default=100_000)
    a = ap.parse_args()
    main(a.csv, a.out, a.chunksize)