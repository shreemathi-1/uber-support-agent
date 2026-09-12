"""
Split Uber's replies into "deflection" (send us a DM / a note) and "informative" (an actual answer).
Reads data/uber_pairs.jsonl, writes data/pairs_{informative,deflection}.jsonl and a `reply_split` block in data/stats.json.
"""
import json
import random
import re
from pathlib import Path

DATA = Path("data")
SEED = 42

DEFLECTION_RE = re.compile(
    r"\bDM\b|direct message|send us a note|reach out|fill out|follow[ -]?up|be in touch|"
    r"\bconnect|been in touch|contact us via|"
    # session-2 fix: redirects that the first list missed (DECISIONS.md #11)
    r"following up|be in contact|in[- ]?app (support|messaging|help|message)|send a note|"
    r"reach this link|check this link|follow the link|for more info|learn more|more information",
    re.IGNORECASE,
)


def classify_reply(reply: str) -> str:
    """'deflection' if the reply only redirects the customer elsewhere, else 'informative'."""
    return "deflection" if DEFLECTION_RE.search(reply) else "informative"


def main() -> None:
    pairs = [json.loads(line) for line in (DATA / "uber_pairs.jsonl").open()]
    split: dict[str, list[dict]] = {"deflection": [], "informative": []}
    for p in pairs:
        split[classify_reply(p["brand_reply"])].append(p)

    for label, rows in split.items():
        with (DATA / f"pairs_{label}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_def, n_inf = len(split["deflection"]), len(split["informative"])
    stats = json.loads((DATA / "stats.json").read_text())
    stats["reply_split"] = {
        "deflection": n_def,
        "informative": n_inf,
        "informative_share": round(n_inf / len(pairs), 4),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print(f"pairs={len(pairs)} deflection={n_def} informative={n_inf} informative_share={n_inf / len(pairs):.3f}")
    print("\n10 random informative replies (eyeball check):")
    for p in random.Random(SEED).sample(split["informative"], 10):
        print(f"  CUSTOMER: {p['customer_text'][:100]}\n  UBER:     {p['brand_reply'][:160]}\n")


if __name__ == "__main__":
    main()
