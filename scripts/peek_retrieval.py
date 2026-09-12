"""
Eyeball check for retrieval: ten fixed queries (one per intent plus two ambiguous) with their top pairs and help chunks.
Prints customer text, source and distance for pairs; title and distance for help chunks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieve import retrieve  # noqa: E402

QUERIES: list[tuple[str, str]] = [
    ("I was charged twice for the same trip last night, $24 each", "fare_dispute"),
    ("my driver took a massive detour and dropped me at the wrong place", "trip_or_driver_issue"),
    ("the app keeps saying verification code invalid and I can't log in", "app_or_account_issue"),
    ("applied to drive 3 weeks ago, background check still pending, no update", "driver_onboarding_or_earnings"),
    ("my Eats order arrived cold and half of it was missing", "uber_eats"),
    ("left my wallet in the back seat an hour ago, how do I reach the driver", "lost_item"),
    ("is Uber available in Coimbatore? how does surge pricing work", "policy_or_info_question"),
    ("lol ok thanks I guess", "other"),
    # ambiguous: could be fare or trip; could be app or eats
    ("driver cancelled when he got here and now I have a $5 fee", "fare_dispute"),
    ("promo code says expired but the email came today, on the food app", "uber_eats"),
]


def main() -> None:
    for query, intent in QUERIES:
        ctx = retrieve(query, intent)
        print(f"\n=== [{intent}] {query}")
        for ex in ctx.examples:
            print(f"  pair  {ex.score:.3f} [{ex.source[:4]}|{ex.intent[:12]:12}] {ex.customer_text[:90]}")
            print(f"        -> {ex.reply_text[:90]}")
        for fact in ctx.facts:
            print(f"  help  {fact.score:.3f} [{fact.intent[:12]:12}] {fact.title}")


if __name__ == "__main__":
    main()
