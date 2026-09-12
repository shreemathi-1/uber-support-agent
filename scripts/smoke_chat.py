"""
End-to-end smoke: ten fixed messages through pipeline.run, printed as a table. Not a metric; a plumbing check.
Respects CACHE_ONLY: a cache miss stops the run with a clear message.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import llm  # noqa: E402
from pipeline.cache import CacheMissError  # noqa: E402
from pipeline.models import ChatRequest, Turn  # noqa: E402
from pipeline.run import run  # noqa: E402

MESSAGES: list[tuple[str, list[Turn]]] = [
    ("I was charged twice for the same trip last night", []),
    ("my driver took a huge detour and dropped me at the wrong address", []),
    ("the app won't send me a verification code so I can't log in", []),
    ("applied to drive 3 weeks ago and my background check is still pending", []),
    ("my Eats order arrived cold and half of it was missing", []),
    ("left my wallet in the back seat an hour ago, how do I reach the driver", []),
    ("is Uber available in Coimbatore?", []),
    ("thanks, that sorted it", []),
    ("Still no reply. I already emailed twice about my locked account", [Turn(role="user", content="my account got locked after a password reset")]),
    ("my driver was drunk and swerving, I felt unsafe the whole ride", []),
    ("you charged me $25 for a ride that never happened", []),
]


def main() -> None:
    llm.batch_mode()
    logging.basicConfig(level=logging.WARNING)
    print(f"{'message':52} {'intent':22} {'reason':14} {'action':8} reply")
    for text, history in MESSAGES:
        try:
            r = run(ChatRequest(message=text, history=history))
        except CacheMissError as err:
            sys.exit(f"CACHE_ONLY=1 and not cached: {err}")
        print(f"{text[:50]:52} {r.intent.value[:22]:22} {r.reason[:14]:14} {r.action:8} {r.reply}")


if __name__ == "__main__":
    main()
