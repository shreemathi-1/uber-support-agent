"""
Probe each configured Groq model with one tiny json_mode call and assert the reply parses as JSON.
Goes through the cache like every other call; pass --fresh to bypass a cached answer with a timestamp nonce.
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import llm  # noqa: E402
from pipeline.cache import cached_llm  # noqa: E402


def probe(model: str, fresh: bool) -> None:
    nonce = f" (probe {int(time.time())})" if fresh else ""
    messages = [
        {"role": "system", "content": "Reply with JSON only."},
        {"role": "user", "content": f'Return {{"ok": true, "model": "<your model name>"}}{nonce}'},
    ]
    call = partial(llm.groq_chat, temperature=0.0, max_tokens=100, json_mode=True, reasoning_effort="low")
    raw = cached_llm(model, messages, call)
    print(f"{model}: {raw!r}")
    parsed = json.loads(raw)
    assert parsed.get("ok") is True, f"{model} answered but not with ok=true: {parsed}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true", help="bypass the cache with a timestamp nonce")
    args = ap.parse_args()
    for model in (llm.primary_model(), llm.fallback_model()):
        probe(model, args.fresh)
    print("both models OK")


if __name__ == "__main__":
    main()
