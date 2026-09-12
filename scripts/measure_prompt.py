"""
Count tokens in the enrich system prompt with tiktoken's o200k_base, the closest public ruler for gpt-oss's o200k_harmony vocabulary.
Prompt tokens are paid on every call, so this number times the calls per day is the quota budget.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiktoken  # noqa: E402

from pipeline.enrich import build_system_prompt  # noqa: E402


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("o200k_base").encode(text))


if __name__ == "__main__":
    prompt = build_system_prompt()
    print(f"enrich system prompt: {count_tokens(prompt)} tokens, {len(prompt)} chars, {len(prompt.splitlines())} lines")
