"""
Constants for the trace's cost line: Groq list prices per million tokens and a token estimator.
The free tier pays nothing; the number answers "what would this request have cost at list price".
"""

# USD per 1M tokens (input, output). Groq pay-as-you-go list prices as of 2026-09-13; cached-input discounts ignored.
# Source: https://groq.com/pricing (via https://www.cloudzero.com/blog/groq-pricing/ summary). Recheck before quoting in the report.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

# The cache stores response text only, and tiktoken's o200k_base needs a network download on first use (hangs offline),
# so tokens are estimated from characters. English averages ~4 chars/token; label it so nobody mistakes it for usage.
CHARS_PER_TOKEN = 4
TOKEN_COUNTER = "chars/4 estimate"


def count_tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN) if text else 0


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """0.0 for a model not in the table rather than a guess."""
    price_in, price_out = PRICES_PER_M.get(model, (0.0, 0.0))
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 6)
