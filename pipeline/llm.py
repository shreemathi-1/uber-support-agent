"""
The only module that talks to Groq. Everyone else passes `groq_chat` (via functools.partial) to cache.cached_llm as call_fn.
Retries 429/5xx/connection errors with exponential backoff (3 attempts), then tries FALLBACK_MODEL once, then raises.
"""
import logging
import os
import time

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, Groq, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()
log = logging.getLogger(__name__)

Messages = list[dict[str, str]]


def primary_model() -> str:
    return os.environ.get("PRIMARY_MODEL", "llama-3.3-70b-versatile")


def fallback_model() -> str:
    return os.environ.get("FALLBACK_MODEL", "llama-3.1-8b-instant")


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


# tenacity retry-with-backoff pattern from https://tenacity.readthedocs.io (see docs/BORROWED.md)
@retry(retry=retry_if_exception(_is_transient), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def _call(client: Groq, model: str, messages: Messages, temperature: float, max_tokens: int, json_mode: bool) -> str:
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra
    )
    time.sleep(float(os.environ.get("RATE_SLEEP", "2")))  # free tier: ~10 RPM on the 70B model
    return resp.choices[0].message.content or ""


def groq_chat(model: str, messages: Messages, temperature: float = 0.0, max_tokens: int = 300,
              json_mode: bool = False) -> str:
    """One chat completion. Signature (model, messages, ...) matches what cache.cached_llm passes to call_fn."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env). Use CACHE_ONLY=1 to run without a key.")
    client = Groq(api_key=api_key)
    try:
        return _call(client, model, messages, temperature, max_tokens, json_mode)
    except Exception as exc:  # retries exhausted or a non-transient error
        if model == fallback_model():
            raise
        log.warning("model %s failed (%s: %s); falling back to %s once", model, type(exc).__name__, exc, fallback_model())
        return _call(client, fallback_model(), messages, temperature, max_tokens, json_mode)
