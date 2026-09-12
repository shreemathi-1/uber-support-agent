"""
LLM judge for reply quality: Gemini scores five rubric items 1-5 and explains itself, through the same cache as everything else.
A different model family from the drafter so it is not grading its own style. Plus a MiniLM similarity as a tone-only signal.
"""
import json
import os
import time

from pipeline.cache import CallFn, cached_llm

RUBRIC = """You are grading a reply written by Uber's in-app support assistant. Score each item from 1 (bad) to 5 (excellent):

1. acknowledges_issue: the reply names the customer's specific problem, not a generic apology.
2. asks_right_identifier: it asks for exactly one piece of information that would let support act (trip date, email on the account, order number), and does not ask for something the customer already gave.
3. self_serve_step_correct_or_absent: if it gives a self-serve step, the step is supported by the supplied help-centre facts; if it gives none, score 5 when no facts were supplied and 3 when facts were available but unused.
4. no_promises_or_links: 5 if it promises no refund, credit, or timeline and contains no link, DM request, or "contact us elsewhere"; 1 if it does any of those.
5. tone: plain, warm, direct; at most one apology; no exclamation marks or emoji; under 280 characters.

Return JSON only: {"scores": {"acknowledges_issue": n, "asks_right_identifier": n, "self_serve_step_correct_or_absent": n, "no_promises_or_links": n, "tone": n}, "rationale": "<two sentences>"}"""

ITEMS = ["acknowledges_issue", "asks_right_identifier", "self_serve_step_correct_or_absent", "no_promises_or_links", "tone"]


def judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", "gemini-3.5-flash-lite")


def gemini_call_fn(model: str, messages: list[dict]) -> str:
    """call_fn for cached_llm: system turn as system_instruction, user turn as the prompt, JSON out, temperature 0."""
    import google.generativeai as genai  # eval-only dependency, imported here so the pipeline never loads it

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (get one at aistudio.google.com). Use --skip-judge or CACHE_ONLY=1.")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model, system_instruction=messages[0]["content"],
                                   generation_config={"temperature": 0, "response_mime_type": "application/json"})
    text = client.generate_content(messages[1]["content"]).text
    time.sleep(float(os.environ.get("RATE_SLEEP", "2")))
    return text


def judge(message: str, reply: str, facts: list[str], call_fn: CallFn | None = None) -> dict:
    """Returns {"scores": {...}, "total": int, "rationale": str}. Raises on an unparseable answer."""
    fact_text = "\n".join(f"- {f}" for f in facts) or "(none supplied)"
    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"Help-centre facts supplied to the assistant:\n{fact_text}\n\nCustomer message: {message}\n\nAssistant reply: {reply}"},
    ]
    data = json.loads(cached_llm(judge_model(), messages, call_fn or gemini_call_fn))
    scores = {k: max(1, min(5, int(data["scores"][k]))) for k in ITEMS}
    return {"scores": scores, "total": sum(scores.values()), "rationale": str(data.get("rationale", ""))}


def embedding_similarity(reply: str, brand_reply: str) -> float:
    """Cosine similarity of MiniLM embeddings. Tone signal only: it rewards 'DM us your email' for everything."""
    from pipeline.retrieve import embed

    a, b = embed([reply, brand_reply])
    return round(sum(x * y for x, y in zip(a, b)), 4)
