"""
Draft step: one LLM call writes the in-app reply in Uber's voice, grounded in retrieved examples (tone) and help facts.
Post-processing strips links, caps at 280 characters on a sentence boundary, and prefixes repeat-contact acknowledgement.
"""
import re
from functools import partial

from pipeline import llm
from pipeline.cache import CallFn, cached_llm
from pipeline.models import Context, Enrichment, RetrievedChunk, Turn
from pipeline.taxonomy import INTENT_DEFINITIONS

MAX_CHARS = 280
MAX_FACT_DISTANCE = 0.6  # help chunks farther than this are not shown to the drafter (see peek_retrieval output)
REPEAT_PREFIX = "Sorry you've had to contact us again. "
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|co|net|org)\b\S*)", re.I)

SYSTEM_PROMPT = """You write replies for Uber's in-app support chat. Voice: plain, warm, direct. At most one short apology. No exclamation marks, no emoji, no marketing. Under 280 characters total.

Write exactly this shape, in this order, as one short paragraph:
(a) Acknowledge the customer's specific issue in one sentence.
(b) ONLY if help centre facts are supplied: one concrete self-serve step in one sentence, taken from those facts.
(c) Ask for exactly one identifier, the one named under "ask for". If the customer already gave it, do not ask again.
(d) One sentence on what happens next.

Hard rules:
- This is an in-app chat. Never say "DM us", "send us a note", "reach out via", "contact us at", and never include any link or URL.
- Never promise a refund, credit, adjustment, or any timeline.
- Never state a policy or procedure that is not in the supplied facts.
- The style examples show tone only. Do not copy their links or DM requests.
Reply with the message text only."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def usable_facts(context: Context) -> list[RetrievedChunk]:
    return [f for f in context.facts if f.score <= MAX_FACT_DISTANCE]


def build_user_prompt(message: str, history: list[Turn], enrichment: Enrichment, context: Context) -> str:
    examples = sorted(context.examples, key=lambda e: (e.source != "informative", e.score))[:5]
    facts = usable_facts(context)
    lines = [f"Intent: {enrichment.intent.value}", f"Ask for: {INTENT_DEFINITIONS[enrichment.intent]['ask_for']}", ""]
    lines.append("Style examples (tone only, do not copy links or DM requests):")
    lines += [f"Customer: {e.customer_text}\nUber: {e.reply_text}" for e in examples] or ["(none)"]
    lines.append("")
    if facts:
        lines.append("Help centre facts (you may use):")
        lines += [f"[{f.title}] {f.text}" for f in facts]
    else:
        lines.append("No help centre facts available - skip step (b).")
    lines.append("")
    lines += [f"Earlier ({t.role}): {t.content}" for t in history]
    lines.append(f"Customer message: {message}")
    return "\n".join(lines)


def postprocess_with_notes(text: str, is_repeat_contact: bool) -> tuple[str, list[str]]:
    """(final text, names of the steps that changed it). The notes go into the trace; the text goes to the customer."""
    notes: list[str] = []
    stripped = URL_RE.sub("", text)
    if stripped != text:
        notes.append("stripped_urls")
    text = re.sub(r"\s+", " ", stripped).strip()
    if text != stripped:
        notes.append("collapsed_whitespace")
    if is_repeat_contact and "again" not in text.lower():
        text = REPEAT_PREFIX + text
        notes.append("added_repeat_contact_prefix")
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        text = cut[: end + 1] if end > 0 else cut.rstrip()
        notes.append(f"cut_to_{MAX_CHARS}_chars")
    return text, notes


def postprocess(text: str, is_repeat_contact: bool) -> str:
    return postprocess_with_notes(text, is_repeat_contact)[0]


def default_call_fn() -> CallFn:
    # gpt-oss spends tokens on reasoning before the reply; 160 tokens with effort=None returned empty content (DECISIONS.md #27)
    return partial(llm.groq_chat, temperature=0.3, max_tokens=400, json_mode=False, reasoning_effort="low")


def draft_raw(message: str, history: list[Turn], enrichment: Enrichment, context: Context, call_fn: CallFn | None = None) -> str:
    """The model's reply before post-processing. run.py calls this so the trace can show raw next to final."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(message, history, enrichment, context)},
    ]
    return cached_llm(llm.primary_model(), messages, call_fn or default_call_fn())


def draft(message: str, history: list[Turn], enrichment: Enrichment, context: Context, call_fn: CallFn | None = None) -> str:
    return postprocess(draft_raw(message, history, enrichment, context, call_fn), enrichment.entities.is_repeat_contact)
