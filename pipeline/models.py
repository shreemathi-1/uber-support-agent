"""
Pydantic contracts shared by the pipeline, the API and the eval harness.
Every stage passes one of these objects to the next; nothing else crosses a stage boundary.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pipeline.taxonomy import Intent

NON_NUMERIC = re.compile(r"[^\d.]")


class Entities(BaseModel):
    email: str | None = None
    trip_date: str | None = None
    city: str | None = None
    amount: float | None = None
    mentions_safety: bool = False
    mentions_legal: bool = False
    is_repeat_contact: bool = False

    @field_validator("amount", mode="before")
    @classmethod
    def _amount_from_currency_string(cls, v: object) -> object:
        """The model sometimes returns "£4.25" or "Rs 555" despite the schema; keep the number, drop the symbol.
        Done here rather than in the prompt so enrich cache keys do not change (DECISIONS.md #47)."""
        if isinstance(v, str):
            digits = NON_NUMERIC.sub("", v)
            try:
                return float(digits) if digits else None
            except ValueError:  # e.g. "4.2.5": let the normal float error surface and the retry handle it
                return v
        return v


class Enrichment(BaseModel):
    """Output of the single enrich LLM call."""
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    sentiment: Literal["negative", "neutral", "positive"]
    urgency: Literal["low", "medium", "high"]
    entities: Entities


class Turn(BaseModel):
    role: Literal["user", "agent"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []  # last 2 turns max
    conversation_id: str | None = None

    @field_validator("history")
    @classmethod
    def _max_two_turns(cls, v: list[Turn]) -> list[Turn]:
        if len(v) > 2:
            raise ValueError("history holds at most the last 2 turns")
        return v


class RetrievedPair(BaseModel):
    """One historical customer→Uber exchange from the Chroma `pairs` collection. score = cosine distance, lower is closer."""
    id: str
    customer_text: str
    reply_text: str
    intent: str
    source: str  # "informative" | "deflection"
    score: float


class RetrievedChunk(BaseModel):
    """One help-centre article chunk from the Chroma `help` collection. score = cosine distance, lower is closer."""
    id: str
    title: str
    source_url: str
    intent: str
    text: str
    score: float


class Context(BaseModel):
    """What retrieval hands the drafter: similar historical exchanges and help-centre facts, nearest first."""
    examples: list[RetrievedPair] = []
    facts: list[RetrievedChunk] = []


class ChatResponse(BaseModel):
    conversation_id: str
    intent: Intent
    confidence: float
    sentiment: str
    urgency: str
    entities: Entities
    reply: str
    action: Literal["auto", "escalate"]
    reason: str  # rule name, "unsafe_draft", or "auto"
    retrieved_ids: list[str]
    latency_ms: int
