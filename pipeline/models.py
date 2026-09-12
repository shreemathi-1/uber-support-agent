"""
Pydantic contracts shared by the pipeline, the API and the eval harness.
Every stage passes one of these objects to the next; nothing else crosses a stage boundary.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pipeline.taxonomy import Intent


class Entities(BaseModel):
    email: str | None = None
    trip_date: str | None = None
    city: str | None = None
    amount: float | None = None
    mentions_safety: bool = False
    mentions_legal: bool = False
    is_repeat_contact: bool = False


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
