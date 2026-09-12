"""
FastAPI surface over pipeline.run: POST /chat, the escalation queue, and a health endpoint.
No auth; CORS open to any localhost origin (the Vite dev server, whichever port it lands on).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipeline import db, llm
from pipeline.models import ChatRequest, ChatResponse
from pipeline.retrieve import _model, get_collection
from pipeline.run import run

logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    _model()  # load MiniLM once at startup rather than on the first request
    get_collection("pairs"), get_collection("help")
    yield


app = FastAPI(title="Uber_Support agent", lifespan=lifespan)
# Any localhost port: Vite moves to 5174+ when 5173 is busy, and this API has no auth to protect anyway.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
                   allow_methods=["*"], allow_headers=["*"])


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return run(request)


@app.get("/escalations")
def escalations(status: str = "open") -> list[dict]:
    return db.list_escalations(status)


@app.post("/escalations/{escalation_id}/resolve")
def resolve(escalation_id: int) -> dict:
    if not db.resolve_escalation(escalation_id):
        raise HTTPException(status_code=404, detail="no open escalation with that id")
    return {"id": escalation_id, "status": "resolved"}


@app.get("/health")
def health() -> dict:
    return {
        "models": {"primary": llm.primary_model(), "fallback": llm.fallback_model()},
        "index": {"pairs": get_collection("pairs").count(), "help": get_collection("help").count()},
        "cache_rows": db.count("llm_cache"),
    }
