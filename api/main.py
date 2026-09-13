"""
FastAPI surface over pipeline.run: POST /chat, the escalation queue, traces for the Trace page, eval metrics, health.
No auth; CORS open to any localhost origin (the Vite dev server, whichever port it lands on).
"""
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipeline import cache, db, llm
from pipeline.cache import CacheMissError
from pipeline.models import ChatRequest, ChatResponse
from pipeline.retrieve import _model, get_collection
from pipeline.run import run

logging.basicConfig(level=logging.INFO, format="%(message)s")
SUMMARY_PATH = Path("eval/results/summary.json")


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
def chat(request: ChatRequest, x_cache_only: str | None = Header(default=None)) -> ChatResponse:
    """`X-Cache-Only: 1` (the Trace page's Replay button) answers from llm_cache only for this request; a miss is a 409."""
    token = cache.FORCE_CACHE_ONLY.set(x_cache_only == "1")
    try:
        return run(request)
    except CacheMissError as err:
        raise HTTPException(status_code=409, detail=f"not cached, replay refused: {err}")
    finally:
        cache.FORCE_CACHE_ONLY.reset(token)


@app.get("/escalations")
def escalations(status: str = "open") -> list[dict]:
    return db.list_escalations(status)


@app.post("/escalations/{escalation_id}/resolve")
def resolve(escalation_id: int) -> dict:
    if not db.resolve_escalation(escalation_id):
        raise HTTPException(status_code=404, detail="no open escalation with that id")
    return {"id": escalation_id, "status": "resolved"}


@app.get("/traces")
def traces(limit: int = 50) -> list[dict]:
    return db.list_traces(limit)


@app.get("/traces/{ticket_id}")
def trace(ticket_id: int) -> dict:
    row = db.get_trace(ticket_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no ticket with that id")
    return row


@app.get("/metrics")
def metrics() -> dict:
    """eval/results/summary.json (one entry per system) wrapped with when it was written; {"available": false} until make eval ran."""
    if not SUMMARY_PATH.exists():
        return {"available": False}
    written = datetime.fromtimestamp(SUMMARY_PATH.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return {"available": True, "written_at": written, "systems": json.loads(SUMMARY_PATH.read_text())}


@app.get("/health")
def health() -> dict:
    return {
        "models": {"primary": llm.primary_model(), "fallback": llm.fallback_model()},
        "index": {"pairs": get_collection("pairs").count(), "help": get_collection("help").count()},
        "cache_rows": db.count("llm_cache"),
    }
