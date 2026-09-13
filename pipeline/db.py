"""
SQLite access for the app: connection factory, schema, and the handful of queries run.py and the API need.
Tables: tickets (every request), escalations (queue for humans), llm_cache (every LLM call).
"""
import json
import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "data/app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    message         TEXT NOT NULL,
    intent          TEXT NOT NULL,
    confidence      REAL NOT NULL,
    sentiment       TEXT NOT NULL,
    urgency         TEXT NOT NULL,
    entities_json   TEXT NOT NULL,
    reply           TEXT NOT NULL,
    action          TEXT NOT NULL,
    reason          TEXT NOT NULL,
    retrieved_ids   TEXT NOT NULL,
    latency_ms      INTEGER NOT NULL,
    trace           TEXT
);

CREATE TABLE IF NOT EXISTS escalations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       INTEGER NOT NULL REFERENCES tickets(id),
    conversation_id TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    message         TEXT NOT NULL,
    intent          TEXT NOT NULL,
    urgency         TEXT NOT NULL,
    reason          TEXT NOT NULL,
    entities_json   TEXT NOT NULL,
    draft           TEXT,
    retrieved_ids   TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key           TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    response      TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def db_path() -> str:
    """Resolved at call time so tests can point DB_PATH at a temp file."""
    return os.environ.get("DB_PATH", DEFAULT_DB_PATH)


def get_conn() -> sqlite3.Connection:
    """Open a connection with dict-like rows. Creates the parent dir if missing."""
    path = db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they do not exist, and add columns older databases lack. Safe to call repeatedly."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tickets)")}
        if "trace" not in columns:  # tickets created before the Trace page; SQLite has no ADD COLUMN IF NOT EXISTS
            conn.execute("ALTER TABLE tickets ADD COLUMN trace TEXT")


def _insert(table: str, row: dict) -> int:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with get_conn() as conn:
        cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
        return int(cur.lastrowid)


def insert_ticket(row: dict) -> int:
    return _insert("tickets", row)


def insert_escalation(row: dict) -> int:
    return _insert("escalations", row)


def set_ticket_reply(ticket_id: int, reply: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE tickets SET reply = ? WHERE id = ?", (reply, ticket_id))


def set_ticket_trace(ticket_id: int, trace_json: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE tickets SET trace = ? WHERE id = ?", (trace_json, ticket_id))


def list_traces(limit: int = 50) -> list[dict]:
    """Newest tickets first, one summary row each; has_trace is false for rows written before the trace column existed."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, created_at AS timestamp, message, intent, action, reason, latency_ms, trace IS NOT NULL AS has_trace "
            "FROM tickets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "has_trace": bool(r["has_trace"])} for r in rows]


def get_trace(ticket_id: int) -> dict | None:
    """{id, timestamp, trace} with the trace parsed, or None when there is no such ticket."""
    with get_conn() as conn:
        row = conn.execute("SELECT id, created_at AS timestamp, trace FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "timestamp": row["timestamp"], "trace": json.loads(row["trace"]) if row["trace"] else None}


def list_escalations(status: str = "open") -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM escalations WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["entities"] = json.loads(d.pop("entities_json"))
        d["retrieved_ids"] = json.loads(d["retrieved_ids"])
        out.append(d)
    return out


def resolve_escalation(escalation_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("UPDATE escalations SET status = 'resolved' WHERE id = ? AND status = 'open'", (escalation_id,))
        return cur.rowcount == 1


def count(table: str) -> int:
    with get_conn() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
