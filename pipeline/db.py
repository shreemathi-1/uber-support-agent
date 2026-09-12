"""
SQLite access for the app: one connection factory and one schema initialiser.
Tables: tickets (auto-handled), escalations (routed to a human), llm_cache (every LLM call).
"""
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
    latency_ms      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS escalations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    message         TEXT NOT NULL,
    intent          TEXT NOT NULL,
    urgency         TEXT NOT NULL,
    reason          TEXT NOT NULL,
    entities_json   TEXT NOT NULL,
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
    """Create all tables if they do not exist. Safe to call repeatedly."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
