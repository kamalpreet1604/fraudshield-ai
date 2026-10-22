"""Save every scored payment in a small SQLite database, so there is a history of decisions.

SQLite is a whole database in one file. No server to install: Python has it built in.
The file is data/predictions.db by default (set FRAUD_DB_PATH to use another file, e.g. in tests).
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def db_path() -> Path:
    return Path(os.environ.get("FRAUD_DB_PATH", "data/predictions.db"))


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row   # rows can be read like dicts
    # create the table the first time; does nothing if it already exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            amount      REAL NOT NULL,
            product     TEXT,
            fraud_score REAL NOT NULL,
            decision    TEXT NOT NULL
        )
        """
    )
    return conn


def save_prediction(amount: float, product: str | None, fraud_score: float, decision: str) -> int:
    """Store one scored payment and return its id."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, amount, product, fraud_score, decision) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), amount, product, fraud_score, decision),
        )
        new_id = cur.lastrowid
    conn.close()
    return new_id


def recent(limit: int = 20) -> list[dict]:
    """The newest scored payments first."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
