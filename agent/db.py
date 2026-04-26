"""SQLite persistence for LiveOps Agent.

Single file at <repo>/data/liveops.sqlite3. WAL mode so the dashboard, the
runner, and the auto loop can all touch it concurrently without locking.

Tables:
- users:    username PK, password_hash, created_at
- anomalies: anomaly events with metric + z-score + LLM explanation
- actions:   action events with outcome (pending/success/failed/ignored)

A best-effort one-time CSV importer (`migrate_csv_if_needed`) folds the
legacy `data/anomaly_log.csv` and `data/action_log.csv` into the db on first
run, then leaves a `.imported` marker so it doesn't run again.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("LIVEOPS_DB", str(BASE_DIR / "data" / "liveops.sqlite3")))


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anomalies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    username    TEXT,
    region      TEXT,
    product_id  TEXT,
    orders      INTEGER,
    inventory   INTEGER,
    revenue     REAL,
    metric      TEXT NOT NULL DEFAULT 'revenue',
    z_score     REAL,
    explanation TEXT
);
CREATE INDEX IF NOT EXISTS ix_anomalies_user_ts ON anomalies(username, ts);

CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    username    TEXT,
    region      TEXT,
    product_id  TEXT,
    action      TEXT NOT NULL,
    outcome     TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS ix_actions_user_ts ON actions(username, ts);
CREATE INDEX IF NOT EXISTS ix_actions_outcome ON actions(outcome);
"""


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Yield a connection with WAL + foreign keys enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if missing. Safe to call repeatedly."""
    with connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------- users ----------------------------------------

def create_user(username: str, password_hash: str) -> bool:
    """Return True if created, False if username already exists."""
    init_db()
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, _now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_user(username: str) -> Optional[sqlite3.Row]:
    init_db()
    with connect() as conn:
        cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cur.fetchone()


def list_usernames() -> list[str]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT username FROM users ORDER BY username").fetchall()
        return [r["username"] for r in rows]


# ---------------------------- anomalies ------------------------------------

def insert_anomaly(
    *,
    username: Optional[str],
    region: str,
    product_id: str,
    orders: int,
    inventory: int,
    revenue: float,
    metric: str = "revenue",
    z_score: Optional[float] = None,
    explanation: str = "",
) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO anomalies
               (ts, username, region, product_id, orders, inventory, revenue, metric, z_score, explanation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), username, region, product_id, orders, inventory, revenue, metric, z_score, explanation),
        )
        return int(cur.lastrowid)


def read_anomalies(username: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
    init_db()
    with connect() as conn:
        if username:
            df = pd.read_sql_query(
                "SELECT * FROM anomalies WHERE username = ? ORDER BY id DESC LIMIT ?",
                conn, params=(username, limit),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM anomalies ORDER BY id DESC LIMIT ?",
                conn, params=(limit,),
            )
    return df


# ---------------------------- actions --------------------------------------

def insert_action(
    *,
    username: Optional[str],
    region: str,
    product_id: str,
    action: str,
    outcome: str = "pending",
) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO actions(ts, username, region, product_id, action, outcome)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (_now(), username, region, product_id, action, outcome),
        )
        return int(cur.lastrowid)


def update_action_outcome(action_id: int, outcome: str) -> None:
    init_db()
    with connect() as conn:
        conn.execute("UPDATE actions SET outcome = ? WHERE id = ?", (outcome, action_id))


def read_actions(username: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
    init_db()
    with connect() as conn:
        if username:
            df = pd.read_sql_query(
                "SELECT * FROM actions WHERE username = ? ORDER BY id DESC LIMIT ?",
                conn, params=(username, limit),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM actions ORDER BY id DESC LIMIT ?",
                conn, params=(limit,),
            )
    return df


def action_success_rates(username: Optional[str] = None) -> pd.Series:
    """Return success-rate per action label (success / total non-pending)."""
    init_db()
    with connect() as conn:
        if username:
            df = pd.read_sql_query(
                "SELECT action, outcome FROM actions WHERE username = ?",
                conn, params=(username,),
            )
        else:
            df = pd.read_sql_query("SELECT action, outcome FROM actions", conn)

    if df.empty:
        return pd.Series(dtype=float)
    decided = df[df["outcome"].isin(["success", "failed"])]
    if decided.empty:
        return pd.Series(dtype=float)
    grouped = decided.groupby("action")["outcome"]
    return (grouped.apply(lambda s: (s == "success").sum() / len(s))).astype(float)


# ---------------------------- one-time CSV migration -----------------------

def migrate_csv_if_needed() -> None:
    """Fold legacy CSV logs into the db once. Idempotent."""
    init_db()
    marker = DB_PATH.parent / ".csv_imported"
    if marker.exists():
        return

    anomaly_csv = BASE_DIR / "data" / "anomaly_log.csv"
    action_csv = BASE_DIR / "data" / "action_log.csv"

    try:
        if anomaly_csv.exists():
            df = pd.read_csv(anomaly_csv)
            with connect() as conn:
                for _, r in df.iterrows():
                    conn.execute(
                        """INSERT INTO anomalies
                           (ts, username, region, product_id, orders, inventory, revenue, metric, z_score, explanation)
                           VALUES (?, NULL, ?, ?, ?, ?, ?, 'revenue', NULL, ?)""",
                        (
                            str(r.get("timestamp", _now())),
                            str(r.get("region", "")),
                            str(r.get("product_id", "")),
                            int(r.get("orders", 0) or 0),
                            int(r.get("inventory", 0) or 0),
                            float(r.get("revenue", 0.0) or 0.0),
                            str(r.get("explanation", "")),
                        ),
                    )
    except Exception as e:
        print(f"[db] anomaly CSV import skipped: {e}")

    try:
        if action_csv.exists():
            df = pd.read_csv(action_csv)
            with connect() as conn:
                for _, r in df.iterrows():
                    conn.execute(
                        """INSERT INTO actions(ts, username, region, product_id, action, outcome)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            str(r.get("timestamp", _now())),
                            str(r.get("username", "") or ""),
                            str(r.get("region", "")),
                            str(r.get("product_id", "")),
                            str(r.get("action", "")),
                            str(r.get("outcome", "pending") or "pending"),
                        ),
                    )
    except Exception as e:
        print(f"[db] action CSV import skipped: {e}")

    try:
        marker.touch()
    except Exception:
        pass
