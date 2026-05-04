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

-- One row per dedupe_key. We update ts on each successful claim so the
-- cooldown window slides from the last fire, not the first.
CREATE TABLE IF NOT EXISTS notifications (
    dedupe_key TEXT PRIMARY KEY,
    ts         TEXT NOT NULL,
    severity   TEXT
);
CREATE INDEX IF NOT EXISTS ix_notifications_ts ON notifications(ts);

-- Per-user persisted workbench uploads.  The on-disk file lives under
-- user_data/workbench_uploads/<username>/<file>.  This table is the index.
CREATE TABLE IF NOT EXISTS user_datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL,
    file          TEXT NOT NULL,         -- on-disk basename, unique per user
    original_name TEXT NOT NULL,         -- what the user uploaded
    n_rows        INTEGER NOT NULL,
    n_cols        INTEGER NOT NULL,
    uploaded_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_datasets_user ON user_datasets(username, uploaded_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS ux_user_datasets_user_file ON user_datasets(username, file);

-- Per-user question history. Stores the question + the planned tools, NOT
-- the full answer (those can be huge). Pinned rows are never auto-pruned.
CREATE TABLE IF NOT EXISTS user_questions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT NOT NULL,
    question       TEXT NOT NULL,
    backend        TEXT NOT NULL,
    actual_backend TEXT,
    planned_tools  TEXT,          -- JSON array
    dataset_name   TEXT,          -- which dataset was active when asked
    pinned         INTEGER NOT NULL DEFAULT 0,
    asked_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_questions_user_ts
    ON user_questions(username, pinned DESC, asked_at DESC);

-- Per-user encrypted connector credentials (Slack webhooks, SMTP creds, …).
-- The `value_encrypted` column holds a Fernet token; never store plaintext.
CREATE TABLE IF NOT EXISTS user_connectors (
    username        TEXT NOT NULL,
    kind            TEXT NOT NULL,        -- slack_webhook, smtp_host, etc.
    value_encrypted BLOB NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (username, kind)
);
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


# ---------------------------- user datasets --------------------------------

def insert_user_dataset(*, username: str, file: str, original_name: str,
                        n_rows: int, n_cols: int) -> int:
    """Persist an uploaded dataset and return the new row id.  Raises
    sqlite3.IntegrityError if (username, file) already exists."""
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO user_datasets(username, file, original_name, "
            "n_rows, n_cols, uploaded_at) VALUES (?, ?, ?, ?, ?, ?)",
            (username, file, original_name, int(n_rows), int(n_cols), _now()),
        )
        return int(cur.lastrowid)


def list_user_datasets(username: str) -> list[dict]:
    """Newest first."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, file, original_name, n_rows, n_cols, uploaded_at "
            "FROM user_datasets WHERE username = ? ORDER BY uploaded_at DESC",
            (username,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_user_dataset(username: str, dataset_id: int) -> Optional[dict]:
    init_db()
    with connect() as conn:
        r = conn.execute(
            "SELECT id, file, original_name, n_rows, n_cols, uploaded_at "
            "FROM user_datasets WHERE username = ? AND id = ?",
            (username, int(dataset_id)),
        ).fetchone()
        return dict(r) if r else None


# ---------------------------- user connectors ------------------------------

def upsert_user_connector(*, username: str, kind: str,
                          value_encrypted: bytes) -> None:
    """Insert or update an encrypted connector value for a user."""
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT INTO user_connectors(username, kind, value_encrypted, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(username, kind) DO UPDATE SET "
            "  value_encrypted = excluded.value_encrypted, "
            "  updated_at      = excluded.updated_at",
            (username, kind, value_encrypted, _now()),
        )


def get_user_connector(username: str, kind: str) -> Optional[bytes]:
    init_db()
    with connect() as conn:
        r = conn.execute(
            "SELECT value_encrypted FROM user_connectors "
            "WHERE username = ? AND kind = ?",
            (username, kind),
        ).fetchone()
        return bytes(r["value_encrypted"]) if r else None


def list_user_connectors(username: str) -> list[dict]:
    """Returns [{kind, updated_at}, …] — never decrypted values."""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT kind, updated_at FROM user_connectors "
            "WHERE username = ? ORDER BY kind",
            (username,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_user_connector(username: str, kind: str) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM user_connectors WHERE username = ? AND kind = ?",
            (username, kind),
        )
        return cur.rowcount > 0


# ---------------------------- user questions -------------------------------

def insert_user_question(*, username: str, question: str, backend: str,
                         actual_backend: Optional[str],
                         planned_tools: list[str],
                         dataset_name: Optional[str],
                         max_unpinned: int = 50) -> int:
    """Persist a workbench question and prune the user's unpinned history
    down to `max_unpinned` rows.  Returns the new row id."""
    import json as _j
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO user_questions("
            "  username, question, backend, actual_backend, planned_tools,"
            "  dataset_name, asked_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, question, backend, actual_backend,
             _j.dumps(planned_tools or []), dataset_name, _now()),
        )
        new_id = int(cur.lastrowid)
        # Prune oldest UNPINNED rows beyond max_unpinned for this user.
        conn.execute(
            "DELETE FROM user_questions "
            "WHERE id IN ("
            "  SELECT id FROM user_questions "
            "  WHERE username = ? AND pinned = 0 "
            "  ORDER BY asked_at DESC LIMIT -1 OFFSET ?"
            ")",
            (username, int(max_unpinned)),
        )
    return new_id


def list_user_questions(username: str, *, limit: int = 50) -> list[dict]:
    """Pinned first, then newest first. Returns dicts with parsed planned_tools."""
    import json as _j
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, question, backend, actual_backend, planned_tools, "
            "       dataset_name, pinned, asked_at "
            "FROM user_questions WHERE username = ? "
            "ORDER BY pinned DESC, asked_at DESC LIMIT ?",
            (username, int(limit)),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["planned_tools"] = _j.loads(d["planned_tools"] or "[]")
            except Exception:
                d["planned_tools"] = []
            d["pinned"] = bool(d["pinned"])
            out.append(d)
        return out


def set_user_question_pinned(username: str, question_id: int,
                              pinned: bool) -> bool:
    """Toggle pinned. Returns True if a row was updated."""
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "UPDATE user_questions SET pinned = ? "
            "WHERE username = ? AND id = ?",
            (1 if pinned else 0, username, int(question_id)),
        )
        return cur.rowcount > 0


def clear_user_questions(username: str, *, keep_pinned: bool = True) -> int:
    """Delete the user's history; pinned rows survive by default.
    Returns the number of deleted rows."""
    init_db()
    with connect() as conn:
        if keep_pinned:
            cur = conn.execute(
                "DELETE FROM user_questions WHERE username = ? AND pinned = 0",
                (username,),
            )
        else:
            cur = conn.execute(
                "DELETE FROM user_questions WHERE username = ?",
                (username,),
            )
        return cur.rowcount


def delete_user_dataset(username: str, dataset_id: int) -> Optional[str]:
    """Returns the deleted file basename so the caller can unlink it from
    disk.  None if no row matched (so the caller can 404)."""
    init_db()
    with connect() as conn:
        r = conn.execute(
            "SELECT file FROM user_datasets WHERE username = ? AND id = ?",
            (username, int(dataset_id)),
        ).fetchone()
        if not r:
            return None
        conn.execute(
            "DELETE FROM user_datasets WHERE username = ? AND id = ?",
            (username, int(dataset_id)),
        )
        return r["file"]


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


# ---------------------------- notification dedupe -------------------------

def should_notify(dedupe_key: str, cooldown_seconds: int = 1800,
                  severity: Optional[str] = None) -> bool:
    """Atomically check + claim a dedupe slot.

    Returns True if no entry for this key exists within the cooldown window.
    On True, also writes/updates the row so subsequent calls within the
    window return False. Returns True (fail-open) if the dedupe table
    operation itself errors — we'd rather over-notify than silently drop.
    """
    if not dedupe_key:
        return True
    init_db()
    now = datetime.utcnow()
    try:
        with connect() as conn:
            row = conn.execute(
                "SELECT ts FROM notifications WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
            if row is not None:
                last = datetime.strptime(row["ts"], "%Y-%m-%dT%H:%M:%S")
                age = (now - last).total_seconds()
                if age < cooldown_seconds:
                    return False
            # Fire allowed — refresh the slot.
            conn.execute(
                """INSERT INTO notifications(dedupe_key, ts, severity)
                   VALUES (?, ?, ?)
                   ON CONFLICT(dedupe_key) DO UPDATE
                   SET ts = excluded.ts, severity = excluded.severity""",
                (dedupe_key, now.strftime("%Y-%m-%dT%H:%M:%S"), severity),
            )
        return True
    except Exception as e:
        print(f"[db:dedupe] error (failing open): {e}")
        return True


def recent_notifications(limit: int = 50) -> pd.DataFrame:
    init_db()
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT dedupe_key, ts, severity FROM notifications ORDER BY ts DESC LIMIT ?",
            conn, params=(limit,),
        )


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
