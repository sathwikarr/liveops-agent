"""Anomaly persistence — backed by SQLite (agent.db).

The CSV-era functions (`save_anomaly_log`, `read_anomaly_log`, `get_log_path`)
are kept as thin compatibility shims so existing callers (main.py,
pages/dashboard.py) don't break.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from agent import db


def get_log_path() -> str:
    """Return the on-disk path of the SQLite database (replaces old CSV path)."""
    return str(db.DB_PATH)


def save_anomaly_log(
    region: str,
    product_id: str,
    orders: int,
    inventory: int,
    revenue: float,
    explanation: str,
    *,
    username: Optional[str] = None,
    metric: str = "revenue",
    z_score: Optional[float] = None,
) -> int:
    """Persist an anomaly. Returns the new row id."""
    return db.insert_anomaly(
        username=username,
        region=region,
        product_id=product_id,
        orders=int(orders or 0),
        inventory=int(inventory or 0),
        revenue=float(revenue or 0.0),
        metric=metric,
        z_score=z_score,
        explanation=explanation or "",
    )


def read_anomaly_log(n: int = 200, username: Optional[str] = None) -> pd.DataFrame:
    """Return last n anomalies. If username is given, filter to that user."""
    db.migrate_csv_if_needed()
    df = db.read_anomalies(username=username, limit=n)
    # Keep column names compatible with the old CSV-era schema where possible
    if "ts" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    return df
