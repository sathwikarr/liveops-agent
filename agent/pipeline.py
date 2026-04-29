"""LiveOps agent pipeline — UI-agnostic core.

One pass = read latest CSV → detect anomalies → explain → log → act → notify.
Callers (the FastAPI /run-agent route, the standalone `auto_agent.py` CLI loop,
or any future runner) pass `log` and `on_anomaly` callbacks so this module
never imports a UI framework.

Used to live in `pages/run_agent.py` alongside the Streamlit chrome — moved
here in Phase 15 when the Streamlit pages were retired.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from agent.action import simulate_action
from agent.detect import detect_anomalies, read_latest_data
from agent.explain import explain_anomaly
from agent.memory import save_anomaly_log


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLS = {"region", "product_id", "orders", "inventory", "revenue"}

# Where per-user CSV uploads live. Override with LIVEOPS_USER_DATA so prod
# (Fly.io) can point this onto the persisted volume at /app/data/user_data.
USER_DATA_DIR = Path(os.getenv("LIVEOPS_USER_DATA", str(REPO_ROOT / "user_data")))


def _user_csv_path(username: str) -> Path:
    return USER_DATA_DIR / f"{username}.csv"


def run_pipeline(
    username: str,
    *,
    threshold: float = 1.5,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    on_anomaly: Optional[Callable[[dict, str, str], None]] = None,
) -> int:
    """Run one pass of the LiveOps pipeline. Returns the number of anomalies
    acted on. UI-agnostic — callers receive progress via `log` and per-anomaly
    detail via `on_anomaly(row_dict, explanation, action)`."""
    csv_path = _user_csv_path(username)
    if not csv_path.exists():
        log(f"⚠️ No CSV for user {username!r} at {csv_path}")
        return 0

    df = read_latest_data(str(csv_path), n=2000)
    if df is None or df.empty:
        log("ℹ️ No rows to analyze yet.")
        return 0

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        log(f"❌ CSV is missing required columns: {', '.join(sorted(missing))}")
        return 0

    anomalies = detect_anomalies(df, threshold=threshold).head(top_k)
    if anomalies.empty:
        log("✅ No anomalies found.")
        return 0

    count = 0
    for _, row in anomalies.iterrows():
        region = row["region"]
        product_id = row["product_id"]
        metric = row["metric"]
        z = float(row["z_score"])
        revenue = float(row["revenue"])
        orders = int(row["orders"])
        inventory = int(row["inventory"])

        explanation = explain_anomaly(region, product_id, orders, inventory, revenue)

        try:
            save_anomaly_log(
                region, product_id, orders, inventory, revenue, explanation,
                username=username, metric=metric, z_score=z,
            )
        except Exception as e:
            log(f"⚠️ Could not save anomaly: {e}")

        action = "ℹ️ Log only – no action needed"
        try:
            action = simulate_action(username, region, product_id, orders, inventory, revenue)
        except Exception as e:
            log(f"⚠️ Action layer failed: {e}")

        if on_anomaly:
            on_anomaly(dict(row), explanation, action)

        count += 1

    log(f"📦 Processed {count} anomalies for {username}.")
    return count


__all__ = ["run_pipeline"]
