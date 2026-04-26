"""Anomaly detection for LiveOps.

- `read_latest_data`: robust CSV loader (handles header / no-header).
- `zscore_anomaly_detection`: legacy single-column, single-population z-score.
- `detect_anomalies`: NEW. Per-segment z-score across multiple metrics.
   Falls back to global when a segment has too few samples.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

REQUIRED_COLS = ["timestamp", "region", "product_id", "orders", "inventory", "revenue"]
DEFAULT_METRICS: tuple[str, ...] = ("revenue", "orders", "inventory")


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #

def read_latest_data(
    csv_file: str = "data/live_stream.csv",
    n: int = 500,
    has_header: Optional[bool] = None,
    expected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read last `n` rows from a CSV robustly."""
    try:
        df = pd.read_csv(csv_file, on_bad_lines="skip", engine="python")
        read_as_header = True
    except Exception:
        df = pd.read_csv(
            csv_file,
            names=expected_columns or REQUIRED_COLS,
            header=None,
            on_bad_lines="skip",
            engine="python",
        )
        read_as_header = False

    if has_header is False and read_as_header:
        df = pd.read_csv(
            csv_file,
            names=expected_columns or REQUIRED_COLS,
            header=None,
            on_bad_lines="skip",
            engine="python",
        )

    if expected_columns:
        lower_map = {c.lower(): c for c in df.columns}
        aligned = []
        for need in expected_columns:
            c = lower_map.get(need.lower())
            if c is None:
                df[need] = pd.NA
                aligned.append(need)
            else:
                if c != need:
                    df.rename(columns={c: need}, inplace=True)
                aligned.append(need)
        df = df[aligned]

    for col in ["orders", "inventory", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.tail(max(n, 1)).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _zscore_series(values: pd.Series) -> pd.Series:
    """Population z-score; returns NaNs when std is 0 or sample is too small."""
    series = pd.to_numeric(values, errors="coerce")
    if series.dropna().size < 3:
        return pd.Series(np.nan, index=series.index)
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / std


def zscore_anomaly_detection(
    df: pd.DataFrame,
    column: str,
    threshold: float = 2.5,
    min_rows: int = 20,
) -> pd.DataFrame:
    """Legacy: global z-score on a single column. Kept for backward compat."""
    if column not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["z_score"])

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < max(min_rows, 3):
        return pd.DataFrame(columns=list(df.columns) + ["z_score"])

    z = _zscore_series(df[column])
    if z.dropna().empty:
        return pd.DataFrame(columns=list(df.columns) + ["z_score"])

    out = df.copy()
    out["z_score"] = z
    anomalies = out[out["z_score"].abs() > threshold]

    keep_cols = [c for c in ["timestamp", "region", "product_id", column, "z_score"] if c in anomalies.columns]
    if keep_cols:
        anomalies = anomalies[keep_cols]
    return anomalies.sort_values(by="z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def detect_anomalies(
    df: pd.DataFrame,
    metrics: Iterable[str] = DEFAULT_METRICS,
    threshold: float = 2.5,
    min_segment_rows: int = 8,
    min_global_rows: int = 20,
    group_by: tuple[str, ...] = ("region", "product_id"),
) -> pd.DataFrame:
    """Per-segment, multi-metric anomaly detection.

    For each metric, z-score WITHIN each (region, product_id) segment that has
    at least `min_segment_rows` rows. Segments smaller than that fall back to
    the global z-score (so we don't lose signal on long-tail products).

    Returns a long-format DataFrame with columns:
        timestamp, region, product_id, metric, value, z_score,
        scope ('segment'|'global'), orders, inventory, revenue
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "region", "product_id", "metric", "value",
            "z_score", "scope", "orders", "inventory", "revenue",
        ])

    rows = []
    have_groups = all(c in df.columns for c in group_by)

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Per-segment pass
        seg_index_used = set()
        if have_groups:
            for keys, sub in df.groupby(list(group_by), dropna=False):
                if len(sub) < min_segment_rows:
                    continue
                z = _zscore_series(sub[metric])
                hits = sub[z.abs() > threshold].copy()
                hits["__z__"] = z[z.abs() > threshold]
                for idx, row in hits.iterrows():
                    seg_index_used.add(idx)
                    rows.append({
                        "timestamp": row.get("timestamp"),
                        "region": row.get("region"),
                        "product_id": row.get("product_id"),
                        "metric": metric,
                        "value": float(row.get(metric, np.nan)),
                        "z_score": float(row["__z__"]),
                        "scope": "segment",
                        "orders": int(row.get("orders", 0) or 0),
                        "inventory": int(row.get("inventory", 0) or 0),
                        "revenue": float(row.get("revenue", 0.0) or 0.0),
                    })

        # Global fallback for rows in small segments (or when no group cols)
        if len(df) >= min_global_rows:
            z = _zscore_series(df[metric])
            global_hits = df[z.abs() > threshold]
            for idx, row in global_hits.iterrows():
                if idx in seg_index_used:
                    continue
                rows.append({
                    "timestamp": row.get("timestamp"),
                    "region": row.get("region"),
                    "product_id": row.get("product_id"),
                    "metric": metric,
                    "value": float(row.get(metric, np.nan)),
                    "z_score": float(z.iloc[idx]) if idx in z.index else np.nan,
                    "scope": "global",
                    "orders": int(row.get("orders", 0) or 0),
                    "inventory": int(row.get("inventory", 0) or 0),
                    "revenue": float(row.get("revenue", 0.0) or 0.0),
                })

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "region", "product_id", "metric", "value",
            "z_score", "scope", "orders", "inventory", "revenue",
        ])

    out = pd.DataFrame(rows)
    return out.sort_values(by="z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = read_latest_data()
    print("\n🔍 Per-segment, multi-metric anomalies:")
    print(detect_anomalies(df).head(20))
