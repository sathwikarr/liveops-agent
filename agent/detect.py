# agent/detect.py
import pandas as pd
import numpy as np
from typing import List, Optional

REQUIRED_COLS = ["timestamp", "region", "product_id", "orders", "inventory", "revenue"]

def read_latest_data(
    csv_file: str = "data/live_stream.csv",
    n: int = 500,
    has_header: Optional[bool] = None,
    expected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read last n rows from a CSV robustly.
    - If has_header is None: try to auto-detect header vs no-header.
    - Coerces numeric columns and keeps only expected columns if provided.
    """
    # First attempt: assume headers are present
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

    # If caller forced header behavior, override when needed
    if has_header is False and read_as_header:
        df = pd.read_csv(
            csv_file,
            names=expected_columns or REQUIRED_COLS,
            header=None,
            on_bad_lines="skip",
            engine="python",
        )

    # If expected_columns provided, select/rename as needed
    if expected_columns:
        # Try to align case-insensitively
        lower_map = {c.lower(): c for c in df.columns}
        aligned = []
        for need in expected_columns:
            c = lower_map.get(need.lower())
            if c is None:
                # create missing column if absent
                df[need] = pd.NA
                aligned.append(need)
            else:
                if c != need:
                    df.rename(columns={c: need}, inplace=True)
                aligned.append(need)
        df = df[aligned]

    # Coerce numeric columns
    for col in ["orders", "inventory", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only last n non-empty rows
    df = df.tail(max(n, 1)).reset_index(drop=True)
    return df


def zscore_anomaly_detection(
    df: pd.DataFrame,
    column: str,
    threshold: float = 2.5,
    min_rows: int = 20,
) -> pd.DataFrame:
    """
    Return rows where |z| > threshold for the given numeric column.
    - Does not mutate the original df.
    - Handles small samples and zero-std safely.
    """
    if column not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["z_score"])

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < max(min_rows, 3):
        # Not enough data to judge anomalies
        return pd.DataFrame(columns=list(df.columns) + ["z_score"])

    mean = series.mean()
    std = series.std(ddof=0)  # population std

    if std == 0 or np.isnan(std):
        # No variance -> no anomalies by z-score
        return pd.DataFrame(columns=list(df.columns) + ["z_score"])

    # Compute z on a copy to avoid mutating df
    out = df.copy()
    out["__value__"] = pd.to_numeric(out[column], errors="coerce")
    out["z_score"] = (out["__value__"] - mean) / std
    anomalies = out[out["z_score"].abs() > threshold].drop(columns="__value__", errors="ignore")

    # Keep useful columns (include z_score)
    keep_cols = [c for c in ["timestamp", "region", "product_id", column, "z_score"] if c in anomalies.columns]
    if keep_cols:
        anomalies = anomalies[keep_cols]

    # Sort by biggest deviation first
    return anomalies.sort_values(by="z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = read_latest_data()
    print("\n🔍 Revenue Anomalies:")
    print(zscore_anomaly_detection(df, "revenue"))

    print("\n🔍 Inventory Anomalies:")
    print(zscore_anomaly_detection(df, "inventory"))
