# agent/memory.py
from pathlib import Path
import pandas as pd
from datetime import datetime

# repo root -> <repo>/data/anomaly_log.csv
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR / "data" / "anomaly_log.csv"

def get_log_path() -> str:
    return str(LOG_FILE)

def save_anomaly_log(region, product_id, orders, inventory, revenue, explanation):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "region": region,
        "product_id": product_id,
        "orders": orders,
        "inventory": inventory,
        "revenue": revenue,
        "explanation": explanation,
    }
    df = pd.DataFrame([row])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", index=False, header=header)

def read_anomaly_log(n: int = 200) -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=["timestamp","region","product_id","orders","inventory","revenue","explanation"])
    df = pd.read_csv(LOG_FILE)
    return df.tail(n)
