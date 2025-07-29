import pandas as pd
import os
from datetime import datetime

LOG_FILE = "data/anomaly_log.csv"

def save_anomaly_log(region, product_id, orders, inventory, revenue, explanation):
    data = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "region": region,
        "product_id": product_id,
        "orders": orders,
        "inventory": inventory,
        "revenue": revenue,
        "explanation": explanation
    }
    df = pd.DataFrame([data])
    df.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))

def read_anomaly_log(n=20):
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(LOG_FILE)
    return df.tail(n)
