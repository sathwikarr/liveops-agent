import pandas as pd
import numpy as np

def read_latest_data(csv_file="data/live_stream.csv", n=100):
    """Reads the last n rows from the data stream"""
    df = pd.read_csv(csv_file, names=["timestamp", "region", "product_id", "orders", "inventory", "revenue"])
    return df.tail(n)

def zscore_anomaly_detection(df, column, threshold=2.5):
    """Returns rows where value is more than `threshold` std dev from mean"""
    values = df[column].astype(float)
    mean = np.mean(values)
    std = np.std(values)

    df["z_score"] = (values - mean) / std
    anomalies = df[np.abs(df["z_score"]) > threshold]
    return anomalies[["timestamp", "region", "product_id", column, "z_score"]]

if __name__ == "__main__":
    df = read_latest_data()
    print("\n🔍 Revenue Anomalies:")
    print(zscore_anomaly_detection(df, "revenue"))

    print("\n🔍 Inventory Anomalies:")
    print(zscore_anomaly_detection(df, "inventory"))
