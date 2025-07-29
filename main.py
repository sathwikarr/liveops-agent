from agent.detect import read_latest_data, zscore_anomaly_detection
# from agent.explain import explain_anomaly  # Uncomment when using real API

import datetime
import os

def mock_explain(region, product_id, orders, inventory, revenue):
    return f"""🤖 Mocked Explanation:
1. Regional surge in demand in {region}.
2. Product {product_id} might be trending or discounted.
3. A bulk order or pricing issue may have caused the spike in revenue ({revenue})."""

def run_liveops_agent():
    print("🧠 LiveOps Agent Running...\n")

    df = read_latest_data()

    anomalies = zscore_anomaly_detection(df, "revenue")

    if anomalies.empty:
        print("✅ No anomalies found.")
        return

    for _, row in anomalies.iterrows():
        region = row["region"]
        product_id = row["product_id"]
        revenue = row["revenue"]
        orders = df[(df["product_id"] == product_id) & (df["region"] == region)]["orders"].iloc[-1]
        inventory = df[(df["product_id"] == product_id) & (df["region"] == region)]["inventory"].iloc[-1]

        # explanation = explain_anomaly(region, product_id, orders, inventory, revenue)
        explanation = mock_explain(region, product_id, orders, inventory, revenue)

        print("⚠️ Anomaly Detected:")
        print(f"Product: {product_id}, Region: {region}, Revenue: {revenue}")
        print(explanation)
        print("-" * 50)

        from agent.memory import save_anomaly_log
        save_anomaly_log(region, product_id, orders, inventory, revenue, explanation)
        
        from agent.action import simulate_action
        action = simulate_action(region, product_id, orders, inventory, revenue)
        print("📦 Action Taken:", action)


if __name__ == "__main__":
    run_liveops_agent()
