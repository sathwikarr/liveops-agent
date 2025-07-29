import random
import time
import pandas as pd
from datetime import datetime

REGIONS = ["North", "South", "East", "West"]
PRODUCT_IDS = [f"P{100+i}" for i in range(10)]

def generate_data():
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    region = random.choice(REGIONS)
    product_id = random.choice(PRODUCT_IDS)
    orders = random.randint(5, 20)
    inventory = random.randint(10, 100)
    revenue = round(orders * random.uniform(20.0, 60.0), 2)

    return {
        "timestamp": timestamp,
        "region": region,
        "product_id": product_id,
        "orders": orders,
        "inventory": inventory,
        "revenue": revenue
    }

def simulate_stream(to_csv=False, delay=1):
    while True:
        data = generate_data()
        print(data)

        if to_csv:
            df = pd.DataFrame([data])
            df.to_csv("data/live_stream.csv", mode='a', index=False, header=False)

        time.sleep(delay)

if __name__ == "__main__":
    simulate_stream(to_csv=True)
