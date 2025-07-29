import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_anomaly(region, product_id, orders, inventory, revenue):
    return (
        f"1. There was a regional surge in demand in {region}.\n"
        f"2. Product {product_id} might be trending or part of a promotion.\n"
        f"3. A bulk purchase or unexpected price shift may have caused the spike in revenue ({revenue})."
    )

# Example usage
if __name__ == "__main__":
    result = explain_anomaly("East", "P102", 18, 25, 1093.99)
    print("🤖 Mocked Explanation:\n", result)
