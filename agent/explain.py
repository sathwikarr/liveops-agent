import os
import openai
from dotenv import load_dotenv

# Load .env values
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def explain_anomaly(region, product_id, orders, inventory, revenue):
    prompt = f"""
You're an AI operations analyst.

An anomaly was detected:
- Region: {region}
- Product: {product_id}
- Orders: {orders}
- Inventory: {inventory}
- Revenue: {revenue}

Give 3 possible reasons this anomaly occurred. Be concise and use bullet points.
"""

    try:
        response = openai.ChatCompletion.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Or try llama-3
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM error: {e}"
