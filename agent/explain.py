# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import streamlit as st

# # Load .env values
# load_dotenv()
# client = OpenAI(
#     api_key=st.secrets["OPENAI_API_KEY"],
#     base_url=st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
# )
# def explain_anomaly(region, product_id, orders, inventory, revenue):
#     prompt = f"""
# You're an AI operations analyst.

# An anomaly was detected:
# - Region: {region}
# - Product: {product_id}
# - Orders: {orders}
# - Inventory: {inventory}
# - Revenue: {revenue}

# Give 3 possible reasons this anomaly occurred. Be concise and use bullet points.
# """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",  # Change to a model your API supports
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.5
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"⚠️ LLM error: {e}"

import os
from openai import OpenAI

# Create OpenAI client (API key from environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_anomaly(product_id, region, value):
    """
    Generates an explanation for a detected anomaly using OpenAI's GPT model.
    """
    prompt = (
        f"An anomaly was detected in product {product_id} "
        f"in the {region} region with a value of {value}. "
        f"Explain possible reasons for this anomaly."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can change to gpt-4o for higher quality
        messages=[
            {"role": "system", "content": "You are a data analyst that explains anomalies clearly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].message.content
