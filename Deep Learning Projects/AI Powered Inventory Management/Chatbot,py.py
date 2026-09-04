import os
from google import genai
from PIL import Image

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def ask_inventory_bot(question, inventory):

    inventory_text = "\n".join(
        f"{k}: {v}" for k, v in inventory.items()
    )

    prompt = f"""
You are an AI Retail Inventory Assistant.

Current Inventory

{inventory_text}

Answer the user's question based on the inventory.
If the question is unrelated, answer normally.
Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

from PIL import Image

def analyze_image(image):

    prompt = """
You are an inventory management expert.

Analyze this warehouse image.

Return:

- Estimated inventory percentage
- Stock condition
- Empty shelves
- Overstocked areas
- Recommendations

Keep the response concise.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            image
        ]
    )

    return response.text

# =============================