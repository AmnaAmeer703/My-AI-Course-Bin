import google.generativeai as genai

genai.configure(api_key="AIzaSyB22thoVuvbCJh06UDuc9v7IXjaP4OEDH4")

model = genai.GenerativeModel("gemini-2.5-flash")


def ask_inventory_bot(question, inventory, source):

    if inventory:
        items = "\n".join(
            f"{k}: {v}" for k, v in inventory.items()
        )
    else:
        items = "No inventory detected."

    prompt = f"""
You are a Warehouse Inventory AI Assistant.

Detection Source:
{source}

Current Inventory:
{items}

Total Inventory:
{sum(inventory.values())}

Question:
{question}
"""

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Gemini Error:\n\n{e}"