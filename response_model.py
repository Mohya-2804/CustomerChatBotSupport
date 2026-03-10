import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ✅ New SDK client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

GEMINI_MODEL = "gemini-2.5-pro"  # ✅ Latest available model

SYSTEM_PROMPT = """You are a helpful and friendly customer support assistant for an e-commerce platform.
You answer customer queries based ONLY on the order information provided in the context below.

Guidelines:
- Be concise, polite, and professional.
- If the context does not contain enough information to answer, say so honestly.
- Do NOT make up order details, prices, dates, or tracking numbers.
- When referencing order information, quote the exact values from the context.
- If a customer asks about multiple orders, address each one separately.
"""


def build_context_from_hits(hits: list[dict]) -> str:
    """Format Qdrant search results into a readable context block."""
    if not hits:
        return "No relevant order information found."

    sections = []
    for i, hit in enumerate(hits, 1):
        payload = hit["payload"]
        text = payload.get("text", "")
        score = hit.get("score", 0)
        sections.append(f"--- Order Record {i} (relevance: {score:.2f}) ---\n{text}")

    return "\n\n".join(sections)


def generate_response(query: str, hits: list[dict], chat_history: list[dict] | None = None) -> str:
    """
    Generate a Gemini response for the user query using the retrieved context.
    """
    context = build_context_from_hits(hits)

    # Build full message with context + question
    user_message = (
        f"Context (retrieved order data):\n{context}\n\n"
        f"Customer question: {query}"
    )

    # Build conversation history for multi-turn
    history = []
    if chat_history:
        for turn in chat_history:
            role = turn["role"]  # "user" or "model"
            text = turn["parts"][0] if isinstance(turn["parts"][0], str) else ""
            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=text)]
                )
            )

    # Add current user message
    history.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.3,
        ),
    )

    return response.text


def get_model_info() -> str:
    return GEMINI_MODEL


if __name__ == "__main__":
    sample_hits = [
        {
            "score": 0.92,
            "payload": {
                "text": (
                    "Order Id: ORD1010\nPlatform: Amazon\nProduct Name: Power Bank\n"
                    "Brand: Mi\nFinal Price: 1149\nCurrency: INR\n"
                    "Order Status: Processing\nTracking Number: TRK90010\n"
                    "Payment Status: Paid"
                ),
                "order_id": "ORD1010",
            },
        }
    ]

    answer = generate_response("Where is my order ORD1010?", sample_hits)
    print("Response:\n", answer)