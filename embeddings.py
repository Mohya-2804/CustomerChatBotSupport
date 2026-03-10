import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ✅ New SDK: google-genai client style
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"  # ✅ Latest Gemini embedding model
VECTOR_DIM = 3072  # gemini-embedding-001 output dimension


def load_customer_data(filepath: str = "Customerdetails.json") -> list[dict]:
    """Load customer order data from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for value in data.values():
        if isinstance(value, list):
            return value
    return [data]


def flatten_order_to_text(order: dict) -> str:
    """Convert an order dict into a readable text chunk for embedding."""
    lines = []
    for key, value in order.items():
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for a given text string."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def generate_query_embedding(query: str) -> list[float]:
    """Generate an embedding vector for a search query."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    )
    return result.embeddings[0].values


def prepare_documents(filepath: str = "Customerdetails.json") -> list[dict]:
    """
    Load orders from JSON, convert each to text, and generate its embedding.
    Returns a list of dicts with keys: id, text, embedding, metadata.
    """
    orders = load_customer_data(filepath)
    documents = []

    for idx, order in enumerate(orders):
        text = flatten_order_to_text(order)
        embedding = generate_embedding(text)
        documents.append(
            {
                "id": idx,
                "text": text,
                "embedding": embedding,
                "metadata": order,
            }
        )
        print(f"[embeddings] Processed order {idx + 1}/{len(orders)}: {order.get('order_id', idx)}")

    return documents


if __name__ == "__main__":
    docs = prepare_documents()
    print(f"\nTotal documents embedded: {len(docs)}")
    print(f"Embedding dimension: {len(docs[0]['embedding'])}")