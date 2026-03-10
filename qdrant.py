import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

load_dotenv()

COLLECTION_NAME = "customer_orders"
VECTOR_DIM = 3072  # gemini-embedding-001 output dimension
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client (local or cloud based on env vars)."""
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(":memory:")


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create (or recreate) the Qdrant collection."""
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"[qdrant] Deleted existing collection '{COLLECTION_NAME}'.")
        else:
            print(f"[qdrant] Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"[qdrant] Collection '{COLLECTION_NAME}' created (dim={VECTOR_DIM}, Cosine).")


def upsert_documents(client: QdrantClient, documents: list[dict]) -> None:
    """Insert or update document vectors in Qdrant."""
    points = []
    for doc in documents:
        payload = {"text": doc["text"], **doc["metadata"]}
        clean_payload = {k: (str(v) if v is None else v) for k, v in payload.items()}
        points.append(
            PointStruct(
                id=doc["id"],
                vector=doc["embedding"],
                payload=clean_payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[qdrant] Upserted {len(points)} points into '{COLLECTION_NAME}'.")


def search_similar(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 3,
    score_threshold: float = 0.3,
) -> list[dict]:
    """
    Perform a cosine similarity search using query_points (new qdrant-client API).
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    hits = []
    for r in results.points:
        hits.append({"score": r.score, "payload": r.payload})

    return hits


def index_documents(documents: list[dict], recreate: bool = False) -> QdrantClient:
    """
    Convenience function: create collection and upsert all documents.
    Returns the Qdrant client for subsequent queries.
    """
    client = get_qdrant_client()
    create_collection(client, recreate=recreate)
    upsert_documents(client, documents)
    return client


if __name__ == "__main__":
    from embeddings import prepare_documents, generate_query_embedding

    docs = prepare_documents()
    client = index_documents(docs, recreate=True)

    test_query = "What is the status of order ORD1010?"
    query_vec = generate_query_embedding(test_query)
    hits = search_similar(client, query_vec, top_k=2)
    print("\nTop search results:")
    for h in hits:
        print(f"  Score: {h['score']:.4f} | Order: {h['payload'].get('order_id')}")