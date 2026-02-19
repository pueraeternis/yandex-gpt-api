import sys
from pathlib import Path

import numpy as np

# Fix path to import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from src.clients.wrapper import get_openai_client
from src.config import config, logger


def get_embedding(client: OpenAI, text: str, model_uri: str) -> np.ndarray:
    """
    Fetch vector embedding for a text string.
    """
    # Replace newlines with spaces for better embedding quality (recommended practice)
    text = text.replace("\n", " ")

    response = client.embeddings.create(
        input=[text],
        model=model_uri,
        encoding_format="float",
    )
    return np.array(response.data[0].embedding)


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute Cosine Similarity manually using NumPy.
    Formula: (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def run_search_demo():
    client = get_openai_client()

    # 1. Our "Knowledge Base" (Documents)
    documents = [
        "Python is a high-level, interpreted programming language known for its readability.",
        "NVIDIA A100 is a powerful GPU designed for AI and High Performance Computing.",
        "Alexander Pushkin was a Russian poet, playwright, and novelist of the Romantic era.",
        "Docker allows you to package applications into containers for easy deployment.",
    ]

    print(f"--- Semantic Search Demo (Docs: {len(documents)}) ---")

    # 2. Vectorize Documents (This usually happens once and is stored in a Vector DB)
    logger.info("Generating embeddings for documents...")
    doc_embeddings = []

    for doc in documents:
        # Note: Use 'text-search-doc' for documents
        emb = get_embedding(client, doc, config.embedding_doc_uri)
        doc_embeddings.append(emb)

    # 3. User Query
    query = "Tell me about graphics cards for Deep Learning"
    print(f"\nQuery: '{query}'")

    # 4. Vectorize Query
    # Note: Use 'text-search-query' for the search term
    query_embedding = get_embedding(client, query, config.embedding_query_uri)

    # 5. Find the best match
    scores = []
    for doc_emb in doc_embeddings:
        score = compute_cosine_similarity(query_embedding, doc_emb)
        scores.append(score)

    # Get index of the highest score
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_doc = documents[best_idx]

    print("\n>>> Best Match:")
    print(f"Score: {best_score:.4f}")
    print(f"Text:  {best_doc}")

    # Debug: Show all scores
    print("\n(All scores:)")
    for i, score in enumerate(scores):
        print(f"[{score:.4f}] {documents[i][:40]}...")


if __name__ == "__main__":
    run_search_demo()
