"""Retriever: embed query, compute cosine similarity, return top-k chunks."""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import INDEX_PATH, CHUNKS_PATH, TOP_K, EMBEDDING_MODEL

# Cache the model at module level
_model = None

def _get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query, compute cosine similarity with all chunks,
    return top-k chunks with their similarity scores.
    """
    # Load index and chunks
    data = np.load(INDEX_PATH)
    embeddings = data["embeddings"]
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Embed the query
    model = _get_model()
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    # Cosine similarity (embeddings are already normalized)
    similarities = embeddings @ query_emb

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(similarities[idx])
        results.append(chunk)

    return results
