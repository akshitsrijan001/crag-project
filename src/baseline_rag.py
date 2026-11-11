"""Baseline RAG: simple retrieve-then-generate pipeline."""
import time
from src.retriever import retrieve
from src.llm import generate

def run_baseline(query: str) -> dict:
    """
    Plain RAG pipeline: retrieve top-k chunks, generate answer.
    Returns: {answer, chunks, latency_seconds}
    """
    start = time.time()

    # Retrieve relevant chunks
    chunks = retrieve(query)

    # Build context from chunks
    context = "\n\n".join(
        f"[Source: {c['source']}] {c['text']}" for c in chunks
    )

    # Generate answer
    prompt = f"""Answer the following question using ONLY the provided context. If the context does not contain enough information to answer the question, say "I don't have enough information to answer this question." Keep your answer concise (under 150 words).

Context:
{context}

Question: {query}

Answer:"""

    answer = generate(prompt)
    latency = time.time() - start

    return {
        "answer": answer,
        "chunks": chunks,
        "latency_seconds": round(latency, 2),
    }
