"""CRAG: Self-Corrective Retrieval-Augmented Generation pipeline."""
import time
from src.retriever import retrieve
from src.evaluator import evaluate_retrieval
from src.corrective import refine_context, rewrite_query
from src.web_search import search as web_search
from src.llm import generate

def run_crag(query: str,
             upper_threshold: float = None,
             lower_threshold: float = None) -> dict:
    """
    Full CRAG pipeline:
    1. Retrieve chunks
    2. Evaluate retrieval quality
    3. Route based on verdict (CORRECT / AMBIGUOUS / INCORRECT)
    4. Generate answer

    Returns dict with all trace information.
    """
    from src.config import UPPER_THRESHOLD, LOWER_THRESHOLD
    upper = upper_threshold if upper_threshold is not None else UPPER_THRESHOLD
    lower = lower_threshold if lower_threshold is not None else LOWER_THRESHOLD

    start = time.time()
    rewritten_query = None
    web_results = []
    final_context = ""
    sources = []

    # Step 1: Retrieve
    chunks = retrieve(query)

    # Step 2: Evaluate
    eval_result = evaluate_retrieval(query, chunks, upper=upper, lower=lower)
    verdict = eval_result["verdict"]

    # Step 3: Route based on verdict
    if verdict == "CORRECT":
        # Refine local context, then generate
        final_context = refine_context(query, eval_result["kept_chunks"])
        sources = [c["source"] for c in eval_result["kept_chunks"]]
        branch_used = "CORRECT → Refine local context → Generate"

    elif verdict == "AMBIGUOUS":
        # Refine local context + augment with web search
        refined = refine_context(query, eval_result["kept_chunks"])
        web_results = web_search(query, n=4)
        web_text = "\n\n".join(
            f"[Web: {r['title']}] {r['snippet']}" for r in web_results
        )
        final_context = f"{refined}\n\n--- Web Search Results ---\n{web_text}" if web_text else refined
        sources = [c["source"] for c in eval_result["kept_chunks"]] + [r["url"] for r in web_results]
        branch_used = "AMBIGUOUS → Refine + Web search → Generate"

    else:  # INCORRECT
        # Discard local context, rewrite query, search web
        rewritten_query = rewrite_query(query)
        web_results = web_search(rewritten_query, n=4)
        if web_results:
            final_context = "\n\n".join(
                f"[Web: {r['title']}] {r['snippet']}" for r in web_results
            )
        else:
            final_context = "No relevant information could be found from web search."
        sources = [r["url"] for r in web_results]
        branch_used = "INCORRECT → Rewrite query → Web search → Generate"

    # Step 4: Generate answer
    gen_prompt = f"""Answer the following question using ONLY the provided context. If the context does not contain enough information to answer the question, say "I don't have enough information to answer this question." Keep your answer concise (under 150 words).

Context:
{final_context}

Question: {query}

Answer:"""

    answer = generate(gen_prompt)
    latency = time.time() - start

    return {
        "answer": answer,
        "verdict": verdict,
        "scores": eval_result["scores"],
        "reasons": eval_result["reasons"],
        "rewritten_query": rewritten_query,
        "local_chunks": chunks,
        "web_results": web_results,
        "final_context": final_context,
        "latency_seconds": round(latency, 2),
        "branch_used": branch_used,
        "sources": sources,
    }
