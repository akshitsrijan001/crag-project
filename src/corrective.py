"""Corrective actions: refine context and rewrite queries."""
from src.llm import generate

def refine_context(query: str, chunks: list[dict]) -> str:
    """
    Extract only the sentences relevant to the query from the given chunks.
    Acts as a 'knowledge strip' filter to reduce noise.
    """
    if not chunks:
        return ""

    combined = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""You are a knowledge extraction assistant. Given a user query and some text passages, extract ONLY the sentences that are directly relevant to answering the query. Remove all irrelevant information.

User Query: {query}

Text Passages:
{combined}

Return ONLY the relevant sentences as a clean, condensed paragraph. If nothing is relevant, return "No relevant information found."
"""
    return generate(prompt)

def rewrite_query(query: str) -> str:
    """
    Rewrite the user's question into a concise, keyword-rich web search query.
    Max 12 words.
    """
    prompt = f"""Rewrite the following question into a concise, keyword-rich web search query (maximum 12 words). Return ONLY the rewritten query, nothing else.

Question: {query}

Rewritten search query:"""
    result = generate(prompt)
    # Clean up: take first line, strip quotes
    result = result.strip().strip('"').strip("'").split('\n')[0]
    return result
