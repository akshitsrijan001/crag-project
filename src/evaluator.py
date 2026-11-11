"""Retrieval Evaluator: scores retrieved chunks for relevance to the query."""
import json
import re
from src.llm import generate
from src.config import UPPER_THRESHOLD, LOWER_THRESHOLD

def evaluate_retrieval(query: str, chunks: list[dict],
                       upper: float = UPPER_THRESHOLD,
                       lower: float = LOWER_THRESHOLD) -> dict:
    """
    Evaluate retrieved chunks for relevance to the query.
    Uses a single batched LLM call for efficiency.
    Returns: {verdict, scores, reasons, kept_chunks}
    """
    if not chunks:
        return {
            "verdict": "INCORRECT",
            "scores": [],
            "reasons": [],
            "kept_chunks": [],
        }

    # Build a batched prompt for all chunks
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n--- Chunk {i+1} ---\n{chunk['text']}\n"

    prompt = f"""You are a retrieval evaluator. Given a user query and {len(chunks)} retrieved text chunks, score how relevant EACH chunk is to answering the query.

User Query: {query}

Retrieved Chunks:
{chunks_text}

Return ONLY a JSON array with exactly {len(chunks)} objects, one per chunk in order.
Each object must have:
- "score": a float from 0.0 to 1.0 (1.0 = perfectly relevant, 0.0 = completely irrelevant)
- "reason": a single short sentence explaining the score

Example output format:
[{{"score": 0.8, "reason": "Directly discusses the topic"}}, {{"score": 0.2, "reason": "Unrelated content"}}]

Return ONLY the JSON array, no other text."""

    raw = generate(prompt)

    # Parse the JSON response robustly
    scores = []
    reasons = []
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r'```(?:json)?\s*', '', raw)
        cleaned = cleaned.strip().rstrip('`')
        parsed = json.loads(cleaned)

        if isinstance(parsed, list):
            for item in parsed:
                scores.append(float(item.get("score", 0.0)))
                reasons.append(str(item.get("reason", "")))
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: try to extract scores with regex
        score_matches = re.findall(r'"score"\s*:\s*([\d.]+)', raw)
        reason_matches = re.findall(r'"reason"\s*:\s*"([^"]*)"', raw)
        for s in score_matches:
            try:
                scores.append(float(s))
            except ValueError:
                scores.append(0.0)
        reasons = reason_matches if reason_matches else ["" for _ in scores]

    # Pad if we got fewer results than chunks
    while len(scores) < len(chunks):
        scores.append(0.0)
        reasons.append("Could not evaluate")

    # Truncate if we got more
    scores = scores[:len(chunks)]
    reasons = reasons[:len(chunks)]

    # Determine verdict based on max score
    max_score = max(scores) if scores else 0.0

    if max_score >= upper:
        verdict = "CORRECT"
    elif max_score < lower:
        verdict = "INCORRECT"
    else:
        verdict = "AMBIGUOUS"

    # Keep chunks that score above the lower threshold
    kept_chunks = []
    for i, chunk in enumerate(chunks):
        if scores[i] >= lower:
            kept_chunks.append(chunk)

    return {
        "verdict": verdict,
        "scores": scores,
        "reasons": reasons,
        "kept_chunks": kept_chunks,
    }
