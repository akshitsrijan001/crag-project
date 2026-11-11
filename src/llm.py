"""Gemini LLM wrapper with retry and error handling."""
import time
import re
from google import genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL

# Initialize the Gemini client
_client = None

def _get_client():
    """Lazy-initialize the Gemini client."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
            raise ValueError("GEMINI_API_KEY not set. Please add it to your .env file.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

def generate(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini and return the text response. Retries on rate-limit errors with model fallback."""
    client = _get_client()
    candidate_models = [GEMINI_MODEL]
    for fallback in ["gemini-3.5-flash", "gemini-3.1-flash-lite"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_error = None
    for model_name in candidate_models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                last_error = e
                error_str = str(e)
                # If model is deprecated (404) or unavailable (503), break to next fallback model
                if "404" in error_str:
                    print(f"[LLM] Model {model_name} not found or retired, falling back...")
                    break
                # Retry on rate limit (429) or transient server errors (5xx)
                if "429" in error_str or "500" in error_str or "503" in error_str:
                    wait = (attempt + 1) * 2
                    print(f"[LLM] {model_name} error ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    break

    raise RuntimeError(f"LLM call failed after retries and fallbacks: {last_error}")
