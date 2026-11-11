"""Web search wrapper using DuckDuckGo (ddgs)."""

def search(query: str, n: int = 4) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    Returns list of {title, snippet, url}. Fails gracefully.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=n))
        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            })
        return results
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return []
