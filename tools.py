from __future__ import annotations
import json
from duckduckgo_search import DDGS

# Module-level document store — set via set_document_store() from app startup.
_doc_store = None


def set_document_store(store) -> None:
    global _doc_store
    _doc_store = store


# ------------------------------------------------------------------
# Tool schemas
# ------------------------------------------------------------------

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information on a topic. "
            "Use this when you need factual grounding, recent data, or evidence to support your argument."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query."
                }
            },
            "required": ["query"]
        }
    }
}

_RAG_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Search the local document corpus for relevant passages. "
            "Use this to retrieve evidence from documents uploaded for this debate "
            "before falling back to web search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused retrieval query."
                }
            },
            "required": ["query"]
        }
    }
}

# Legacy constant kept for backward compatibility.
TOOLS = [_WEB_SEARCH_TOOL]


def get_tools(use_rag: bool = False) -> list[dict]:
    """Return the tool list for an agent, optionally including RAG search."""
    tools = [_WEB_SEARCH_TOOL]
    if use_rag:
        tools.append(_RAG_SEARCH_TOOL)
    return tools


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------

def web_search(query: str, max_results: int = 3) -> str:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('href', 'N/A')}\n"
                    f"Summary: {r.get('body', 'N/A')}"
                )
    except Exception as e:
        return f"Search failed: {e}"

    return "\n\n---\n\n".join(results) if results else "No results found."


def rag_search(query: str, k: int = 3) -> str:
    if _doc_store is None or _doc_store.is_empty:
        return "RAG corpus is empty. No documents have been loaded."
    results = _doc_store.search(query, k=k)
    if not results:
        return "No relevant documents found in the local corpus."
    parts = []
    for r in results:
        source = r["metadata"].get("filename", "uploaded document")
        parts.append(f"[Source: {source} | Relevance {r['score']:.2f}]\n{r['chunk']}")
    return "\n\n---\n\n".join(parts)


def dispatch_tool(name: str, arguments: str) -> str:
    args = json.loads(arguments)
    if name == "web_search":
        return web_search(args["query"])
    if name == "rag_search":
        return rag_search(args["query"])
    return f"Unknown tool: {name}"
