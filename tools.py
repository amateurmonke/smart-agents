import json
from duckduckgo_search import DDGS

TOOLS = [
    {
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
]


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


def dispatch_tool(name: str, arguments: str) -> str:
    args = json.loads(arguments)
    if name == "web_search":
        return web_search(args["query"])
    return f"Unknown tool: {name}"
