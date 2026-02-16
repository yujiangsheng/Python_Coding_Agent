"""
external_memory.py — External knowledge retrieval (web search, docs).

Provides the agent with access to information beyond its training data
by querying search engines and fetching web content.
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ExternalMemory:
    """Retrieves external knowledge via web search and document fetching."""

    def __init__(self, config: dict):
        self.enabled: bool = config.get("enabled", True)
        self.max_results: int = config.get("max_results", 5)
        self.search_engine: str = config.get("search_engine", "duckduckgo")
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search the web for information relevant to a query.

        Returns a list of {title, url, snippet} dicts.
        """
        if not self.enabled:
            return []

        max_results = max_results or self.max_results

        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self._cache:
            logger.debug(f"External memory cache hit: '{query}'")
            return self._cache[cache_key]

        results = self._search_duckduckgo(query, max_results)
        self._cache[cache_key] = results
        return results

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Use duckduckgo_search library."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            results = []
            for item in raw:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", item.get("link", "")),
                    "snippet": item.get("body", item.get("snippet", "")),
                })
            logger.info(f"External search: '{query}' → {len(results)} results")
            return results
        except ImportError:
            logger.warning("duckduckgo_search not available, trying fallback")
            return self._search_fallback(query, max_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _search_fallback(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Minimal fallback using requests + DuckDuckGo HTML (best effort)."""
        try:
            import requests
            url = "https://html.duckduckgo.com/html/"
            resp = requests.post(url, data={"q": query}, timeout=10,
                                 headers={"User-Agent": "Mozilla/5.0"})
            results = []
            # Simple regex extraction
            for match in re.finditer(
                r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>',
                resp.text,
            ):
                href, title = match.groups()
                title = re.sub(r"<.*?>", "", title)
                results.append({"title": title, "url": href, "snippet": ""})
                if len(results) >= max_results:
                    break
            return results
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    def search_and_summarize(self, query: str) -> str:
        """Search and return a combined text summary of results."""
        results = self.search(query)
        if not results:
            return "No external results found."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}")
        return "\n\n".join(parts)

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "search_engine": self.search_engine,
            "cached_queries": len(self._cache),
        }
