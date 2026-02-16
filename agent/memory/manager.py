"""
manager.py — Unified memory management with RAG pipeline.

Orchestrates all four memory tiers and provides the agent with a single
API for storing, recalling, and maintaining knowledge.

Enhanced with:
  - **RAG pipeline**: retrieve → rerank → filter → augment context.
  - **Auto search fallback**: when internal recall is insufficient,
    automatically queries external search and folds results in.
  - **Experience replay**: convenience wrappers for storing & recalling
    structured (task, solution, outcome) triplets.
  - **Consolidation**: periodically merges near-duplicate long-term
    memories to keep the store lean.
"""

import logging
from typing import Any, Dict, List, Optional

from agent.memory.working_memory import WorkingMemory
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.persistent_memory import PersistentMemory
from agent.memory.external_memory import ExternalMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified gateway to all four memory tiers with a RAG pipeline.

    Tiers:
      1. Working Memory   — in-session conversation & scratch
      2. Long-Term Memory — vector-indexed cross-session knowledge
         (surprise gating + dedup + relevance decay)
      3. Persistent Memory — structured knowledge base on disk
         (experience replay + semantic dedup)
      4. External Memory   — web search / documentation retrieval
         (auto-fallback when internal recall is weak)
    """

    # Minimum number of *useful* internal recall results before we
    # consider the recall "sufficient" and skip external search.
    AUTO_SEARCH_MIN_RESULTS: int = 2
    # Minimum average similarity of long-term recall results to
    # consider them "confident enough" to skip external search.
    AUTO_SEARCH_MIN_SIM: float = 0.55

    def __init__(self, config: dict):
        self.working = WorkingMemory(config.get("working", {}))
        self.long_term = LongTermMemory(config.get("long_term", {}))
        self.persistent = PersistentMemory(config.get("persistent", {}))
        self.external = ExternalMemory(config.get("external", {}))

        self.auto_search: bool = config.get("auto_search", True)
        logger.info("MemoryManager initialised with 4 tiers + RAG pipeline")

    # ==================================================================
    # RAG Pipeline: retrieve → rerank → filter
    # ==================================================================

    def rag_recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_external: bool = False,
        auto_search_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Full RAG recall across all internal tiers, with optional
        auto-search fallback when results are weak.

        Returns a dict keyed by tier name with ranked results.
        """
        results: Dict[str, Any] = {}

        # --- Stage 1: Retrieve from internal tiers ---
        # Long-term (vector-indexed, with relevance decay)
        lt_results = self.long_term.recall(query, top_k=top_k, use_decay=True)
        results["long_term"] = lt_results

        # Persistent (keyword + category search)
        ps_results = self.persistent.recall(query=query, limit=top_k)
        results["persistent"] = ps_results

        # Experience replay (successful past solutions)
        exp_results = self.persistent.recall_experiences(
            query, success_only=True, limit=3
        )
        results["experiences"] = exp_results

        # Working memory (recent turns keyword match)
        working_hits = self._search_working(query, top_k)
        results["working"] = working_hits

        # --- Stage 2: Rerank / score ---
        # Already ranked by similarity within each tier; we do a simple
        # cross-tier relevance check for the auto-search decision.
        total_internal = (
            len(lt_results) + len(ps_results) + len(exp_results)
        )
        avg_sim = 0.0
        if lt_results:
            avg_sim = sum(r["similarity"] for r in lt_results) / len(lt_results)

        # --- Stage 3: Auto search fallback ---
        if (
            auto_search_fallback
            and self.auto_search
            and (
                total_internal < self.AUTO_SEARCH_MIN_RESULTS
                or avg_sim < self.AUTO_SEARCH_MIN_SIM
            )
        ):
            logger.info(
                f"RAG auto-search: internal recall weak "
                f"(hits={total_internal}, avg_sim={avg_sim:.3f}), "
                f"querying external search"
            )
            ext_results = self.external.search(query, max_results=top_k)
            results["external"] = ext_results
        elif include_external:
            results["external"] = self.external.search(query, max_results=top_k)

        return results

    # ------------------------------------------------------------------
    # Simple recall (backward-compatible)
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        tiers: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Recall relevant information across specified tiers.

        This is the simple/legacy interface. For the full RAG pipeline use
        :meth:`rag_recall`.
        """
        tiers = tiers or ["working", "long_term", "persistent"]
        results: Dict[str, Any] = {}

        if "working" in tiers:
            results["working"] = self._search_working(query, top_k)
        if "long_term" in tiers:
            results["long_term"] = self.long_term.recall(query, top_k=top_k)
        if "persistent" in tiers:
            results["persistent"] = self.persistent.recall(query=query, limit=top_k)
        if "external" in tiers:
            results["external"] = self.external.search(query, max_results=top_k)

        return results

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        category: str = "general",
        key: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Store information in both long-term and persistent memory.

        Deduplication is handled transparently at both tiers:
        - Long-term: surprise gating rejects near-duplicates.
        - Persistent: trigram similarity rejects near-duplicates.
        """
        # Long-term (vector-indexed, surprise gated)
        self.long_term.store(text, metadata=metadata)

        # Persistent (structured)
        store_key = key or text[:80].replace("\n", " ")
        self.persistent.store(
            category=category,
            key=store_key,
            value=text,
            metadata=metadata,
        )

    def add_conversation_turn(
        self, role: str, content: str, metadata: Optional[dict] = None
    ):
        """Add a turn to working memory."""
        self.working.add_turn(role, content, metadata)

    def search_external(self, query: str) -> str:
        """Search the web and return a text summary."""
        return self.external.search_and_summarize(query)

    # ==================================================================
    # Context assembly (RAG-augmented)
    # ==================================================================

    def get_context_messages(
        self,
        system_prompt: str = "",
        relevant_query: Optional[str] = None,
        *,
        use_rag: bool = True,
    ) -> List[Dict[str, str]]:
        """Build context messages enriched with recalled memories.

        When *use_rag* is True and a *relevant_query* is given, the full
        RAG pipeline is used (retrieve → rerank → auto-search fallback).
        """
        messages: List[Dict[str, str]] = []

        # System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Inject recalled memories as context
        if relevant_query:
            if use_rag:
                recalled = self.rag_recall(
                    relevant_query,
                    top_k=5,
                    auto_search_fallback=True,
                )
            else:
                recalled = self.recall(
                    relevant_query,
                    tiers=["long_term", "persistent"],
                    top_k=3,
                )

            context_parts = self._format_recalled(recalled)
            if context_parts:
                messages.append(
                    {
                        "role": "system",
                        "content": "[Retrieved Memories]\n" + "\n".join(context_parts),
                    }
                )

        # Scratchpad
        if self.working.scratchpad:
            scratch = self.working._summarize_scratchpad()
            messages.append({"role": "system", "content": f"[Scratchpad]\n{scratch}"})

        # Conversation history
        messages.extend(self.working.get_turns())
        return messages

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_all(self):
        """Persist all memory tiers to disk."""
        self.long_term.save_index()
        self.persistent.save()
        logger.info("All memory tiers saved")

    def summary(self) -> dict:
        return {
            "working": self.working.summary(),
            "long_term": self.long_term.summary(),
            "persistent": self.persistent.summary(),
            "external": self.external.summary(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _search_working(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword search across recent working memory turns."""
        working_results = []
        q_lower = query.lower()
        for turn in self.working.get_full_turns():
            if q_lower in turn["content"].lower():
                working_results.append(turn)
        return working_results[-top_k:]

    @staticmethod
    def _format_recalled(recalled: Dict[str, Any]) -> List[str]:
        """Render recalled memories into text lines for LLM context."""
        parts: List[str] = []

        for mem in recalled.get("long_term", []):
            parts.append(f"[Memory] {mem['text']}")

        for mem in recalled.get("persistent", []):
            val = mem.get("value", mem.get("key", ""))
            cat = mem.get("category", "?")
            parts.append(f"[Knowledge:{cat}] {val}")

        for exp in recalled.get("experiences", []):
            val = exp.get("value", {})
            if isinstance(val, dict):
                parts.append(
                    f"[Experience] Task: {val.get('task', '?')} → "
                    f"Solution: {val.get('solution', '?')[:300]}"
                )

        for ext in recalled.get("external", []):
            title = ext.get("title", "")
            snippet = ext.get("snippet", "")
            parts.append(f"[Web] {title}: {snippet}")

        return parts
