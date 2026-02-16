"""
persistent_memory.py — Structured knowledge base persisted on disk.

Stores categorised knowledge: learned coding patterns, API references,
error resolutions, user preferences, self-improvement history, and
**experience replays** (task → solution → outcome triplets).

Enhanced with:
  - **Experience replay**: structured (task, solution, outcome) triplets
    enabling the agent to re-use proven solutions for similar future tasks.
  - **Semantic dedup**: optional text-similarity check before inserting
    to avoid near-duplicate knowledge entries.
"""

import json
import logging
import os
import time
import hashlib
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PersistentMemory:
    """Disk-backed structured knowledge store with experience replay.

    Categories organise knowledge:
      - patterns       : learned coding patterns / idioms
      - errors          : error → resolution mappings
      - preferences     : user coding style preferences
      - api_knowledge   : API / library usage notes
      - improvements    : self-improvement records
      - experiences     : (task, solution, outcome) triplets  ← NEW
      - custom          : user-defined
    """

    BUILTIN_CATEGORIES = [
        "patterns",
        "errors",
        "preferences",
        "api_knowledge",
        "improvements",
        "experiences",
        "custom",
    ]

    def __init__(self, config: dict):
        self.db_path: str = config.get("db_path", "data/memories/persistent.json")
        self.max_entries: int = config.get("max_entries", 50000)
        # Semantic dedup: skip entries whose key text is > dedup_ratio similar
        self.dedup_enabled: bool = config.get("dedup_enabled", True)
        self.dedup_ratio: float = config.get("dedup_ratio", 0.90)
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self._dirty: bool = False
        self._last_save: float = 0.0
        self._save_interval: float = 5.0  # debounce: min seconds between saves
        self._load()

    # ------------------------------------------------------------------
    # Lightweight text similarity (for dedup — no embedding model needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _trigram_set(text: str) -> set:
        """Return the character-trigram set of *text*."""
        t = text.lower().strip()
        if len(t) < 3:
            return {t}
        return {t[i : i + 3] for i in range(len(t) - 2)}

    def _text_similarity(self, a: str, b: str) -> float:
        """Jaccard similarity over character trigrams — fast & dependency-free."""
        sa, sb = self._trigram_set(a), self._trigram_set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _extract_comparable_text(key: str, value) -> str:
        """Extract meaningful text from an entry for dedup comparison.

        For dict values (e.g. experiences), extracts the task/solution
        text instead of comparing opaque hash keys.
        """
        if isinstance(value, dict):
            # Experience replay entries: compare actual task + solution text
            parts = []
            for field in ("task", "solution", "outcome", "weakness",
                          "principle", "lesson"):
                v = value.get(field, "")
                if isinstance(v, str) and v:
                    parts.append(v[:500])
            if parts:
                return " ".join(parts)
        if isinstance(value, str):
            return f"{key} {value}"
        return key

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(
        self,
        category: str,
        key: str,
        value: Any,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Store a knowledge entry under a category.

        Returns the key if stored/updated, or *None* if deduplicated away.
        """
        if category not in self.data:
            self.data[category] = []

        # Check for existing entry with same key — update in place
        for entry in self.data[category]:
            if entry.get("key") == key:
                entry["value"] = value
                entry["metadata"] = {**entry.get("metadata", {}), **(metadata or {})}
                entry["updated"] = time.time()
                entry["access_count"] = entry.get("access_count", 0) + 1
                self._auto_save()
                return key

        # --- Semantic dedup (character-trigram Jaccard) ---
        if self.dedup_enabled:
            combined_text = self._extract_comparable_text(key, value)
            for entry in self.data[category]:
                existing_text = self._extract_comparable_text(
                    entry.get("key", ""), entry.get("value")
                )
                if self._text_similarity(combined_text, existing_text) >= self.dedup_ratio:
                    # Near-duplicate: bump existing instead of inserting
                    entry["access_count"] = entry.get("access_count", 0) + 1
                    entry["updated"] = time.time()
                    self._auto_save()
                    logger.debug(
                        f"Persistent dedup: [{category}] key='{key}' "
                        f"similar to existing='{entry['key'][:60]}'"
                    )
                    return None

        # New entry
        self.data[category].append({
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "created": time.time(),
            "updated": time.time(),
            "access_count": 0,
        })

        self._enforce_limits()
        self._auto_save()
        logger.debug(f"Persistent memory: stored [{category}] {key}")
        return key

    # ------------------------------------------------------------------
    # Experience Replay (task → solution → outcome triplets)
    # ------------------------------------------------------------------

    def store_experience(
        self,
        task: str,
        solution: str,
        outcome: str,
        *,
        success: bool = True,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Store a structured experience replay entry.

        Args:
            task: Description of what was asked / the problem.
            solution: The code or approach that was used.
            outcome: What happened (success message, error, etc.).
            success: Whether the experience was successful.
            metadata: Extra metadata (intent type, etc.).
        """
        exp_key = hashlib.sha256(task.encode()).hexdigest()[:12]
        value = {
            "task": task[:500],
            "solution": solution[:2000],
            "outcome": outcome[:500],
            "success": success,
        }
        return self.store(
            category="experiences",
            key=f"exp_{exp_key}",
            value=value,
            metadata={**(metadata or {}), "success": success},
        )

    def recall_experiences(
        self,
        query: str,
        *,
        success_only: bool = True,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recall experience replay entries relevant to *query*.

        Performs a substring search on task/solution fields and optionally
        filters to successful experiences only.
        """
        results = []
        q_lower = query.lower()
        for entry in self.data.get("experiences", []):
            val = entry.get("value", {})
            if isinstance(val, dict):
                text = f"{val.get('task', '')} {val.get('solution', '')}".lower()
            else:
                text = str(val).lower()

            if q_lower not in text:
                continue

            if success_only and isinstance(val, dict) and not val.get("success", True):
                continue

            entry["access_count"] = entry.get("access_count", 0) + 1
            results.append({**entry, "category": "experiences"})

        results.sort(key=lambda x: x.get("updated", 0), reverse=True)
        return results[:limit]

    def recall(
        self,
        category: Optional[str] = None,
        key: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve entries by category, key, or substring query."""
        results = []

        categories = [category] if category else list(self.data.keys())

        for cat in categories:
            for entry in self.data.get(cat, []):
                if key and entry["key"] != key:
                    continue
                if query:
                    text = json.dumps(entry, ensure_ascii=False).lower()
                    if query.lower() not in text:
                        continue
                entry["access_count"] = entry.get("access_count", 0) + 1
                results.append({**entry, "category": cat})

        # Sort by recency
        results.sort(key=lambda x: x.get("updated", 0), reverse=True)
        return results[:limit]

    def list_categories(self) -> Dict[str, int]:
        """Return category names with entry counts."""
        return {cat: len(entries) for cat, entries in self.data.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Explicitly save to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        total = sum(len(v) for v in self.data.values())
        self._dirty = False
        self._last_save = time.time()
        logger.info(f"Persistent memory saved: {total} entries → {self.db_path}")

    def _load(self):
        if not os.path.exists(self.db_path):
            # Initialise with empty built-in categories
            self.data = {cat: [] for cat in self.BUILTIN_CATEGORIES}
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            total = sum(len(v) for v in self.data.values())
            logger.info(f"Persistent memory loaded: {total} entries from {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to load persistent memory: {e}")
            self.data = {cat: [] for cat in self.BUILTIN_CATEGORIES}

    def _auto_save(self):
        """Save after mutations, but debounce to avoid excessive I/O."""
        self._dirty = True
        now = time.time()
        if now - self._last_save >= self._save_interval:
            self.save()

    def _enforce_limits(self):
        """Trim oldest entries if total count exceeds max."""
        total = sum(len(v) for v in self.data.values())
        if total <= self.max_entries:
            return
        # Flatten, sort by oldest updated, remove excess
        all_entries = []
        for cat, entries in self.data.items():
            for e in entries:
                all_entries.append((cat, e))
        all_entries.sort(key=lambda x: x[1].get("updated", 0))
        excess = total - self.max_entries
        for cat, entry in all_entries[:excess]:
            self.data[cat].remove(entry)

    def deduplicate(
        self,
        threshold: float = 0.0,
        *,
        dry_run: bool = False,
    ) -> dict:
        """Remove near-duplicate entries across all categories.

        For each pair of entries within a category whose text similarity
        exceeds *threshold* (defaults to self.dedup_ratio), the newer entry
        is removed and its access_count is merged into the older one.

        Returns {category: removed_count, ...} summary.
        """
        if threshold <= 0:
            threshold = self.dedup_ratio

        removed_summary: dict = {}
        for cat, entries in self.data.items():
            if not entries:
                continue
            keep = []
            removed = 0
            for entry in entries:
                text = self._extract_comparable_text(
                    entry.get("key", ""), entry.get("value")
                )
                merged = False
                for existing in keep:
                    existing_text = self._extract_comparable_text(
                        existing.get("key", ""), existing.get("value")
                    )
                    if self._text_similarity(text, existing_text) >= threshold:
                        # Merge: bump access_count & keep most recent timestamp
                        if not dry_run:
                            existing["access_count"] = (
                                existing.get("access_count", 0)
                                + entry.get("access_count", 0)
                                + 1
                            )
                            existing["updated"] = max(
                                existing.get("updated", 0),
                                entry.get("updated", 0),
                            )
                        merged = True
                        removed += 1
                        break
                if not merged:
                    keep.append(entry)

            if removed > 0:
                removed_summary[cat] = removed
                if not dry_run:
                    self.data[cat] = keep

        if not dry_run and removed_summary:
            self.save()
            total_removed = sum(removed_summary.values())
            logger.info(
                f"Persistent dedup cleanup: removed {total_removed} duplicates "
                f"({removed_summary})"
            )

        return removed_summary

    def summary(self) -> dict:
        total = sum(len(v) for v in self.data.values())
        return {
            "total_entries": total,
            "categories": self.list_categories(),
            "db_path": self.db_path,
        }
