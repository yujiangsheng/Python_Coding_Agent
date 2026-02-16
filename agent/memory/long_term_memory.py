"""
long_term_memory.py — Vector-indexed long-term memory across sessions.

Uses sentence-level embeddings (via a lightweight model or TF-IDF fallback)
to store and retrieve memories by semantic similarity. Persisted to disk
as a JSON index so it survives restarts.

Enhanced with:
  - **Surprise gating**: only stores genuinely novel memories
    based on a novelty/surprise score relative to existing entries.
  - **Semantic deduplication**: rejects near-duplicate entries.
  - **Relevance decay**: older, unaccessed memories gradually lose weight.
  - **Memory consolidation**: merges highly similar entries periodically.
"""

import json
import logging
import math
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# Vector utilities
# ======================================================================


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _batch_cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Efficient cosine similarities between *query* and each row of *matrix*."""
    norms = np.linalg.norm(matrix, axis=1)
    q_norm = np.linalg.norm(query)
    denom = norms * q_norm
    denom[denom == 0] = 1e-10
    return matrix @ query / denom


class LongTermMemory:
    """Semantically-indexed long-term memory with surprise-based gating.

    Key concepts:
      - **Surprise score**: 1 − max_similarity_to_existing. A high value means
        the incoming memory is novel; a low value means it is redundant.
      - **Dedup threshold**: if max similarity ≥ this value, the incoming text
        is considered a duplicate and only the existing entry's metadata is
        bumped (not stored again).
      - **Relevance decay**: when scoring recall candidates, a time-decay
        factor downweights memories that have not been accessed recently.
      - **Consolidation**: very similar entries (>consolidation_threshold) can
        be merged into a single representative memory.

    Each memory entry has:
      - id: unique hash
      - text: the stored content
      - embedding: vector representation
      - metadata: tags, source, timestamp, access_count, surprise_score
    """

    def __init__(self, config: dict):
        self.embedding_dim: int = config.get("embedding_dim", 384)
        self.max_entries: int = config.get("max_entries", 100000)
        self.similarity_threshold: float = config.get("similarity_threshold", 0.65)
        self.index_path: str = config.get(
            "index_path", "data/memories/long_term_index.json"
        )

        # --- Surprise gating parameters ---
        # Minimum surprise score (1 − max_sim) to accept a new memory
        self.surprise_threshold: float = config.get("surprise_threshold", 0.15)
        # Near-duplicate threshold above which we skip storage entirely
        self.dedup_threshold: float = config.get("dedup_threshold", 0.90)
        # Relevance decay half-life in seconds (7 days)
        self.decay_half_life: float = config.get("decay_half_life", 7 * 86400)
        # Consolidation similarity threshold – merge entries above this
        self.consolidation_threshold: float = config.get(
            "consolidation_threshold", 0.88
        )

        self.entries: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._emb_matrix: Optional[np.ndarray] = None  # cached matrix
        self._emb_ids: List[str] = []  # ids aligned with _emb_matrix rows
        self._matrix_dirty: bool = True
        self._encoder = None

        self._load_index()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_encoder(self):
        """Lazy-load a lightweight sentence encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded SentenceTransformer for long-term memory embeddings")
            except ImportError:
                logger.info("sentence-transformers not available; using TF-IDF fallback")
                self._encoder = "tfidf"
        return self._encoder

    def _embed(self, text: str) -> np.ndarray:
        """Compute an embedding for the given text."""
        encoder = self._get_encoder()
        if encoder == "tfidf":
            return self._tfidf_embed(text)
        return encoder.encode(text, convert_to_numpy=True).astype(np.float32)

    def _tfidf_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-hashes embedding as a fallback."""
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.embedding_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ------------------------------------------------------------------
    # Embedding matrix cache (for batch cosine operations)
    # ------------------------------------------------------------------

    def _rebuild_matrix(self):
        """Rebuild the cached numpy matrix from individual embeddings."""
        if not self.embeddings:
            self._emb_matrix = np.empty((0, self.embedding_dim), dtype=np.float32)
            self._emb_ids = []
        else:
            self._emb_ids = list(self.embeddings.keys())
            self._emb_matrix = np.stack(
                [self.embeddings[mid] for mid in self._emb_ids]
            )
        self._matrix_dirty = False

    def _get_matrix(self) -> Tuple[np.ndarray, List[str]]:
        if self._matrix_dirty or self._emb_matrix is None:
            self._rebuild_matrix()
        return self._emb_matrix, self._emb_ids

    # ------------------------------------------------------------------
    # Surprise gating
    # ------------------------------------------------------------------

    def compute_surprise(self, embedding: np.ndarray) -> Tuple[float, Optional[str]]:
        """Compute how *surprising* (novel) an embedding is w.r.t. existing store.

        Returns (surprise_score, nearest_id).
        surprise = 1 − max_similarity.  Range [0, 1].
        """
        mat, ids = self._get_matrix()
        if mat.shape[0] == 0:
            return 1.0, None  # empty store → maximally novel

        sims = _batch_cosine(embedding, mat)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        return 1.0 - max_sim, ids[best_idx]

    # ------------------------------------------------------------------
    # CRUD (with surprise gating & dedup)
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        metadata: Optional[dict] = None,
        *,
        force: bool = False,
    ) -> Optional[str]:
        """Store a memory entry, applying surprise gating.

        Returns the memory id if stored, or *None* if the entry was judged
        too similar to existing memories (deduplicated).

        Args:
            text: The memory content.
            metadata: Optional metadata dict.
            force: Bypass surprise gating and always store.
        """
        mem_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Exact-hash duplicate → just bump access metadata
        if mem_id in self.entries:
            self.entries[mem_id]["metadata"]["access_count"] = (
                self.entries[mem_id]["metadata"].get("access_count", 0) + 1
            )
            self.entries[mem_id]["metadata"]["last_accessed"] = time.time()
            logger.debug(f"Long-term: exact duplicate {mem_id}, bumped access")
            return mem_id

        embedding = self._embed(text)

        # ---- Surprise gating ----
        surprise, nearest_id = self.compute_surprise(embedding)

        if not force:
            # Hard dedup: near-duplicate detected
            if surprise < (1.0 - self.dedup_threshold):
                # Similarity > dedup_threshold ⇒ skip
                if nearest_id and nearest_id in self.entries:
                    self.entries[nearest_id]["metadata"]["access_count"] = (
                        self.entries[nearest_id]["metadata"].get("access_count", 0) + 1
                    )
                    self.entries[nearest_id]["metadata"]["last_accessed"] = time.time()
                logger.debug(
                    f"Long-term: dedup — surprise={surprise:.3f}, "
                    f"nearest={nearest_id}, skipped"
                )
                return None

            # Soft gate: not novel enough
            if surprise < self.surprise_threshold:
                logger.debug(
                    f"Long-term: low surprise {surprise:.3f} < {self.surprise_threshold}, "
                    f"skipped"
                )
                return None

        # ---- Accept memory ----
        self.entries[mem_id] = {
            "text": text,
            "metadata": {
                **(metadata or {}),
                "created": time.time(),
                "last_accessed": time.time(),
                "access_count": 0,
                "surprise_score": round(surprise, 4),
            },
        }
        self.embeddings[mem_id] = embedding
        self._matrix_dirty = True

        # Evict if over capacity
        if len(self.entries) > self.max_entries:
            self._evict()

        logger.debug(
            f"Long-term: stored {mem_id} (surprise={surprise:.3f}, "
            f"total={len(self.entries)})"
        )
        return mem_id

    def recall(
        self,
        query: str,
        top_k: int = 5,
        *,
        use_decay: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most relevant memories for a query.

        When *use_decay* is True, raw cosine similarity is blended with a
        time-decay factor so that stale, unaccessed memories rank lower.
        """
        if not self.entries:
            return []

        query_emb = self._embed(query)
        mat, ids = self._get_matrix()
        if mat.shape[0] == 0:
            return []

        raw_sims = _batch_cosine(query_emb, mat)
        now = time.time()

        scored: List[Tuple[float, str]] = []
        for idx, mem_id in enumerate(ids):
            sim = float(raw_sims[idx])
            if sim < self.similarity_threshold:
                continue

            if use_decay:
                last_access = self.entries[mem_id]["metadata"].get(
                    "last_accessed", now
                )
                age = max(now - last_access, 0)
                # Exponential decay: score *= 2^(-age / half_life)
                decay = math.pow(2, -age / self.decay_half_life)
                # Blend: 80% similarity + 20% decay
                score = 0.8 * sim + 0.2 * decay
            else:
                score = sim

            scored.append((score, mem_id))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, mem_id in scored[:top_k]:
            entry = self.entries[mem_id]
            entry["metadata"]["access_count"] = (
                entry["metadata"].get("access_count", 0) + 1
            )
            entry["metadata"]["last_accessed"] = time.time()
            results.append(
                {
                    "id": mem_id,
                    "text": entry["text"],
                    "similarity": round(score, 4),
                    "metadata": entry["metadata"],
                }
            )

        return results

    # ------------------------------------------------------------------
    # Memory consolidation
    # ------------------------------------------------------------------

    def consolidate(self, dry_run: bool = False) -> int:
        """Merge highly similar entries to reduce redundancy.

        For each cluster of entries with mutual similarity ≥
        consolidation_threshold, the most-accessed entry is kept and the
        others are folded in (their access counts are summed).

        Returns the number of entries removed.
        """
        if len(self.entries) < 2:
            return 0

        mat, ids = self._get_matrix()
        removed_ids: set = set()
        merged_count = 0

        # Build pairwise similarity above threshold (greedy clustering)
        for i in range(len(ids)):
            if ids[i] in removed_ids:
                continue
            cluster = [ids[i]]
            for j in range(i + 1, len(ids)):
                if ids[j] in removed_ids:
                    continue
                sim = float(_cosine_similarity(mat[i], mat[j]))
                if sim >= self.consolidation_threshold:
                    cluster.append(ids[j])

            if len(cluster) < 2:
                continue

            # Keep the entry with highest access_count; merge the rest
            cluster.sort(
                key=lambda mid: self.entries[mid]["metadata"].get("access_count", 0),
                reverse=True,
            )
            keeper = cluster[0]
            for mid in cluster[1:]:
                if dry_run:
                    merged_count += 1
                    continue
                # Sum access counts into keeper
                self.entries[keeper]["metadata"]["access_count"] = (
                    self.entries[keeper]["metadata"].get("access_count", 0)
                    + self.entries[mid]["metadata"].get("access_count", 0)
                )
                del self.entries[mid]
                del self.embeddings[mid]
                removed_ids.add(mid)
                merged_count += 1

        if not dry_run and merged_count > 0:
            self._matrix_dirty = True
            logger.info(f"Consolidation: merged {merged_count} redundant entries")

        return merged_count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self):
        """Persist the memory index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        data = {}
        for mem_id, entry in self.entries.items():
            data[mem_id] = {
                "text": entry["text"],
                "metadata": entry["metadata"],
                "embedding": self.embeddings[mem_id].tolist(),
            }

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

        logger.info(f"Long-term memory saved: {len(data)} entries → {self.index_path}")

    def _load_index(self):
        """Load the memory index from disk if it exists."""
        if not os.path.exists(self.index_path):
            return

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for mem_id, record in data.items():
                self.entries[mem_id] = {
                    "text": record["text"],
                    "metadata": record["metadata"],
                }
                self.embeddings[mem_id] = np.array(record["embedding"], dtype=np.float32)

            logger.info(f"Long-term memory loaded: {len(self.entries)} entries from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load long-term memory: {e}")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _evict(self):
        """Remove the least-recently-accessed entries to stay within capacity."""
        entries_by_access = sorted(
            self.entries.items(),
            key=lambda kv: kv[1]["metadata"].get("last_accessed", 0),
        )
        while len(self.entries) > self.max_entries:
            mem_id, _ = entries_by_access.pop(0)
            del self.entries[mem_id]
            del self.embeddings[mem_id]
        self._matrix_dirty = True

    def summary(self) -> dict:
        return {
            "total_entries": len(self.entries),
            "index_path": self.index_path,
            "embedding_dim": self.embedding_dim,
            "surprise_threshold": self.surprise_threshold,
            "dedup_threshold": self.dedup_threshold,
        }
