"""
working_memory.py — Short-term, session-scoped memory.

Maintains the recent conversation context (turns) and actively-used code
snippets within the current session. This is the "scratchpad" the agent
uses while thinking.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from agent.utils import estimate_tokens

logger = logging.getLogger(__name__)


class WorkingMemory:
    """In-memory, session-scoped working memory.

    Stores recent conversation turns, intermediate reasoning steps,
    and currently-relevant code snippets. Automatically evicts the
    oldest entries when capacity limits are reached.
    """

    def __init__(self, config: dict):
        self.max_turns: int = config.get("max_turns", 20)
        self.max_tokens: int = config.get("max_tokens", 8192)
        self.turns: List[Dict[str, Any]] = []
        self.scratchpad: Dict[str, Any] = {}  # Named scratch slots
        self.token_count: int = 0

    # ------------------------------------------------------------------
    # Conversation turns
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str, metadata: Optional[dict] = None):
        """Add a conversation turn (user / assistant / system)."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.turns.append(entry)
        self.token_count += self._estimate_tokens(content)
        self._evict_if_needed()
        logger.debug(f"Working memory: added {role} turn ({len(self.turns)} total)")

    def get_turns(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Return recent turns as [{role, content}, …] for the model."""
        subset = self.turns[-last_n:] if last_n else self.turns
        return [{"role": t["role"], "content": t["content"]} for t in subset]

    def get_full_turns(self) -> List[Dict[str, Any]]:
        """Return turns with full metadata."""
        return list(self.turns)

    # ------------------------------------------------------------------
    # Scratchpad (named slots for intermediate results)
    # ------------------------------------------------------------------

    def set_scratch(self, key: str, value: Any):
        """Store a named intermediate value."""
        self.scratchpad[key] = {
            "value": value,
            "timestamp": time.time(),
        }



    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """CJK-aware token estimate (delegates to shared utility)."""
        return estimate_tokens(text)

    def _evict_if_needed(self):
        """Remove oldest turns to stay within limits."""
        while len(self.turns) > self.max_turns:
            removed = self.turns.pop(0)
            self.token_count -= self._estimate_tokens(removed["content"])

        while self.token_count > self.max_tokens and len(self.turns) > 1:
            removed = self.turns.pop(0)
            self.token_count -= self._estimate_tokens(removed["content"])

    def _summarize_scratchpad(self) -> str:
        lines = []
        for k, v in self.scratchpad.items():
            val = v["value"]
            if isinstance(val, str) and len(val) > 200:
                val = val[:200] + "…"
            lines.append(f"- {k}: {val}")
        return "\n".join(lines)

    def clear(self):
        """Clear all working memory."""
        self.turns.clear()
        self.scratchpad.clear()
        self.token_count = 0

    def summary(self) -> dict:
        return {
            "turns": len(self.turns),
            "estimated_tokens": self.token_count,
            "scratch_keys": list(self.scratchpad.keys()),
        }
