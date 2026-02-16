"""
meta_knowledge.py — Meta-knowledge and meta-experience mining.

Goes beyond first-order knowledge (facts, patterns, experiences) to
extract **higher-order insights**:
  - *Meta-patterns*: recurring strategies that work across domains
    (e.g. "divide-and-conquer often helps with sequence problems").
  - *Meta-experiences*: observations about the learning process itself
    (e.g. "debug tasks are resolved faster when error messages are included").
  - *Failure analysis*: systematic patterns in failed attempts.

The miner periodically scans accumulated persistent memory and uses the
LLM to distill generalised principles, which are then stored back as
high-priority knowledge for future context injection.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.utils import strip_code_fences, parse_json_response

logger = logging.getLogger(__name__)

__all__ = ["MetaKnowledgeMiner"]

# ======================================================================
# LLM prompt templates
# ======================================================================

_META_PATTERN_PROMPT = """\
你是一个高级反思引擎。下面是一组编程任务的经验记录（task → solution → outcome）。
请从中提炼出**通用的元知识/元经验**（高阶规律），而不是重复具体事实。

经验记录：
{experiences}

请输出 JSON 数组，每条元知识包含：
- "principle": 一句话描述的通用原则（不要引用具体任务）
- "category": 分类（strategy / anti_pattern / debugging_heuristic / design_principle / learning_insight）
- "confidence": 置信度 0‒1
- "evidence_count": 支撑该原则的经验条数

只输出 JSON 数组，不要输出其他内容。"""

_FAILURE_ANALYSIS_PROMPT = """\
你是一个故障分析引擎。下面是一组失败的编程经验记录。
请分析失败模式，提取可复用的教训。

失败记录：
{failures}

请输出 JSON 数组，每条包含：
- "lesson": 教训描述
- "root_cause_pattern": 根因模式
- "prevention": 避免方法
- "confidence": 置信度 0‒1

只输出 JSON 数组，不要输出其他内容。"""


class MetaKnowledgeMiner:
    """Extracts meta-knowledge from accumulated experiences.

    Usage::

        miner = MetaKnowledgeMiner(model_loader, persistent_memory)
        insights = miner.mine()          # full mining cycle
        miner.store_insights(insights)    # persist back
    """

    # Persistent memory category for meta-knowledge
    META_CATEGORY = "meta_knowledge"

    def __init__(
        self,
        model_loader,
        persistent_memory,
        config: Optional[dict] = None,
    ):
        self._model = model_loader
        self._persistent = persistent_memory
        cfg = config or {}
        # Min experiences required before attempting meta-mining
        self.min_experiences: int = cfg.get("min_experiences", 5)
        # Max experiences fed to the LLM per mining call
        self.batch_size: int = cfg.get("batch_size", 20)
        # Cool-down between mining runs (seconds, default 1 hour)
        self.cooldown: float = cfg.get("cooldown", 3600)
        self._last_mine: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mine(self, *, force: bool = False) -> List[Dict[str, Any]]:
        """Run a full meta-knowledge mining cycle.

        Returns a list of extracted insights (both patterns and failure
        lessons combined).
        """
        now = time.time()
        if not force and (now - self._last_mine) < self.cooldown:
            logger.debug(
                "MetaKnowledgeMiner: cooldown active, skipping "
                f"({self.cooldown - (now - self._last_mine):.0f}s remaining)"
            )
            return []

        experiences = self._gather_experiences()
        if len(experiences) < self.min_experiences:
            logger.info(
                f"MetaKnowledgeMiner: only {len(experiences)} experiences "
                f"(need {self.min_experiences}), skipping"
            )
            return []

        insights: List[Dict[str, Any]] = []

        # 1. Mine meta-patterns from successes
        successes = [e for e in experiences if self._is_success(e)]
        if successes:
            patterns = self._mine_patterns(successes[: self.batch_size])
            insights.extend(patterns)

        # 2. Mine failure lessons
        failures = [e for e in experiences if not self._is_success(e)]
        if failures:
            lessons = self._mine_failures(failures[: self.batch_size])
            insights.extend(lessons)

        self._last_mine = time.time()
        logger.info(
            f"MetaKnowledgeMiner: extracted {len(insights)} insights "
            f"from {len(experiences)} experiences"
        )
        return insights

    def store_insights(self, insights: List[Dict[str, Any]]) -> int:
        """Persist extracted insights into persistent memory.

        Returns the number of *new* insights stored (dedup filters some).
        """
        stored = 0
        # Ensure meta_knowledge category exists
        if self.META_CATEGORY not in self._persistent.data:
            self._persistent.data[self.META_CATEGORY] = []

        for insight in insights:
            key = self._insight_key(insight)
            result = self._persistent.store(
                category=self.META_CATEGORY,
                key=key,
                value=insight,
                metadata={"mined_at": time.time()},
            )
            if result is not None:
                stored += 1

        logger.info(f"MetaKnowledgeMiner: stored {stored}/{len(insights)} insights")
        return stored

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gather_experiences(self) -> List[Dict[str, Any]]:
        """Collect experience entries from persistent memory."""
        return self._persistent.recall(
            category="experiences", limit=self.batch_size * 2,
        )

    @staticmethod
    def _is_success(entry: Dict[str, Any]) -> bool:
        val = entry.get("value", {})
        if isinstance(val, dict):
            return val.get("success", True)
        meta = entry.get("metadata", {})
        return meta.get("success", True)

    def _mine_patterns(
        self, experiences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract meta-patterns from successful experiences."""
        text = self._format_experiences(experiences)
        prompt = _META_PATTERN_PROMPT.format(experiences=text)
        return self._llm_extract(prompt, kind="meta_pattern")

    def _mine_failures(
        self, failures: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract lessons from failed experiences."""
        text = self._format_experiences(failures)
        prompt = _FAILURE_ANALYSIS_PROMPT.format(failures=text)
        return self._llm_extract(prompt, kind="failure_lesson")

    def _llm_extract(
        self, prompt: str, kind: str
    ) -> List[Dict[str, Any]]:
        """Call the LLM and parse the JSON array response."""
        if self._model is None:
            logger.warning("MetaKnowledgeMiner: no model, skipping LLM extraction")
            return []

        try:
            messages = [{"role": "user", "content": prompt}]
            raw = self._model.generate(messages)
            return self._parse_json_array(raw, kind)
        except Exception as e:
            logger.error(f"MetaKnowledgeMiner LLM extraction failed: {e}")
            return []

    @staticmethod
    def _format_experiences(entries: List[Dict[str, Any]]) -> str:
        lines = []
        for i, e in enumerate(entries, 1):
            val = e.get("value", {})
            if isinstance(val, dict):
                task = val.get("task", "?")
                sol = val.get("solution", "?")[:200]
                out = val.get("outcome", "?")
                lines.append(f"{i}. Task: {task}\n   Solution: {sol}\n   Outcome: {out}")
            else:
                lines.append(f"{i}. {str(val)[:300]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_array(
        raw: str, kind: str
    ) -> List[Dict[str, Any]]:
        """Best-effort extraction of a JSON array from LLM output."""
        try:
            result = parse_json_response(strip_code_fences(raw))
            if not isinstance(result, list):
                return []
            for item in result:
                if isinstance(item, dict):
                    item["kind"] = kind
            return result
        except (json.JSONDecodeError, ValueError):
            return []

    @staticmethod
    def _insight_key(insight: Dict[str, Any]) -> str:
        """Generate a dedup-friendly key from an insight."""
        text = insight.get("principle", insight.get("lesson", ""))
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        kind = insight.get("kind", "meta")
        return f"{kind}_{h}"

    def summary(self) -> dict:
        meta_entries = self._persistent.recall(
            category=self.META_CATEGORY, limit=10000,
        )
        return {
            "meta_knowledge_entries": len(meta_entries),
            "min_experiences": self.min_experiences,
            "last_mined": self._last_mine,
        }
