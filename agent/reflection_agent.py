"""
reflection_agent.py — Systematic multi-level reflection engine for PyCoder.

Five reflection levels: Response, Reasoning Audit, Execution, Session
Retrospective, and Evolution Tracking.

Author & Maintainer: Jiangsheng Yu
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agent.utils import (
    strip_code_fences, parse_json_response, JsonStore,
    QUALITY_MARKERS_BAD, QUALITY_MARKERS_GOOD,
)

logger = logging.getLogger(__name__)

__all__ = ["ReflectionAgent", "ReflectionRecord", "ReflectionLevel"]


# ======================================================================
# Data structures
# ======================================================================

class ReflectionLevel(str, Enum):
    """Granularity of a reflection."""
    RESPONSE = "response"           # single-turn answer quality
    REASONING = "reasoning"         # logical-chain audit
    EXECUTION = "execution"         # code-execution post-mortem
    SESSION = "session"             # multi-turn retrospective
    EVOLUTION = "evolution"         # cross-session growth tracking


@dataclass
class QualityScore:
    """Multi-dimensional quality assessment of a response."""
    correctness: float = 0.0        # 0-1  factual & logical accuracy
    completeness: float = 0.0       # 0-1  did it fully answer the question?
    clarity: float = 0.0            # 0-1  readability & structure
    efficiency: float = 0.0         # 0-1  code quality / solution elegance
    overall: float = 0.0            # 0-1  weighted aggregate

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QualityScore":
        return cls(**{k: d.get(k, 0.0) for k in cls.__dataclass_fields__})


@dataclass
class ReflectionRecord:
    """One reflection observation."""
    level: str = ReflectionLevel.RESPONSE
    timestamp: float = 0.0
    quality: Optional[QualityScore] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    reasoning_issues: List[str] = field(default_factory=list)
    context_snippet: str = ""       # first N chars of the triggering text
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.quality:
            d["quality"] = self.quality.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReflectionRecord":
        q = d.pop("quality", None)
        rec = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if q:
            rec.quality = QualityScore.from_dict(q)
        return rec


# ======================================================================
# LLM prompt templates
# ======================================================================

_RESPONSE_REFLECTION_PROMPT = """\
你是一个严格的自我审查引擎。请审视以下编程智能体的回答，给出客观质量评估。

用户问题：
{user_message}

智能体回答：
{response}

请从以下维度评分（0‒1）并分析：
1. **correctness**: 回答是否准确？代码是否正确？
2. **completeness**: 是否完整回答了用户的需求？有遗漏吗？
3. **clarity**: 解说清晰吗？结构合理吗？
4. **efficiency**: 代码/方案是否高效优雅？

同时列出：
- strengths: 回答的亮点（数组）
- weaknesses: 存在的不足（数组）
- suggestions: 下次如何做得更好（数组）

输出 JSON:
{{
  "correctness": 0.X,
  "completeness": 0.X,
  "clarity": 0.X,
  "efficiency": 0.X,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."]
}}
只输出 JSON，不要输出其他内容。"""


_REASONING_AUDIT_PROMPT = """\
你是一个推理审查引擎。检查以下回答中是否存在推理链缺陷。

用户问题：
{user_message}

智能体回答：
{response}

请识别以下问题（如果存在）：
1. **逻辑跳跃**: 结论缺少推导步骤
2. **前后矛盾**: 回答的不同部分之间互相冲突
3. **无根据声明**: 没有依据的断言或可能的幻觉
4. **遗漏关键条件**: 忽略了重要的边界条件或前提
5. **过度自信**: 对不确定的事情表现得过于肯定

输出 JSON:
{{
  "issues": [
    {{"type": "...", "description": "...", "severity": "low|medium|high", "location": "..."}}
  ],
  "overall_reasoning_quality": 0.X,
  "needs_revision": true/false,
  "revision_hint": "如何修正（可选）"
}}
只输出 JSON，不要输出其他内容。"""


_EXECUTION_REFLECTION_PROMPT = """\
你是一个代码执行分析引擎。以下是一段代码的执行结果，请进行反思分析。

原始需求：
{task}

生成的代码：
```python
{code}
```

执行结果：
- 成功: {success}
- 输出: {output}
- 错误: {error}

请分析：
1. 代码是否正确实现了需求？
2. 如果失败，根因是什么？
3. 有哪些可以改进的地方？
4. 从这次执行中可以学到什么教训？

输出 JSON:
{{
  "meets_requirement": true/false,
  "root_cause": "失败的根因（如果失败）",
  "improvements": ["可改进之处"],
  "lessons": ["可复用的教训"],
  "fix_strategy": "修复策略（如果需要）"
}}
只输出 JSON，不要输出其他内容。"""


_SESSION_RETROSPECTIVE_PROMPT = """\
你是一个高级自我反思引擎。以下是编程智能体在一次会话中的反思记录汇总。
请进行会话级别的回顾分析，提炼出可以指导未来进化的规律。

会话反思汇总：
- 总交互数: {interaction_count}
- 平均质量分: {avg_quality:.2f}
- 优势模式: {strength_patterns}
- 弱点模式: {weakness_patterns}
- 推理问题: {reasoning_issues}

详细记录：
{reflection_details}

请输出 JSON:
{{
  "session_rating": 0.X,
  "recurring_strengths": ["反复出现的优势"],
  "recurring_weaknesses": ["反复出现的弱点"],
  "evolution_goals": [
    {{"goal": "具体改进目标", "priority": "high|medium|low", "approach": "建议方法"}}
  ],
  "key_lessons": ["本次会话的核心教训"],
  "progress_note": "与之前相比的进步/退步观察"
}}
只输出 JSON，不要输出其他内容。"""


# ======================================================================
# Evolution Tracker — persistent cross-session growth tracking
# ======================================================================

class EvolutionTracker:
    """Track quality metrics over time to measure real improvement.

    Persists a JSON file with per-session snapshots and rolling averages.
    """

    def __init__(self, db_path: str = "data/evolution.json"):
        self.db_path = db_path
        self._store = JsonStore(db_path, default=[])
        self._store.load()
        self.snapshots: List[Dict[str, Any]] = self._store.data

    def save(self):
        self._store.data = self.snapshots
        self._store.save()

    def record_session(
        self,
        *,
        interaction_count: int,
        avg_quality: float,
        strengths: List[str],
        weaknesses: List[str],
        goals: List[Dict[str, Any]],
        lessons: List[str],
    ):
        """Add a session snapshot to the evolution history."""
        snapshot = {
            "timestamp": time.time(),
            "interaction_count": interaction_count,
            "avg_quality": round(avg_quality, 3),
            "strengths": strengths[:10],
            "weaknesses": weaknesses[:10],
            "goals": goals[:10],
            "lessons": lessons[:10],
        }
        self.snapshots.append(snapshot)
        self.save()
        logger.info(
            f"EvolutionTracker: session recorded "
            f"(total snapshots={len(self.snapshots)})"
        )

    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """Compute quality trend over the last *window* sessions."""
        if not self.snapshots:
            return {
                "trend": "no_data",
                "sessions_tracked": 0,
                "avg_quality": 0.0,
                "recent_avg": 0.0,
                "quality_delta": 0.0,
            }

        recent = self.snapshots[-window:]
        all_avg = sum(s["avg_quality"] for s in self.snapshots) / len(self.snapshots)
        recent_avg = sum(s["avg_quality"] for s in recent) / len(recent)
        delta = recent_avg - all_avg

        if delta > 0.05:
            trend = "improving"
        elif delta < -0.05:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "sessions_tracked": len(self.snapshots),
            "avg_quality": round(all_avg, 3),
            "recent_avg": round(recent_avg, 3),
            "quality_delta": round(delta, 3),
        }

    def get_unresolved_goals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve evolution goals from recent sessions.

        These can be injected into the self-improvement engine as
        high-priority targets.
        """
        goals: List[Dict[str, Any]] = []
        for snapshot in reversed(self.snapshots[-5:]):
            for g in snapshot.get("goals", []):
                goals.append(g)
        # Deduplicate by goal text
        seen = set()
        unique: List[Dict[str, Any]] = []
        for g in goals:
            key = g.get("goal", "")
            if key not in seen:
                seen.add(key)
                unique.append(g)
        return unique[:limit]

    def get_recurring_weaknesses(self, min_count: int = 2) -> List[Tuple[str, int]]:
        """Find weaknesses that appear across multiple sessions."""
        counter: Counter = Counter()
        for snapshot in self.snapshots:
            for w in snapshot.get("weaknesses", []):
                counter[w] += 1
        return [
            (w, c) for w, c in counter.most_common(20)
            if c >= min_count
        ]

    def summary(self) -> Dict[str, Any]:
        trend = self.get_trend()
        return {
            "sessions_tracked": len(self.snapshots),
            "trend": trend["trend"],
            "avg_quality": trend["avg_quality"],
            "recent_avg": trend["recent_avg"],
            "quality_delta": trend["quality_delta"],
        }


# ======================================================================
# ReflectionAgent — the main reflection engine
# ======================================================================

class ReflectionAgent:
    def __init__(self, use_llm=True, model=None, cooldown=60, max_records=100):
        self._use_llm = use_llm
        self._model = model
        self._llm_cooldown = cooldown
        self._max_session_records = max_records
        self._session_records = []
        self._evolution = EvolutionTracker()
        self._persistent = None
        self._last_llm_call = 0
        self._quality_threshold = 0.6

    @property
    def evolution_tracker(self) -> EvolutionTracker:
        return self._evolution

    def reflect_on_response(self, user_message: str, response: str, intent_type: str = "conversation") -> ReflectionRecord:
        """Lightweight rule-based reflection used as resilient fallback."""
        text = response or ""
        text_lower = text.lower()

        good_hits = sum(1 for m in QUALITY_MARKERS_GOOD if m in text_lower)
        bad_hits = sum(1 for m in QUALITY_MARKERS_BAD if m in text_lower)
        has_code = "```" in text or "def " in text

        correctness = max(0.0, min(1.0, 0.55 + 0.05 * good_hits - 0.08 * bad_hits + (0.05 if has_code else 0.0)))
        completeness = max(0.0, min(1.0, 0.45 + min(len(text), 1200) / 3000 + (0.05 if has_code else 0.0)))
        clarity = max(0.0, min(1.0, 0.5 + (0.1 if "\n" in text else 0.0) + (0.1 if "```" in text else 0.0)))
        efficiency = max(0.0, min(1.0, 0.45 + (0.1 if "o(" in text_lower else 0.0) + (0.05 if "type" in text_lower else 0.0)))
        overall = round((0.4 * correctness + 0.25 * completeness + 0.2 * clarity + 0.15 * efficiency), 3)

        strengths: List[str] = []
        weaknesses: List[str] = []
        suggestions: List[str] = []

        if has_code:
            strengths.append("包含代码实现")
        if "```" in text:
            strengths.append("结构化输出")
        if overall < self._quality_threshold:
            weaknesses.append("回答质量低于阈值")
            suggestions.append("补充完整实现与边界条件")
        if len(text.strip()) < 60:
            weaknesses.append("回答过短")
            suggestions.append("增加关键步骤解释")
        if bad_hits > good_hits:
            weaknesses.append("不确定性表达偏多")
            suggestions.append("先给可执行最小解再补充说明")

        quality = QualityScore(
            correctness=round(correctness, 3),
            completeness=round(completeness, 3),
            clarity=round(clarity, 3),
            efficiency=round(efficiency, 3),
            overall=overall,
        )

        record = ReflectionRecord(
            level=ReflectionLevel.RESPONSE,
            quality=quality,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            context_snippet=(user_message or "")[:200],
            metadata={"intent_type": intent_type},
        )
        self._session_records.append(record)
        if len(self._session_records) > self._max_session_records:
            self._session_records = self._session_records[-self._max_session_records:]
        return record

    def get_session_stats(self) -> Dict[str, Any]:
        records = self._session_records
        if not records:
            return {
                "total_reflections": 0,
                "avg_quality": 0.0,
                "weakness_count": 0,
                "level_counts": {},
            }

        qualities = [r.quality.overall for r in records if r.quality]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        weakness_count = sum(len(r.weaknesses) for r in records)
        level_counts = Counter(r.level if isinstance(r.level, str) else r.level.value for r in records)
        return {
            "total_reflections": len(records),
            "avg_quality": round(avg_quality, 3),
            "weakness_count": weakness_count,
            "level_counts": dict(level_counts),
        }

    def session_retrospective(self, interaction_count: int) -> Dict[str, Any]:
        stats = self.get_session_stats()
        all_strengths: List[str] = []
        all_weaknesses: List[str] = []
        for r in self._session_records:
            all_strengths.extend(r.strengths)
            all_weaknesses.extend(r.weaknesses)

        strengths_counter = Counter(all_strengths)
        weaknesses_counter = Counter(all_weaknesses)
        recurring_strengths = [k for k, _ in strengths_counter.most_common(10)]
        recurring_weaknesses = [k for k, _ in weaknesses_counter.most_common(10)]

        evolution_goals: List[Dict[str, Any]] = []
        for w in recurring_weaknesses[:5]:
            evolution_goals.append({
                "goal": f"改进：{w}",
                "priority": "high" if weaknesses_counter[w] >= 2 else "medium",
                "approach": "增加验证反馈闭环并补足边界条件",
            })

        key_lessons = [f"保持优势：{s}" for s in recurring_strengths[:3]]
        for w in recurring_weaknesses[:3]:
            key_lessons.append(f"重点修复：{w}")

        self._evolution.record_session(
            interaction_count=interaction_count,
            avg_quality=stats["avg_quality"],
            strengths=recurring_strengths,
            weaknesses=recurring_weaknesses,
            goals=evolution_goals,
            lessons=key_lessons,
        )

        return {
            "session_rating": stats["avg_quality"],
            "recurring_strengths": recurring_strengths,
            "recurring_weaknesses": recurring_weaknesses,
            "evolution_goals": evolution_goals,
            "key_lessons": key_lessons,
            "progress_note": "已记录到进化追踪器。",
        }

    def evolution_trend(self) -> Dict[str, Any]:
        return self._evolution.get_trend()

    def recurring_weaknesses(self, min_count: int = 2) -> List[Tuple[str, int]]:
        return self._evolution.get_recurring_weaknesses(min_count=min_count)

    def evolution_goals(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._evolution.get_unresolved_goals(limit=limit)
