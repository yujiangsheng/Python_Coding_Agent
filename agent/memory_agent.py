"""
memory_agent.py — LLM-powered memory management agent.

An intelligent agent that sits between the coding agent and the memory
system.  It decides **which memory tier** to use for each piece of
information, prevents the self-improvement engine from repeating past
mistakes, and actively encourages exploration of novel approaches.

Core responsibilities
---------------------
1. **Smart routing** — classify incoming information by type and route
   it to the optimal memory tier (working / long-term / persistent /
   external).
2. **Error prevention** — maintain an *error registry* that the
   self-improvement engine queries before applying a change.  If a
   similar change has failed before, the registry blocks it and
   suggests an alternative.
3. **Exploration guidance** — track which approaches have been tried
   for a given problem class and suggest unexplored directions so the
   agent keeps learning instead of converging on a local optimum.
4. **Memory hygiene** — advise when to consolidate, promote, or forget
   memories across tiers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.utils import (
    parse_json_response, strip_code_fences, JsonStore,
)

logger = logging.getLogger(__name__)

__all__ = ["MemoryAgent"]


# ======================================================================
# LLM prompt templates
# ======================================================================

_CLASSIFY_PROMPT = """\
你是一个记忆管理智能体。请根据以下信息内容，判断它应该被存储到哪个记忆层级。

记忆层级说明：
- working   : 当前会话临时信息（对话片段、中间计算）
- long_term : 跨会话语义知识（代码模式、通用技巧、概念关联）
- persistent: 结构化事实（错误解决方案、API用法、用户偏好、经验记录）
- external  : 外部资源引用（URL、文档链接、待验证信息）

信息内容：
{content}

上下文（最近对话）：
{context}

请输出 JSON：
{{"tier": "<working|long_term|persistent|external>", \
"category": "<该层级内的分类>", \
"reason": "<一句话解释>"}}
只输出 JSON。"""


_ERROR_CHECK_PROMPT = """\
你是一个自我改进安全顾问。编程智能体打算对自己的源码做以下修改。
请结合历史失败记录，判断这次修改是否安全。

## 拟修改
文件: {file_path}
描述: {description}
差异摘要:
{diff_summary}

## 历史失败记录（相似修改）
{failure_history}

请输出 JSON：
{{"safe": true/false, \
"risk_level": "<low|medium|high>", \
"similar_failure": "<最相似失败的简述或 null>", \
"suggestion": "<改进建议（若不安全）或 null>"}}
只输出 JSON。"""


_EXPLORATION_PROMPT = """\
你是一个探索顾问。编程智能体正在改进自身代码，以下是已尝试过的方法。
请建议 **尚未尝试** 的、具有创新性的改进方向。

## 目标文件 / 模块
{target}

## 已尝试的方法
{tried_approaches}

## 已知的元知识 / 高阶规律
{meta_knowledge}

请输出 JSON 数组，每条包含：
- "approach": 建议的新方法（一句话）
- "rationale": 为什么值得尝试
- "priority": high / medium / low

只输出 JSON 数组。"""


# ======================================================================
# Information type → default tier mapping (rule-based fast path)
# ======================================================================

_INFO_TYPE_TIER: Dict[str, Tuple[str, str]] = {
    # info_type          → (tier, category)
    "conversation":       ("working",    "turns"),
    "scratch":            ("working",    "scratch"),
    "code_pattern":       ("long_term",  "patterns"),
    "concept":            ("long_term",  "concepts"),
    "error_resolution":   ("persistent", "errors"),
    "api_reference":      ("persistent", "api_knowledge"),
    "user_preference":    ("persistent", "preferences"),
    "experience":         ("persistent", "experiences"),
    "meta_knowledge":     ("persistent", "meta_knowledge"),
    "external_reference": ("external",   "web"),
    "improvement_record": ("persistent", "improvements"),
}


# ======================================================================
# Error Registry  — tracks failed self-improvement attempts
# ======================================================================

class ErrorRegistry:
    """Persistent registry of failed improvement attempts.

    Each entry captures the file, description, diff fingerprint, error
    reason, and a timestamp. Before a new improvement is applied, the
    registry can be queried for *similar* past failures.
    """

    def __init__(self, db_path: str = "data/error_registry.json"):
        self.db_path = db_path
        self._store = JsonStore(db_path, default=[])
        self._store.load()
        self.entries: List[Dict[str, Any]] = self._store.data

    # ---- persistence ------------------------------------------------

    def save(self):
        self._store.data = self.entries
        self._store.save()

    # ---- public API -------------------------------------------------

    def record_failure(
        self,
        file_path: str,
        description: str,
        diff: str,
        error_reason: str,
    ):
        """Record a failed self-improvement attempt."""
        entry = {
            "file": os.path.basename(file_path),
            "description": description,
            "diff_hash": hashlib.sha256(diff.encode()).hexdigest()[:12],
            "diff_summary": diff[:500],
            "error": error_reason,
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self.save()
        logger.info(
            f"ErrorRegistry: recorded failure for {entry['file']}: "
            f"{description[:80]}"
        )

    def record_success(
        self,
        file_path: str,
        description: str,
    ):
        """Record a successful improvement (for approach tracking)."""
        entry = {
            "file": os.path.basename(file_path),
            "description": description,
            "diff_hash": "",
            "diff_summary": "",
            "error": None,  # None → success
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self.save()

    def find_similar_failures(
        self,
        file_path: str,
        description: str,
        *,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find past failures that are textually similar to *description*.

        Uses keyword overlap (Jaccard on word sets) as a fast heuristic.
        """
        target_file = os.path.basename(file_path)
        desc_words = set(description.lower().split())

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in self.entries:
            if entry.get("error") is None:
                continue  # skip successes
            # Same file gets a relevance boost
            file_match = 1.0 if entry["file"] == target_file else 0.0
            entry_words = set(entry["description"].lower().split())
            if not desc_words or not entry_words:
                continue
            jaccard = len(desc_words & entry_words) / len(desc_words | entry_words)
            score = 0.6 * jaccard + 0.4 * file_match
            if score > 0.15:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def get_tried_approaches(
        self,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """Return all past attempts (success + failure) for a given file."""
        target = os.path.basename(file_path)
        return [
            e for e in self.entries
            if e["file"] == target
        ]

    def summary(self) -> Dict[str, Any]:
        total = len(self.entries)
        failures = sum(1 for e in self.entries if e.get("error") is not None)
        return {
            "total_entries": total,
            "failures": failures,
            "successes": total - failures,
        }


# ======================================================================
# MemoryAgent — the intelligent memory management agent
# ======================================================================

class MemoryAgent:
    """LLM-powered agent that manages memory routing, error prevention,
    and exploration guidance for the coding agent.

    Usage::

        ma = MemoryAgent(model, memory_manager, config)
        tier, cat = ma.smart_route(info_text, context)
        check = ma.pre_improvement_check(file, desc, diff)
        ideas = ma.suggest_exploration(file, meta_knowledge)
    """

    def __init__(
        self,
        model_loader,
        memory_manager,
        config: Optional[dict] = None,
    ):
        self._model = model_loader
        self._memory = memory_manager
        cfg = config or {}
        self._error_registry = ErrorRegistry(
            db_path=cfg.get("error_registry_path", "data/error_registry.json"),
        )
        # Whether to use LLM for routing (True) or only rules (False)
        self._llm_routing: bool = cfg.get("llm_routing", True)
        # Max exploration suggestions per call
        self._max_suggestions: int = cfg.get("max_exploration_suggestions", 5)

    # ==================================================================
    # 1. Smart routing — classify info → select memory tier
    # ==================================================================

    def smart_route(
        self,
        content: str,
        info_type: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Decide which memory tier and category to use.

        Args:
            content: The information to be stored.
            info_type: Optional explicit type hint (e.g. "error_resolution").
            context: Optional recent conversation context for the LLM.

        Returns:
            (tier, category) tuple.
        """
        # Fast path: explicit type
        if info_type and info_type in _INFO_TYPE_TIER:
            tier, cat = _INFO_TYPE_TIER[info_type]
            logger.debug(f"MemoryAgent: rule-routed '{info_type}' → {tier}/{cat}")
            return tier, cat

        # Fast path: heuristic rules
        tier_cat = self._rule_classify(content)
        if tier_cat is not None:
            return tier_cat

        # Slow path: LLM classification
        if self._model and self._llm_routing:
            return self._llm_classify(content, context or "")

        # Fallback
        return "long_term", "general"

    def route_and_store(
        self,
        content: str,
        info_type: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Classify *content* and store it in the chosen tier.

        Returns a dict with ``tier``, ``category``, ``stored`` (bool).
        """
        tier, category = self.smart_route(content, info_type, context)

        stored = False
        if tier == "working":
            self._memory.working.set_scratch(
                category, {"text": content, "ts": time.time()}
            )
            stored = True
        elif tier == "long_term":
            mid = self._memory.long_term.store(content, metadata=metadata)
            stored = mid is not None
        elif tier == "persistent":
            key = content[:80].replace("\n", " ")
            result = self._memory.persistent.store(
                category=category, key=key, value=content, metadata=metadata,
            )
            stored = result is not None
        elif tier == "external":
            # External tier is read-only; just log the reference
            logger.info(f"MemoryAgent: external reference noted: {content[:100]}")
            stored = False

        logger.info(
            f"MemoryAgent: stored={stored} → {tier}/{category} "
            f"({len(content)} chars)"
        )
        return {"tier": tier, "category": category, "stored": stored}

    # ==================================================================
    # 2. Error prevention — pre-improvement safety check
    # ==================================================================

    def pre_improvement_check(
        self,
        file_path: str,
        description: str,
        diff: str,
    ) -> Dict[str, Any]:
        """Check whether a proposed self-improvement is safe.

        Queries the error registry for similar past failures and
        optionally asks the LLM for a safety judgement.

        Returns:
            {
              "safe": bool,
              "risk_level": "low" | "medium" | "high",
              "similar_failures": [...],
              "suggestion": str | None,
            }
        """
        similar = self._error_registry.find_similar_failures(
            file_path, description, limit=5,
        )

        # If exact diff hash matches a past failure → instant block
        diff_hash = hashlib.sha256(diff.encode()).hexdigest()[:12]
        for entry in similar:
            if entry.get("diff_hash") == diff_hash:
                logger.warning(
                    f"MemoryAgent: BLOCKED — identical diff already failed: "
                    f"{entry['description'][:80]}"
                )
                return {
                    "safe": False,
                    "risk_level": "high",
                    "similar_failures": similar,
                    "suggestion": (
                        f"此修改与之前失败的尝试完全相同 "
                        f"(error: {entry['error'][:200]})。"
                        f"请尝试不同的方法。"
                    ),
                }

        # If many similar failures exist → high risk
        if len(similar) >= 3:
            return {
                "safe": False,
                "risk_level": "high",
                "similar_failures": similar,
                "suggestion": (
                    f"已有 {len(similar)} 条类似修改失败。"
                    f"建议换一个全新的方向。"
                ),
            }

        # LLM safety check (if model available and there are some failures)
        if self._model and similar:
            return self._llm_safety_check(
                file_path, description, diff, similar,
            )

        # Default: safe with appropriate risk level
        risk = "medium" if similar else "low"
        return {
            "safe": True,
            "risk_level": risk,
            "similar_failures": similar,
            "suggestion": None,
        }

    def record_improvement_result(
        self,
        file_path: str,
        description: str,
        diff: str,
        *,
        success: bool,
        error_reason: str = "",
    ):
        """Called after a self-improvement attempt completes."""
        if success:
            self._error_registry.record_success(file_path, description)
            # Store the successful approach as an experience
            self._memory.persistent.store_experience(
                task=f"[Self-Improvement] {description}",
                solution=diff[:1000],
                outcome="Applied successfully",
                success=True,
                metadata={"file": os.path.basename(file_path)},
            )
        else:
            self._error_registry.record_failure(
                file_path, description, diff, error_reason,
            )
            # Store as a failed experience too (for meta-learning)
            self._memory.persistent.store_experience(
                task=f"[Self-Improvement] {description}",
                solution=diff[:1000],
                outcome=f"FAILED: {error_reason}",
                success=False,
                metadata={"file": os.path.basename(file_path)},
            )

    # ==================================================================
    # 3. Exploration guidance — suggest novel approaches
    # ==================================================================

    def suggest_exploration(
        self,
        target_file: str,
        meta_knowledge: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Suggest unexplored improvement directions for *target_file*.

        Examines the error registry for what has been tried, the
        meta-knowledge store for high-level insights, and asks the LLM
        to propose creative new approaches.
        """
        tried = self._error_registry.get_tried_approaches(target_file)
        if not tried and not self._model:
            return []

        tried_text = "\n".join(
            f"- [{('✓' if e.get('error') is None else '✗')}] "
            f"{e['description']}"
            for e in tried[-15:]  # last 15 attempts
        ) or "（尚无尝试记录）"

        meta_text = ""
        if meta_knowledge:
            meta_text = "\n".join(
                f"- {m.get('principle', m.get('lesson', ''))}"
                for m in meta_knowledge[:10]
            )
        if not meta_text:
            meta_text = "（暂无元知识）"

        if not self._model:
            # Without LLM, return a rule-based suggestion
            return self._rule_suggest(target_file, tried)

        prompt = _EXPLORATION_PROMPT.format(
            target=os.path.basename(target_file),
            tried_approaches=tried_text,
            meta_knowledge=meta_text,
        )

        try:
            raw = self._model.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=1024,
                temperature=0.6,
            )
            suggestions = parse_json_response(strip_code_fences(raw))
            if isinstance(suggestions, list):
                return suggestions[: self._max_suggestions]
        except Exception as e:
            logger.error(f"MemoryAgent exploration failed: {e}")

        return self._rule_suggest(target_file, tried)

    # ==================================================================
    # 4. Memory hygiene — advise on maintenance
    # ==================================================================

    def advise_maintenance(self) -> List[str]:
        """Return a list of maintenance recommendations."""
        advice: List[str] = []

        # Long-term memory consolidation
        lt_summary = self._memory.long_term.summary()
        if lt_summary["total_entries"] > 500:
            advice.append(
                f"长期记忆已有 {lt_summary['total_entries']} 条，"
                f"建议运行 consolidate() 合并相似条目。"
            )

        # Persistent memory size
        ps_summary = self._memory.persistent.summary()
        if ps_summary["total_entries"] > 5000:
            advice.append(
                f"持久记忆已有 {ps_summary['total_entries']} 条，"
                f"建议清理低访问量条目。"
            )

        # Error registry bloat
        err_summary = self._error_registry.summary()
        if err_summary["failures"] > 50:
            advice.append(
                f"错误注册表已有 {err_summary['failures']} 条失败记录，"
                f"建议提炼规律后归档。"
            )

        if not advice:
            advice.append("记忆系统状态良好，暂无维护建议。")

        return advice

    # ==================================================================
    # Internals — rule-based classifiers
    # ==================================================================

    @staticmethod
    def _rule_classify(content: str) -> Optional[Tuple[str, str]]:
        """Fast heuristic classification based on content markers."""
        c = content.lower()

        # Error / traceback / exception → persistent/errors
        if any(kw in c for kw in [
            "error", "traceback", "exception", "错误", "bug", "fix",
        ]):
            return "persistent", "errors"

        # API / import / library reference → persistent/api_knowledge
        if any(kw in c for kw in [
            "import ", "api", "library", "pip install", "文档", "documentation",
        ]):
            return "persistent", "api_knowledge"

        # User preference markers
        if any(kw in c for kw in [
            "偏好", "preference", "style", "风格", "习惯",
        ]):
            return "persistent", "preferences"

        # URL / http → external
        if "http://" in c or "https://" in c:
            return "external", "web"

        # Experience markers
        if any(kw in c for kw in [
            "task:", "solution:", "outcome:", "经验", "experience",
        ]):
            return "persistent", "experiences"

        # Short conversational text → working
        if len(content) < 100 and not any(
            kw in c for kw in ["def ", "class ", "import "]
        ):
            return "working", "turns"

        return None

    @staticmethod
    def _rule_suggest(
        target_file: str,
        tried: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rule-based exploration suggestions when LLM is unavailable."""
        tried_descriptions = {
            e["description"].lower() for e in tried
        }
        candidates = [
            {
                "approach": "添加更完善的类型注解和运行时类型检查",
                "rationale": "提高代码健壮性并利于静态分析",
                "priority": "medium",
            },
            {
                "approach": "引入缓存 / memoization 优化热路径",
                "rationale": "减少重复计算，提升性能",
                "priority": "medium",
            },
            {
                "approach": "增加防御性编程——边界检查和优雅降级",
                "rationale": "减少运行时崩溃",
                "priority": "high",
            },
            {
                "approach": "重构大函数为更小的、可测试的单元",
                "rationale": "降低复杂度，提高可维护性",
                "priority": "medium",
            },
            {
                "approach": "添加异步支持以提高I/O密集操作的吞吐",
                "rationale": "网络请求和文件操作可以并行化",
                "priority": "low",
            },
        ]
        # Filter out approaches that look already tried
        novel = []
        for c in candidates:
            if not any(
                kw in c["approach"].lower()
                for desc in tried_descriptions
                for kw in desc.split() if len(kw) > 3
            ):
                novel.append(c)
        return novel[:3] if novel else candidates[:2]

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _llm_classify(
        self, content: str, context: str
    ) -> Tuple[str, str]:
        """Use LLM to classify content into a memory tier."""
        prompt = _CLASSIFY_PROMPT.format(
            content=content[:1000],
            context=context[:500],
        )
        try:
            raw = self._model.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=256,
                temperature=0.1,
            )
            data = parse_json_response(strip_code_fences(raw))
            tier = data.get("tier", "long_term")
            category = data.get("category", "general")
            # Validate tier
            if tier not in ("working", "long_term", "persistent", "external"):
                tier = "long_term"
            logger.info(
                f"MemoryAgent LLM routing: {tier}/{category} "
                f"({data.get('reason', '')})"
            )
            return tier, category
        except Exception as e:
            logger.warning(f"MemoryAgent LLM classify failed: {e}")
            return "long_term", "general"

    def _llm_safety_check(
        self,
        file_path: str,
        description: str,
        diff: str,
        similar_failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ask LLM whether a proposed change is safe given failure history."""
        failure_text = "\n".join(
            f"- {e['description']} → ERROR: {e['error'][:200]}"
            for e in similar_failures[:5]
        )
        prompt = _ERROR_CHECK_PROMPT.format(
            file_path=os.path.basename(file_path),
            description=description,
            diff_summary=diff[:800],
            failure_history=failure_text,
        )
        try:
            raw = self._model.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=512,
                temperature=0.1,
            )
            data = parse_json_response(strip_code_fences(raw))
            return {
                "safe": data.get("safe", True),
                "risk_level": data.get("risk_level", "medium"),
                "similar_failures": similar_failures,
                "suggestion": data.get("suggestion"),
            }
        except Exception as e:
            logger.warning(f"MemoryAgent LLM safety check failed: {e}")
            return {
                "safe": True,
                "risk_level": "medium",
                "similar_failures": similar_failures,
                "suggestion": None,
            }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @property
    def error_registry(self) -> ErrorRegistry:
        """Expose the error registry for direct queries."""
        return self._error_registry

    def summary(self) -> Dict[str, Any]:
        """Return a status summary."""
        return {
            "error_registry": self._error_registry.summary(),
            "llm_routing": self._llm_routing,
            "max_suggestions": self._max_suggestions,
        }
