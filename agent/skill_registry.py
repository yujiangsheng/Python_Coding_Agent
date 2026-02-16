"""
skill_registry.py — Agent skill description, tracking, and growth.

Maintains a structured registry of the agent's capabilities (skills).
Each skill has:
  - **name & description**: human-readable summary.
  - **level**: proficiency (novice → advanced → expert) based on success
    rate and volume of relevant experiences.
  - **metrics**: success count, failure count, average confidence.
  - **examples**: representative past uses.

The registry supports:
  - Self-assessment: the agent can describe what it can do.
  - Skill growth: metrics are updated after each interaction.
  - Skill gap detection: identifies areas that need improvement.
  - Skill descriptions for sub-agents: used by the orchestrator.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.utils import JsonStore

logger = logging.getLogger(__name__)

__all__ = ["SkillRegistry", "Skill"]


# ======================================================================
# Skill level thresholds
# ======================================================================

class SkillLevel:
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

    @staticmethod
    def from_metrics(success_count: int, total: int) -> str:
        if total == 0:
            return SkillLevel.NOVICE
        rate = success_count / total
        if total >= 50 and rate >= 0.90:
            return SkillLevel.EXPERT
        if total >= 20 and rate >= 0.80:
            return SkillLevel.ADVANCED
        if total >= 5 and rate >= 0.60:
            return SkillLevel.INTERMEDIATE
        return SkillLevel.NOVICE


# ======================================================================
# Skill dataclass
# ======================================================================

class Skill:
    """Represents a single agent capability."""

    def __init__(
        self,
        name: str,
        description: str,
        *,
        category: str = "general",
        level: str = SkillLevel.NOVICE,
        success_count: int = 0,
        failure_count: int = 0,
        total_uses: int = 0,
        examples: Optional[List[str]] = None,
        created: float = 0.0,
        updated: float = 0.0,
    ):
        self.name = name
        self.description = description
        self.category = category
        self.level = level
        self.success_count = success_count
        self.failure_count = failure_count
        self.total_uses = total_uses
        self.examples: List[str] = examples or []
        self.created = created or time.time()
        self.updated = updated or time.time()

    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    def record_use(self, success: bool, example: Optional[str] = None):
        """Record a use of this skill and update level."""
        self.total_uses += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.level = SkillLevel.from_metrics(self.success_count, self.total_uses)
        if example:
            self.examples.append(example[:200])
            # Keep last 10 examples
            self.examples = self.examples[-10:]
        self.updated = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "level": self.level,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_uses": self.total_uses,
            "success_rate": round(self.success_rate, 3),
            "examples": self.examples,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            name=d["name"],
            description=d["description"],
            category=d.get("category", "general"),
            level=d.get("level", SkillLevel.NOVICE),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            total_uses=d.get("total_uses", 0),
            examples=d.get("examples", []),
            created=d.get("created", 0.0),
            updated=d.get("updated", 0.0),
        )

    def describe(self) -> str:
        """Human-readable description for self-assessment."""
        return (
            f"**{self.name}** [{self.level}]\n"
            f"  {self.description}\n"
            f"  使用次数: {self.total_uses}, "
            f"成功率: {self.success_rate:.0%}"
        )


# ======================================================================
# Skill Registry
# ======================================================================

# Default skills the agent starts with (seed)
_DEFAULT_SKILLS = [
    {
        "name": "code_generation",
        "description": "根据自然语言需求生成高质量Python代码，包含类型注解和文档字符串",
        "category": "coding",
    },
    {
        "name": "code_debugging",
        "description": "分析错误信息和代码逻辑，定位根因并提供修复方案",
        "category": "coding",
    },
    {
        "name": "code_review",
        "description": "评估代码质量、可读性、性能，给出改进建议",
        "category": "coding",
    },
    {
        "name": "code_explanation",
        "description": "用中文解释代码逻辑、设计模式和关键概念",
        "category": "coding",
    },
    {
        "name": "test_generation",
        "description": "为给定代码生成全面的单元测试和集成测试",
        "category": "coding",
    },
    {
        "name": "intent_understanding",
        "description": "准确理解用户模糊的编程需求并推断真实意图",
        "category": "reasoning",
    },
    {
        "name": "api_knowledge",
        "description": "了解常用Python库的API用法（标准库、numpy、requests等）",
        "category": "knowledge",
    },
    {
        "name": "web_search",
        "description": "通过搜索引擎获取最新技术信息并整合为有用回答",
        "category": "research",
    },
    {
        "name": "self_improvement",
        "description": "分析自身代码并自动改进，提升代码质量和性能",
        "category": "meta",
    },
    {
        "name": "memory_management",
        "description": "管理多层记忆系统，存储和检索知识、经验和元知识",
        "category": "meta",
    },
    {
        "name": "reflection",
        "description": "系统性自我反思，评估回答质量、推理链、执行结果，追踪进化趋势",
        "category": "meta",
    },
    {
        "name": "meta_mining",
        "description": "从经验中提炼通用元知识和高阶规律，分析失败模式",
        "category": "meta",
    },
    {
        "name": "orchestration",
        "description": "设计和协调子智能体分工协作，完成复杂多步骤任务",
        "category": "reasoning",
    },
    {
        "name": "conversation",
        "description": "自然流畅的中英文对话，理解上下文，处理闲聊和模糊指令",
        "category": "general",
    },
]

# Maps IntentType → skill name for automatic tracking
INTENT_SKILL_MAP = {
    "code_generate": "code_generation",
    "code_modify": "code_generation",
    "code_debug": "code_debugging",
    "code_review": "code_review",
    "code_explain": "code_explanation",
    "code_test": "test_generation",
    "question": "api_knowledge",
    "search": "web_search",
    "self_improve": "self_improvement",
    "memory_manage": "memory_management",
    "memory_agent": "memory_management",
    "reflect": "reflection",
    "meta_mine": "meta_mining",
    "orchestrate": "orchestration",
    "skill_describe": "reflection",
    "conversation": "conversation",
    # system_command and unknown are intentionally unmapped
}


class SkillRegistry:
    """Tracks, describes, and grows agent capabilities.

    Usage::

        registry = SkillRegistry(config)
        registry.record("code_generation", success=True, example="wrote sort fn")
        print(registry.describe_all())
        gaps = registry.identify_gaps()
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.db_path: str = cfg.get("db_path", "data/skills.json")
        self.skills: Dict[str, Skill] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        skill_name: str,
        *,
        success: bool,
        example: Optional[str] = None,
    ):
        """Record a usage of a skill and update proficiency."""
        if skill_name not in self.skills:
            # Auto-register unknown skills
            self.skills[skill_name] = Skill(
                name=skill_name,
                description=f"Dynamically discovered skill: {skill_name}",
            )
        self.skills[skill_name].record_use(success, example)
        logger.debug(
            f"Skill '{skill_name}': total={self.skills[skill_name].total_uses}, "
            f"level={self.skills[skill_name].level}"
        )

    def record_for_intent(
        self,
        intent_type: str,
        *,
        success: bool,
        example: Optional[str] = None,
    ):
        """Convenience: record skill use keyed by IntentType string."""
        skill_name = INTENT_SKILL_MAP.get(intent_type)
        if skill_name:
            self.record(skill_name, success=success, example=example)

    def get(self, skill_name: str) -> Optional[Skill]:
        return self.skills.get(skill_name)

    def describe_all(self) -> str:
        """Return a human-readable summary of all skills."""
        if not self.skills:
            return "暂无已注册的技能。"
        lines = ["📋 **PyCoder 技能清单**\n"]
        by_cat: Dict[str, List[Skill]] = {}
        for s in self.skills.values():
            by_cat.setdefault(s.category, []).append(s)

        for cat in sorted(by_cat.keys()):
            lines.append(f"\n### {cat.title()}")
            for s in sorted(by_cat[cat], key=lambda x: x.name):
                lines.append(s.describe())
        return "\n".join(lines)

    def describe_for_orchestrator(self) -> str:
        """Compact skill list suitable for multi-agent prompt injection."""
        parts = []
        for s in sorted(self.skills.values(), key=lambda x: x.name):
            parts.append(
                f"- {s.name} [{s.level}]: {s.description} "
                f"(success_rate={s.success_rate:.0%})"
            )
        return "\n".join(parts)

    def identify_gaps(self) -> List[Dict[str, Any]]:
        """Identify skills that need improvement.

        Returns skills with low success rate or low usage, sorted by
        priority (worst first).
        """
        gaps = []
        for s in self.skills.values():
            priority = 0.0
            reason = ""
            if s.total_uses >= 3 and s.success_rate < 0.5:
                priority = 1.0 - s.success_rate
                reason = f"低成功率({s.success_rate:.0%})"
            elif s.total_uses == 0:
                priority = 0.3
                reason = "尚未使用"
            elif s.total_uses < 3:
                priority = 0.2
                reason = f"使用次数少({s.total_uses})"

            if priority > 0:
                gaps.append({
                    "skill": s.name,
                    "level": s.level,
                    "reason": reason,
                    "priority": round(priority, 3),
                })

        gaps.sort(key=lambda x: x["priority"], reverse=True)
        return gaps

    def add_skill(
        self,
        name: str,
        description: str,
        category: str = "general",
    ) -> Skill:
        """Register a new skill (e.g. learned dynamically)."""
        if name in self.skills:
            # Update description if re-registered
            self.skills[name].description = description
            self.skills[name].category = category
            return self.skills[name]
        skill = Skill(name=name, description=description, category=category)
        self.skills[name] = skill
        logger.info(f"SkillRegistry: registered new skill '{name}'")
        return skill

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        data = {name: s.to_dict() for name, s in self.skills.items()}
        store = JsonStore(self.db_path, default=data)
        store.data = data
        store.save()
        logger.info(f"SkillRegistry saved: {len(data)} skills → {self.db_path}")

    def _load(self):
        store = JsonStore(self.db_path, default={})
        if store.load() and store.data:
            for name, d in store.data.items():
                self.skills[name] = Skill.from_dict(d)
            logger.info(
                f"SkillRegistry loaded: {len(self.skills)} skills from {self.db_path}"
            )
            return

        # Seed with defaults
        for d in _DEFAULT_SKILLS:
            self.skills[d["name"]] = Skill(
                name=d["name"],
                description=d["description"],
                category=d.get("category", "general"),
            )
        logger.info(
            f"SkillRegistry: seeded {len(self.skills)} default skills"
        )

    def summary(self) -> dict:
        by_level: Dict[str, int] = {}
        for s in self.skills.values():
            by_level[s.level] = by_level.get(s.level, 0) + 1
        return {
            "total_skills": len(self.skills),
            "by_level": by_level,
            "total_uses": sum(s.total_uses for s in self.skills.values()),
            "avg_success_rate": round(
                (
                    sum(s.success_rate for s in self.skills.values())
                    / max(len(self.skills), 1)
                ),
                3,
            ),
        }
