"""
agent_orchestrator.py — Multi-agent design and orchestration.

Enables PyCoder to **design**, **describe**, and **coordinate** sub-agents
that collaborate on complex tasks. Each sub-agent is a virtual persona
defined by:
  - A **role** (e.g. CodeReviewer, TestWriter, Researcher).
  - A **skill description** for what it can do.
  - A **system prompt** that shapes its behaviour.
  - An **assigned task** fragment.

Orchestration flow:
  1. User gives a complex task.
  2. PyCoder (as Orchestrator) **decomposes** the task into sub-tasks.
  3. For each sub-task it **designs** or reuses a specialist sub-agent.
  4. Sub-agents execute sequentially (sharing context via working memory).
  5. Orchestrator **synthesises** the results into a final answer.

All sub-agents share the same underlying LLM — they differ only in
their system prompt and context injection. No separate processes.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.utils import strip_code_fences, parse_json_response

logger = logging.getLogger(__name__)

__all__ = ["AgentOrchestrator", "SubAgent"]


# ======================================================================
# LLM prompts for orchestration
# ======================================================================

_DECOMPOSE_PROMPT = """\
你是一个任务分解引擎。用户给出了一个复杂编程任务，请将其分解为子任务，
并为每个子任务指定一个最合适的智能体角色。

可用的 PyCoder 技能（你可以基于这些设计子智能体）：
{skills}

用户任务：
{task}

请输出 JSON 数组，每条包含：
- "subtask": 子任务描述
- "role": 智能体角色名（英文，如 CodeWriter / TestWriter / Researcher / Reviewer / Debugger / Designer）
- "skill_description": 该角色需要的技能描述（中文）
- "priority": 执行优先级 (1=最高)
- "depends_on": 依赖的子任务索引列表（空=无依赖）

只输出 JSON 数组，不要输出其他内容。"""

_AGENT_SYSTEM_PROMPT_TEMPLATE = """\
你是 PyCoder 团队中的 **{role}**。

你的角色描述：{skill_description}

你的任务：
{subtask}

上下文信息（来自之前的子智能体）：
{context}

请直接完成你的任务，输出结果。"""

_SYNTHESIZE_PROMPT = """\
你是 PyCoder 编排器。以下是各子智能体完成子任务的结果。
请综合所有结果，给用户一个完整、连贯的最终回答。

用户原始任务：{task}

子任务结果：
{results}

请输出最终综合回答。"""


# ======================================================================
# SubAgent
# ======================================================================

class SubAgent:
    def __init__(
        self,
        role: str,
        skill_description: str,
        subtask: str,
        priority: int = 1,
        depends_on: List[int] = None,
    ):
        if not role or not role.strip():
            raise ValueError("Role cannot be empty or None")
        if not skill_description:
            skill_description = ""
        if not subtask:
            subtask = ""
        self.role = role
        self.skill_description = skill_description
        self.subtask = subtask
        self.priority = priority
        self.depends_on = depends_on or []
        self.result = None
        self.completed = 0

    # --- Required by test_agent.py – DO NOT REMOVE ---
    def describe(self) -> str:
        """Return a human-readable one-line description."""
        icon = "✓" if self.completed else "○"
        return f"[{icon}] {self.role}: {self.subtask[:50]}"

    def to_dict(self) -> dict:
        """Serialize sub-agent state to a dictionary."""
        return {
            "role": self.role,
            "skill_description": self.skill_description,
            "subtask": self.subtask,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "result": self.result,
            "completed": self.completed,
        }


# ======================================================================
# AgentOrchestrator
# ======================================================================

class AgentOrchestrator:
    """Designs and coordinates sub-agents for complex tasks.

    Usage::

        orch = AgentOrchestrator(model_loader, skill_registry)
        result = orch.orchestrate("Build a REST API with tests and docs")
    """

    def __init__(
        self,
        model_loader,
        skill_registry=None,
        config: Optional[dict] = None,
    ):
        self._model = model_loader
        self._skills = skill_registry
        cfg = config or {}
        self.max_sub_agents: int = cfg.get("max_sub_agents", 6)
        # History of past orchestrations
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def orchestrate(
        self,
        task: str,
        *,
        context: Optional[str] = None,
    ) -> str:
        """Full orchestration: decompose → execute sub-agents → synthesise.

        Args:
            task: The complex user task.
            context: Optional extra context (e.g. recalled memories).

        Returns:
            The synthesised final answer.
        """
        logger.info(f"Orchestrator: starting for task: {task[:200]}")

        # 1. Decompose task into sub-agent assignments
        agents = self.decompose(task)
        if not agents:
            logger.warning("Orchestrator: decomposition returned 0 agents, fallback")
            return self._fallback_generate(task, context)

        # 2. Execute sub-agents in dependency order
        self._execute_agents(agents, context)

        # 3. Synthesise results
        result = self._synthesise(task, agents)

        # 4. Record orchestration history
        self._history.append({
            "task": task[:300],
            "agents": [a.to_dict() for a in agents],
            "timestamp": time.time(),
        })

        return result

    def decompose(self, task: str) -> List[SubAgent]:
        """Use LLM to decompose a task into sub-agent assignments."""
        skills_text = ""
        if self._skills:
            skills_text = self._skills.describe_for_orchestrator()
        else:
            skills_text = "(No skill registry available)"

        prompt = _DECOMPOSE_PROMPT.format(skills=skills_text, task=task)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self._model.generate(messages)
            agents = self._parse_decomposition(raw)
            # Cap at max_sub_agents
            agents = agents[: self.max_sub_agents]
            logger.info(f"Orchestrator: decomposed into {len(agents)} sub-agents")
            return agents
        except Exception as e:
            logger.error(f"Orchestrator decomposition failed: {e}")
            return []

    def describe_agents(self, agents: Optional[List[SubAgent]] = None) -> str:
        """Describe the current set of sub-agents (or from last orchestration)."""
        if agents is None:
            if self._history:
                last = self._history[-1]
                lines = [f"🤖 **上次编排** ({len(last['agents'])} 个子智能体)\n"]
                for a in last["agents"]:
                    status = "✓" if a.get("result") else "○"
                    lines.append(
                        f"  [{status}] **{a['role']}**: {a['subtask']}"
                    )
                return "\n".join(lines)
            return "尚未执行过多智能体编排。"

        lines = [f"🤖 **子智能体编排方案** ({len(agents)} 个)\n"]
        for a in agents:
            lines.append(a.describe())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_agents(
        self,
        agents: List[SubAgent],
        parent_context: Optional[str] = None,
    ):
        """Execute sub-agents respecting dependency order."""
        # Sort by priority (lower = earlier)
        ordered = sorted(agents, key=lambda a: a.priority)
        completed_results: Dict[int, str] = {}

        for idx, agent in enumerate(ordered):
            # Build context from prior results
            ctx_parts = []
            if parent_context:
                ctx_parts.append(f"[User Context] {parent_context[:500]}")
            for dep_idx in agent.depends_on:
                if dep_idx in completed_results:
                    ctx_parts.append(
                        f"[{agents[dep_idx].role} 的结果] "
                        f"{completed_results[dep_idx][:500]}"
                    )

            context_text = "\n".join(ctx_parts) if ctx_parts else "（无）"

            # Build agent-specific system prompt
            sys_prompt = _AGENT_SYSTEM_PROMPT_TEMPLATE.format(
                role=agent.role,
                skill_description=agent.skill_description,
                subtask=agent.subtask,
                context=context_text,
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": agent.subtask},
            ]

            try:
                result = self._model.generate(messages)
                agent.result = result
                agent.completed = time.time()
                completed_results[idx] = result
                logger.info(
                    f"Orchestrator: {agent.role} completed "
                    f"({len(result)} chars)"
                )
            except Exception as e:
                agent.result = f"[Error: {e}]"
                agent.completed = time.time()
                completed_results[idx] = agent.result
                logger.error(f"Orchestrator: {agent.role} failed: {e}")

    def _synthesise(self, task: str, agents: List[SubAgent]) -> str:
        """Combine all sub-agent results into a final answer."""
        result_parts = []
        for i, a in enumerate(agents):
            result_parts.append(
                f"### {i+1}. {a.role}\n"
                f"子任务: {a.subtask}\n"
                f"结果:\n{a.result or '(无结果)'}\n"
            )

        results_text = "\n---\n".join(result_parts)
        prompt = _SYNTHESIZE_PROMPT.format(task=task, results=results_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            return self._model.generate(messages)
        except Exception as e:
            logger.error(f"Orchestrator synthesis failed: {e}")
            # Return raw concatenation as fallback
            return (
                f"**编排结果**（综合失败，以下为各子智能体原始输出）\n\n"
                + results_text
            )

    def _fallback_generate(self, task: str, context: Optional[str]) -> str:
        """Fallback when decomposition fails — just use the LLM directly."""
        messages = [{"role": "user", "content": task}]
        if context:
            messages.insert(
                0, {"role": "system", "content": f"[Context]\n{context}"}
            )
        return self._model.generate(messages)

    @staticmethod
    def _parse_decomposition(raw: str) -> List[SubAgent]:
        """Parse LLM JSON array into SubAgent list."""
        try:
            arr = parse_json_response(strip_code_fences(raw))
        except (json.JSONDecodeError, ValueError):
            return []

        if not isinstance(arr, list):
            return []

        agents = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            agents.append(
                SubAgent(
                    role=item.get("role", "Worker"),
                    skill_description=item.get("skill_description", ""),
                    subtask=item.get("subtask", ""),
                    priority=item.get("priority", len(agents) + 1),
                    depends_on=item.get("depends_on", []),
                )
            )
        return agents

    def summary(self) -> dict:
        return {
            "total_orchestrations": len(self._history),
            "max_sub_agents": self.max_sub_agents,
            "last_orchestration_agents": (
                len(self._history[-1]["agents"]) if self._history else 0
            ),
        }
