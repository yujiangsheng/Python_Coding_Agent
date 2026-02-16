"""
core.py — Main agent orchestrator.

Ties together: model loading, intent understanding, memory management,
code generation, and self-improvement into a unified agent loop.
"""

import logging
import os
import time
import yaml
from typing import Dict, Any, Optional

from agent.exceptions import ConfigError

from agent.model_loader import ModelLoader
from agent.memory.manager import MemoryManager
from agent.intent import IntentUnderstanding, IntentType, Intent
from agent.code_generator import CodeGenerator, extract_code_blocks
from agent.self_improver import SelfImprover
from agent.meta_knowledge import MetaKnowledgeMiner
from agent.skill_registry import SkillRegistry
from agent.agent_orchestrator import AgentOrchestrator
from agent.memory_agent import MemoryAgent
from agent.reflection_agent import ReflectionAgent
from agent.utils import ERROR_MARKERS, UNCERTAINTY_MARKERS

logger = logging.getLogger(__name__)


# ======================================================================
# System prompt
# ======================================================================

SYSTEM_PROMPT = """你是一个高级Python编程智能体（Python Coding Agent），名字叫 PyCoder。

你的核心能力：
1. **理解意图**：准确理解用户的编程需求，即使描述模糊也能推断意图。
2. **代码生成**：生成高质量、可运行的Python代码，包含类型注解和文档字符串。
3. **调试修复**：分析错误，找到根因，提供修复方案。
4. **代码审查**：评估代码质量，提出改进建议。
5. **自我学习**：从每次交互中学习，持续提升编程能力。
6. **记忆管理**：记住用户偏好、常见模式和历史解决方案。
7. **元知识挖掘**：从经验中提炼通用原则和高阶规律。
8. **技能自评**：描述自己的技能并持续增强。
9. **多智能体编排**：设计和协调子智能体协作完成复杂任务。
10. **智能记忆管理**：根据信息类型自动路由到最优记忆层级，防止重复错误，鼓励探索新方法。
11. **系统性反思**：每次回答后自我评估质量，审查推理链，执行后分析结果，持续追踪进化趋势。

回复规则：
- 用中文回复日常交流，代码注释和文档字符串用英文
- 遇到不确定的，先查询记忆和外部资源
- 生成代码后尝试执行验证
- 主动提出改进建议
- 复杂任务可以拆解给子智能体协作完成"""


class CodingAgent:
    """The main Python Coding Agent orchestrator."""

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config = self._load_config(config_path)
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("Initialising PyCoder — Python Coding Agent")
        logger.info("=" * 60)

        # Components (lazy-loaded)
        self._model_loader: Optional[ModelLoader] = None
        self._memory: Optional[MemoryManager] = None
        self._intent: Optional[IntentUnderstanding] = None
        self._codegen: Optional[CodeGenerator] = None
        self._improver: Optional[SelfImprover] = None
        self._meta_miner: Optional[MetaKnowledgeMiner] = None
        self._skill_registry: Optional[SkillRegistry] = None
        self._orchestrator: Optional[AgentOrchestrator] = None
        self._memory_agent: Optional[MemoryAgent] = None
        self._reflection: Optional[ReflectionAgent] = None

        self.session_start = time.time()
        self.interaction_count = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        full_path = os.path.join(self.project_root, config_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if not isinstance(cfg, dict):
                    raise ConfigError(f"Config file {config_path} is not a mapping")
                return cfg
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML in {config_path}: {e}") from e
        logger.warning(f"Config not found at {full_path}, using defaults")
        return {}

    def _setup_logging(self):
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "data/agent.log")
        log_path = os.path.join(self.project_root, log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Only configure root logger if it has no handlers yet
        root = logging.getLogger()
        if not root.handlers:
            root.setLevel(level)
            fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            root.addHandler(fh)
            root.addHandler(sh)

    # ------------------------------------------------------------------
    # Lazy component initialisation
    # ------------------------------------------------------------------

    @property
    def model(self) -> ModelLoader:
        if self._model_loader is None:
            self._model_loader = ModelLoader(self.config.get("model", {}))
            self._model_loader.load()
        return self._model_loader

    @property
    def memory(self) -> MemoryManager:
        if self._memory is None:
            self._memory = MemoryManager(self.config.get("memory", {}))
        return self._memory

    @property
    def intent_engine(self) -> IntentUnderstanding:
        if self._intent is None:
            self._intent = IntentUnderstanding(self.model)
        return self._intent

    @property
    def codegen(self) -> CodeGenerator:
        if self._codegen is None:
            self._codegen = CodeGenerator(self.model, self.config.get("execution", {}))
        return self._codegen

    @property
    def improver(self) -> SelfImprover:
        if self._improver is None:
            self._improver = SelfImprover(
                self.model,
                self.config.get("self_improvement", {}),
                self.project_root,
                memory_agent=self.memory_agent,
                reflection_agent=self.reflection,
            )
        return self._improver

    @property
    def meta_miner(self) -> MetaKnowledgeMiner:
        if self._meta_miner is None:
            self._meta_miner = MetaKnowledgeMiner(
                self.model,
                self.memory.persistent,
                self.config.get("meta_knowledge", {}),
            )
        return self._meta_miner

    @property
    def skills(self) -> SkillRegistry:
        if self._skill_registry is None:
            self._skill_registry = SkillRegistry(
                self.config.get("skills", {}),
            )
        return self._skill_registry

    @property
    def orchestrator(self) -> AgentOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = AgentOrchestrator(
                self.model,
                self.skills,
                self.config.get("orchestration", {}),
            )
        return self._orchestrator

    @property
    def memory_agent(self) -> MemoryAgent:
        if self._memory_agent is None:
            self._memory_agent = MemoryAgent(
                self.model,
                self.memory,
                self.config.get("memory_agent", {}),
            )
        return self._memory_agent

    @property
    def reflection(self) -> ReflectionAgent:
        if self._reflection is None:
            cfg = self.config.get("reflection", {})
            self._reflection = ReflectionAgent(
                use_llm=cfg.get("use_llm", True),
                model=self.model,
                cooldown=cfg.get("cooldown", 60),
                max_records=cfg.get("max_records", 100),
            )
        return self._reflection

    # ------------------------------------------------------------------
    # Main interaction loop
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        This is the main entry point for each interaction.
        Uses the full RAG pipeline for memory recall and auto-search fallback.
        """
        self.interaction_count += 1
        logger.info(f"--- Interaction #{self.interaction_count} ---")
        logger.info(f"User: {user_message[:200]}")

        # 1. Store in working memory
        self.memory.add_conversation_turn("user", user_message)

        # 2. Understand intent
        context_turns = self.memory.working.get_turns(last_n=6)
        intent = self.intent_engine.classify(user_message, context_turns)
        logger.info(f"Intent: {intent.type} (confidence={intent.confidence})")

        # 3. RAG recall — retrieve → rerank → auto-search fallback
        recalled = self.memory.rag_recall(
            user_message,
            top_k=5,
            auto_search_fallback=True,
        )

        # 4. Dispatch to handler based on intent
        response = self._dispatch(intent, user_message, recalled)

        # 5. Store response in working memory
        self.memory.add_conversation_turn("assistant", response)

        # 6. Learn from the interaction (experience replay + long-term)
        self._learn_from_interaction(user_message, intent, response)

        # 7. Reflect on the response (quality assessment + reasoning audit)
        self._reflect_on_response(user_message, intent, response)

        # 8. Post-response: if response has uncertainty markers, auto-search
        response = self._maybe_augment_with_search(response, user_message, intent)

        logger.info(f"Response: {response[:200]}...")
        return response

    def _dispatch(self, intent: Intent, user_message: str,
                  recalled: Dict[str, Any]) -> str:
        """Route to the appropriate handler based on intent type."""
        handlers = {
            IntentType.CODE_GENERATE: self._handle_code_generate,
            IntentType.CODE_MODIFY: self._handle_code_modify,
            IntentType.CODE_EXPLAIN: self._handle_code_explain,
            IntentType.CODE_DEBUG: self._handle_code_debug,
            IntentType.CODE_REVIEW: self._handle_code_review,
            IntentType.CODE_TEST: self._handle_code_test,
            IntentType.QUESTION: self._handle_question,
            IntentType.SEARCH: self._handle_search,
            IntentType.SELF_IMPROVE: self._handle_self_improve,
            IntentType.MEMORY_MANAGE: self._handle_memory,
            IntentType.SYSTEM_COMMAND: self._handle_system,
            IntentType.CONVERSATION: self._handle_conversation,
            IntentType.SKILL_DESCRIBE: self._handle_skill_describe,
            IntentType.META_MINE: self._handle_meta_mine,
            IntentType.ORCHESTRATE: self._handle_orchestrate,
            IntentType.MEMORY_AGENT: self._handle_memory_agent,
            IntentType.REFLECT: self._handle_reflect,
        }

        handler = handlers.get(intent.type, self._handle_conversation)
        try:
            return handler(intent, user_message, recalled)
        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
            return f"处理过程中出现错误：{e}\n\n请重试或换一种方式描述你的需求。"

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_code_generate(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Generate new code."""
        context = self._build_context(recalled)
        result = self.codegen.generate_and_run(msg, context=context)

        response_parts = [result["response"]]
        exec_result = result["result"]

        if hasattr(exec_result, 'summary'):
            response_parts.append(f"\n**执行结果：**\n{exec_result.summary()}")

        if len(result.get("iterations", [])) > 1:
            response_parts.append(
                f"\n（经过 {len(result['iterations'])} 次迭代自动修复）"
            )

        # Store experience via MemoryAgent (single path — no duplication)
        success = hasattr(exec_result, 'success') and exec_result.success
        outcome = exec_result.summary() if hasattr(exec_result, 'summary') else "unknown"
        self.memory_agent.route_and_store(
            f"Task: {msg[:500]}\nCode: {result.get('code', '')[:1000]}\nOutcome: {outcome}",
            info_type="experience",
            metadata={"intent": intent.type, "success": success},
        )

        return "\n".join(response_parts)

    def _handle_code_modify(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Modify existing code."""
        context = self._build_context(recalled)
        response, code = self.codegen.generate(msg, context=context)
        return response

    def _handle_code_explain(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Explain code."""
        code_blocks = extract_code_blocks(msg)
        if code_blocks:
            return self.codegen.explain_code(code_blocks[0])
        return self._handle_question(intent, msg, recalled)

    def _handle_code_debug(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Debug code."""
        context = self._build_context(recalled)
        result = self.codegen.generate_and_run(msg, context=context, auto_fix=True)

        response_parts = [result["response"]]
        exec_result = result["result"]
        if hasattr(exec_result, 'summary'):
            response_parts.append(f"\n**调试结果：**\n{exec_result.summary()}")

        # Store experience via MemoryAgent (single path — no duplication)
        success = hasattr(exec_result, 'success') and exec_result.success
        outcome = exec_result.summary() if hasattr(exec_result, 'summary') else "unknown"
        self.memory_agent.route_and_store(
            f"[Debug] Task: {msg[:400]}\nFix: {result.get('code', '')[:1000]}\nOutcome: {outcome}",
            info_type="experience",
            metadata={"intent": "debug", "success": success},
        )

        return "\n".join(response_parts)

    def _handle_code_review(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Review code."""
        code_blocks = extract_code_blocks(msg)
        if code_blocks:
            return self.codegen.review_code(code_blocks[0])
        messages = self.memory.get_context_messages(SYSTEM_PROMPT, relevant_query=msg)
        messages.append({"role": "user", "content": msg})
        return self.model.generate(messages)

    def _handle_code_test(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Generate tests."""
        code_blocks = extract_code_blocks(msg)
        if code_blocks:
            response, test_code = self.codegen.write_tests(code_blocks[0])
            return response
        messages = self.memory.get_context_messages(SYSTEM_PROMPT, relevant_query=msg)
        messages.append({"role": "user", "content": msg})
        return self.model.generate(messages)

    def _handle_question(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Answer a programming question, enriched with RAG context."""
        messages = self.memory.get_context_messages(
            SYSTEM_PROMPT, relevant_query=msg, use_rag=True
        )
        messages.append({"role": "user", "content": msg})
        return self.model.generate(messages)

    def _handle_search(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Search external resources."""
        search_results = self.memory.search_external(msg)

        # Feed search results into LLM for synthesis
        messages = self.memory.get_context_messages(SYSTEM_PROMPT)
        messages.append({
            "role": "system",
            "content": f"[Search Results]\n{search_results}",
        })
        messages.append({"role": "user", "content": msg})

        response = self.model.generate(messages)

        # Remember useful findings
        self.memory.remember(
            f"Search: {msg}\nFindings: {search_results[:500]}",
            category="api_knowledge",
        )

        return response

    def _handle_self_improve(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Run self-improvement cycle."""
        logger.info("Starting self-improvement cycle")
        records = self.improver.run_improvement_cycle()

        result_lines = ["🔧 **自我改进报告**\n"]
        for record in records:
            status = "✓ 已应用" if record.applied else "✗ 未应用"
            result_lines.append(
                f"- {status} | {record.description} "
                f"(confidence={record.confidence:.2f})"
            )
            if record.diff:
                result_lines.append(f"  ```diff\n{record.diff[:500]}\n  ```")

        result_lines.append(f"\n{self.improver.summary()['summary_text']}")
        return "\n".join(result_lines)

    def _handle_memory(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Handle memory management commands."""
        msg_lower = msg.lower()
        if "记住" in msg_lower or "remember" in msg_lower:
            self.memory.remember(msg, category="custom")
            return "好的，我已经记住了。"
        elif "回忆" in msg_lower or "recall" in msg_lower:
            results = self.memory.recall(msg, tiers=["long_term", "persistent"])
            parts = ["**相关记忆：**\n"]
            for tier, entries in results.items():
                if entries:
                    parts.append(f"*{tier}*:")
                    for e in entries[:5]:
                        text = e.get("text", e.get("value", e.get("key", "?")))
                        if isinstance(text, str) and len(text) > 200:
                            text = text[:200] + "…"
                        parts.append(f"  - {text}")
            return "\n".join(parts) if len(parts) > 1 else "没有找到相关记忆。"
        else:
            summary = self.memory.summary()
            return f"**记忆系统状态：**\n```json\n{yaml.dump(summary, allow_unicode=True)}```"

    def _handle_system(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Handle system command requests."""
        return (
            "出于安全考虑，我不会直接执行系统命令。但我可以：\n"
            "1. 生成你需要的命令脚本\n"
            "2. 解释命令的作用\n"
            "3. 在沙箱中执行Python代码\n\n"
            "请告诉我你具体需要什么。"
        )

    def _handle_conversation(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Handle general conversation."""
        messages = self.memory.get_context_messages(SYSTEM_PROMPT)
        messages.append({"role": "user", "content": msg})
        return self.model.generate(messages)

    def _handle_skill_describe(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Describe the agent's skills and identify gaps."""
        parts = [self.skills.describe_all()]
        gaps = self.skills.identify_gaps()
        if gaps:
            parts.append("\n### 🔍 需要加强的领域")
            for g in gaps[:5]:
                parts.append(
                    f"  - **{g['skill']}** [{g['level']}]: {g['reason']} "
                    f"(priority={g['priority']})"
                )
        return "\n".join(parts)

    def _handle_meta_mine(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Run meta-knowledge mining cycle."""
        logger.info("Starting meta-knowledge mining")
        insights = self.meta_miner.mine(force=True)
        if not insights:
            exp_count = len(self.memory.persistent.recall(
                category="experiences", limit=10000,
            ))
            return (
                "暂时没有足够的经验数据来提炼元知识。\n"
                "继续使用编程功能积累经验后再试。\n\n"
                f"当前经验数: {exp_count}"
            )
        stored = self.meta_miner.store_insights(insights)
        parts = [f"🧠 **元知识挖掘报告**\n提炼了 {len(insights)} 条元知识，"
                 f"新增存储 {stored} 条。\n"]
        for ins in insights:
            kind = ins.get('kind', '?')
            text = ins.get('principle', ins.get('lesson', '?'))
            conf = ins.get('confidence', 0)
            parts.append(f"  - [{kind}] {text} (confidence={conf})")
        return "\n".join(parts)

    def _handle_orchestrate(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Decompose a complex task and run multi-agent orchestration."""
        logger.info("Starting multi-agent orchestration")
        context = self._build_context(recalled)
        context_text = "\n".join(
            c.get("content", "") for c in context
        ) if context else None
        result = self.orchestrator.orchestrate(msg, context=context_text)
        # Append agent plan description
        plan = self.orchestrator.describe_agents()
        return f"{result}\n\n---\n{plan}"

    def _handle_memory_agent(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Show memory agent status and maintenance advice."""
        parts = ["🧠 **记忆管理智能体报告**\n"]
        # Summary
        ma_summary = self.memory_agent.summary()
        err = ma_summary["error_registry"]
        parts.append(
            f"错误注册表: {err['failures']} 次失败 / "
            f"{err['successes']} 次成功"
        )
        # Maintenance advice
        advice = self.memory_agent.advise_maintenance()
        parts.append("\n### 维护建议")
        for a in advice:
            parts.append(f"  - {a}")
        return "\n".join(parts)

    def _handle_reflect(self, intent: Intent, msg: str, recalled: dict) -> str:
        """Show reflection status, trigger retrospective, or show evolution."""
        msg_lower = msg.lower()

        # Session retrospective
        if any(kw in msg_lower for kw in ["回顾", "retrospect", "总结会话", "session"]):
            retro = self.reflection.session_retrospective(self.interaction_count)
            parts = ["🔍 **会话反思回顾**\n"]
            parts.append(f"会话评分: {retro['session_rating']:.2f}")
            if retro["recurring_strengths"]:
                parts.append(f"\n✅ 优势: {', '.join(retro['recurring_strengths'][:5])}")
            if retro["recurring_weaknesses"]:
                parts.append(f"\n⚠️ 弱点: {', '.join(retro['recurring_weaknesses'][:5])}")
            if retro["evolution_goals"]:
                parts.append("\n### 进化目标")
                for g in retro["evolution_goals"][:5]:
                    parts.append(f"  - [{g.get('priority','?')}] {g.get('goal','')}")
            if retro["key_lessons"]:
                parts.append("\n### 关键教训")
                for les in retro["key_lessons"][:5]:
                    parts.append(f"  - {les}")
            parts.append(f"\n{retro.get('progress_note', '')}")
            return "\n".join(parts)

        # Evolution trend
        if any(kw in msg_lower for kw in ["进化", "evolution", "趋势", "trend", "成长"]):
            trend = self.reflection.evolution_trend()
            parts = ["📈 **进化趋势报告**\n"]
            parts.append(f"已追踪会话数: {trend['sessions_tracked']}")
            parts.append(f"总体平均质量: {trend['avg_quality']:.3f}")
            parts.append(f"近期平均质量: {trend['recent_avg']:.3f}")
            parts.append(f"质量变化: {trend['quality_delta']:+.3f}")
            trend_label = {"improving": "📈 上升", "declining": "📉 下降", "stable": "➡️ 稳定", "no_data": "❓ 暂无数据"}
            parts.append(f"趋势: {trend_label.get(trend['trend'], trend['trend'])}")
            # Recurring weaknesses
            rw = self.reflection.recurring_weaknesses()
            if rw:
                parts.append("\n### 反复出现的弱点")
                for w, count in rw[:5]:
                    parts.append(f"  - ({count}次) {w}")
            # Evolution goals
            goals = self.reflection.evolution_goals()
            if goals:
                parts.append("\n### 待实现的进化目标")
                for g in goals[:5]:
                    parts.append(f"  - [{g.get('priority','?')}] {g.get('goal','')}")
            return "\n".join(parts)

        # Default: session stats
        stats = self.reflection.get_session_stats()
        evo = self.reflection.evolution_trend()
        parts = ["🪞 **反思智能体状态**\n"]
        parts.append(f"本次会话反思数: {stats['total_reflections']}")
        parts.append(f"平均质量分: {stats['avg_quality']:.3f}")
        parts.append(f"弱点计数: {stats['weakness_count']}")
        if stats['level_counts']:
            parts.append(f"按级别: {stats['level_counts']}")
        parts.append(f"\n进化追踪: {evo['sessions_tracked']} 个会话, 趋势={evo['trend']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Reflection integration
    # ------------------------------------------------------------------

    def _reflect_on_response(self, user_msg: str, intent: Intent, response: str):
        """Run per-turn reflection after each response.

        Evaluates response quality and, for low-quality responses, logs
        improvement tickets.  This data feeds the session retrospective
        and eventually the evolution tracker.
        """
        # Skip reflection for meta/system intents to avoid loops
        skip_intents = (
            IntentType.SELF_IMPROVE, IntentType.META_MINE,
            IntentType.MEMORY_AGENT, IntentType.MEMORY_MANAGE,
            IntentType.SYSTEM_COMMAND,
        )
        # Also skip if intent matches REFLECT to avoid self-reflection loop
        if intent.type in skip_intents or intent.type == "reflect":
            return

        try:
            record = self.reflection.reflect_on_response(
                user_msg, response, intent_type=intent.type,
            )
            if record.quality and record.quality.overall < self.reflection._quality_threshold:
                logger.info(
                    f"Reflection: low quality ({record.quality.overall:.2f}) "
                    f"detected — weaknesses: {record.weaknesses[:3]}"
                )
        except Exception as e:
            logger.debug(f"Reflection failed (non-critical): {e}")

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _learn_from_interaction(self, user_msg: str, intent: Intent, response: str):
        """Extract and store learnings from this interaction.

        Updates the skill registry and routes significant interactions
        through the MemoryAgent for smart storage.
        """
        # Track skill usage (success heuristic: no error markers in response)
        resp_lower = response.lower()
        success = not any(m in resp_lower for m in ERROR_MARKERS)
        self.skills.record_for_intent(
            intent.type,
            success=success,
            example=user_msg[:200],
        )

        # Use MemoryAgent for smart routing of significant interactions
        # (code handlers already store experiences; here we store patterns)
        routing_intents = (
            IntentType.CODE_REVIEW, IntentType.CODE_TEST,
            IntentType.CODE_EXPLAIN, IntentType.CODE_MODIFY,
            IntentType.QUESTION,
        )
        if intent.type in routing_intents:
            info_type = "concept" if intent.type == IntentType.QUESTION else "code_pattern"
            summary = f"[{intent.type}] User: {user_msg[:200]}\nResponse: {response[:500]}"
            self.memory_agent.route_and_store(
                summary,
                info_type=info_type,
                metadata={"intent": intent.type},
            )

    # ------------------------------------------------------------------
    # Post-response auto-search augmentation
    # ------------------------------------------------------------------

    def _maybe_augment_with_search(self, response: str, user_msg: str, intent: Intent) -> str:
        if intent.type in (IntentType.SEARCH, IntentType.SYSTEM_COMMAND,
                           IntentType.MEMORY_MANAGE, IntentType.SELF_IMPROVE):
            return response

        resp_lower = response.lower()
        has_uncertainty = any(m in resp_lower for m in UNCERTAINTY_MARKERS)

        if not has_uncertainty:
            return response

        logger.info("Detected uncertainty in response, auto-searching…")
        try:
            search_summary = self.memory.search_external(user_msg)
            if search_summary and "No external results" not in search_summary:
                messages = self.memory.get_context_messages(SYSTEM_PROMPT)
                messages.append({
                    "role": "system",
                    "content": f"[Auto-Search Results]\n{search_summary}",
                })
                messages.append({
                    "role": "user",
                    "content": f"请根据以上搜索结果重新回答用户的问题。用户原始问题：{user_msg}",
                })
                augmented = self.model.generate(messages)
                self.memory.remember(
                    f"Auto-search for: {user_msg[:200]}\n{search_summary[:500]}",
                    category="api_knowledge",
                )
                return augmented
        except Exception as e:
            logger.warning(f"Auto-search augmentation failed: {e}")
            return response

    def _build_context(self, recalled: dict) -> list:
        """Build context messages from recalled memories."""
        context = []
        for tier, entries in recalled.items():
            for entry in entries[:3]:
                text = entry.get("text", entry.get("value", ""))
                if text:
                    context.append({
                        "role": "system",
                        "content": f"[Memory:{tier}] {text[:500]}",
                    })
        return context

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def save_session(self):
        """Persist all state."""
        self.memory.save_all()
        if self._skill_registry:
            self.skills.save()
        if self._memory_agent:
            self.memory_agent.error_registry.save()
        if self._reflection:
            tracker = getattr(self.reflection, "evolution_tracker", None)
            if tracker is None:
                tracker = getattr(self.reflection, "_evolution", None)
            if tracker and hasattr(tracker, "save"):
                tracker.save()
        logger.info("Session saved")

    def status(self) -> str:
        """Return agent status summary."""
        uptime = time.time() - self.session_start
        parts = [
            "=" * 50,
            "PyCoder — Python Coding Agent Status",
            "=" * 50,
            f"Uptime: {uptime/60:.1f} minutes",
            f"Interactions: {self.interaction_count}",
        ]

        if self._model_loader:
            info = self.model.get_device_info()
            parts.append(f"Backend: {info.get('backend', '?')}")
            parts.append(f"Model: {info.get('model', '?')}")
            parts.append(f"Device: {info.get('device', '?')}")
            if 'dtype' in info:
                parts.append(f"Dtype: {info['dtype']}")
        if self._memory:
            mem = self.memory.summary()
            parts.append(f"Working Memory: {mem['working']['turns']} turns")
            parts.append(f"Long-term Memory: {mem['long_term']['total_entries']} entries")
            parts.append(f"Persistent Memory: {mem['persistent']['total_entries']} entries")

        if self._improver:
            parts.append(self.improver.summary()["summary_text"])
        if self._skill_registry:
            sk = self.skills.summary()
            parts.append(
                f"Skills: {sk['total_skills']} registered, "
                f"{sk['total_uses']} total uses, "
                f"avg success={sk['avg_success_rate']:.0%}"
            )
        if self._orchestrator:
            parts.append(
                f"Orchestrations: {self.orchestrator.summary()['total_orchestrations']}"
            )
        if self._memory_agent:
            ma = self.memory_agent.summary()["error_registry"]
            parts.append(
                f"Memory Agent: {ma['failures']} failures / "
                f"{ma['successes']} successes tracked"
            )
        if self._reflection:
            rs = self.reflection.get_session_stats()
            evo = self.reflection.evolution_trend()
            parts.append(
                f"Reflection: {rs['total_reflections']} reflections, "
                f"avg_quality={rs['avg_quality']:.3f}, "
                f"evolution={evo['trend']} ({evo['sessions_tracked']} sessions)"
            )

        return "\n".join(parts)


def create_agent(config_path: str = "config.yaml") -> CodingAgent:
    """Factory function to create and return a CodingAgent instance."""
    return CodingAgent(config_path=config_path)
