"""
intent.py — Intent understanding & task decomposition.

Classifies user requests into actionable intents and extracts structured
parameters. Uses the LLM itself for nuanced understanding, with a fast
rule-based pre-classifier for common patterns.
"""

import logging
import re
import json
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

from agent.utils import parse_json_response

logger = logging.getLogger(__name__)

__all__ = ["IntentType", "Intent", "IntentUnderstanding"]


# ======================================================================
# Intent taxonomy
# ======================================================================

class IntentType(str, Enum):
    """Recognised intent types.

    Inherits from ``str`` so values compare equal to plain strings,
    e.g. ``IntentType.CODE_GENERATE == "code_generate"`` is True.
    """
    CODE_GENERATE = "code_generate"
    CODE_MODIFY = "code_modify"
    CODE_EXPLAIN = "code_explain"
    CODE_DEBUG = "code_debug"
    CODE_REVIEW = "code_review"
    CODE_TEST = "code_test"
    QUESTION = "question"
    SEARCH = "search"
    SELF_IMPROVE = "self_improve"
    MEMORY_MANAGE = "memory_manage"
    SYSTEM_COMMAND = "system_command"
    SKILL_DESCRIBE = "skill_describe"
    META_MINE = "meta_mine"
    ORCHESTRATE = "orchestrate"
    MEMORY_AGENT = "memory_agent"
    REFLECT = "reflect"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Parsed user intent with structured parameters."""
    type: str = IntentType.UNKNOWN
    confidence: float = 0.0
    description: str = ""
    language: str = "python"
    parameters: Dict[str, Any] = field(default_factory=dict)
    sub_tasks: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ======================================================================
# Rule-based pre-classifier (fast path)
# ======================================================================

_PATTERNS = [
    (IntentType.CODE_GENERATE, [
        r"(?:写|编写|生成|创建|实现|write|create|generate|implement|make|build)\s*(?:一个|一段|个)?.*(?:代码|函数|类|脚本|程序|code|function|class|script|program)",
        r"(?:帮我|请|please)\s*(?:写|编写|实现|write|implement|create)",
        r"(?:code|function|class|method|script)\s+(?:for|to|that)",
    ]),
    (IntentType.CODE_MODIFY, [
        r"(?:修改|改|重构|优化|refactor|modify|change|update|fix|improve)\s*(?:这|这段|the|this)?\s*(?:代码|code)",
        r"(?:add|添加)\s+(?:a |an )?(?:feature|method|parameter|参数|功能|方法)",
    ]),
    (IntentType.CODE_EXPLAIN, [
        r"(?:解释|说明|explain|describe|what does|怎么理解)",
        r"(?:这段|this)\s*(?:代码|code)\s*(?:是什么意思|做了什么|does what|means?)",
    ]),
    (IntentType.CODE_DEBUG, [
        r"(?:调试|debug|fix|修复|bug|error|错误|异常|exception|traceback)",
        r"(?:为什么|why)\s*(?:报错|出错|fails?|crash|不工作|doesn.t work)",
    ]),
    (IntentType.CODE_REVIEW, [
        r"(?:审查|review|检查|check)\s*(?:代码|code)",
        r"(?:代码质量|code quality|best practice|最佳实践)",
    ]),
    (IntentType.CODE_TEST, [
        r"(?:测试|test|写测试|单元测试|unit test)",
        r"(?:write|generate|create)\s+(?:tests?|unit tests?)",
    ]),
    (IntentType.SEARCH, [
        r"(?:搜索|查找|search|find|look up|查一下)",
    ]),
    (IntentType.REFLECT, [
        r"(?:反思|自我评估|回顾会话|质量评估|reflect|进化趋势|evolution.*trend)",
        r"(?:推理审查|reasoning.*audit|反思.*状态|反思.*报告|reflection.*status)",
    ]),
    (IntentType.SELF_IMPROVE, [
        r"(?:自我改进|self.?improv|improve yourself|学习|learn from|优化自己|自我进化|evolve)",
    ]),
    (IntentType.MEMORY_AGENT, [
        r"(?:记忆管理|memory.?agent|记忆.*智能体|记忆.*状态|memory.*status|记忆.*路由|memory.*rout)",
        r"(?:记忆.*维护|memory.*mainten|错误.*注册|error.*registry|记忆.*分析)",
    ]),
    (IntentType.MEMORY_MANAGE, [
        r"(?:记住|remember|forget|忘记|recall|回忆|记忆|memory)",
    ]),
    (IntentType.SKILL_DESCRIBE, [
        r"(?:技能|能力|擅长|skills?|capabilities?|what can you do|你会什么|你能做什么|描述.*技能)",
    ]),
    (IntentType.META_MINE, [
        r"(?:元知识|元经验|meta.?knowledge|meta.?learn|提炼.*规律|挖掘.*知识|反思|总结经验)",
    ]),
    (IntentType.ORCHESTRATE, [
        r"(?:编排|协作|多智能体|子智能体|orchestrat|multi.?agent|sub.?agent|团队|分工|拆解.*任务)",
    ]),
    (IntentType.SYSTEM_COMMAND, [
        r"(?:运行|执行|run|execute|shell|terminal|命令|command)\s",
    ]),
]


def _rule_classify(text: str) -> Optional[Intent]:
    text_lower = text.lower()
    best_match = None
    best_confidence = 0.0
    
    # Track matched positions to avoid overlapping matches
    matched_positions = set()
    
    for intent_type, patterns in _PATTERNS:
        for i, pattern in enumerate(patterns):
            # Use re.finditer to get all matches with positions
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                # Check if this match overlaps with any previous match
                if not any(pos[0] < end and pos[1] > start for pos in matched_positions):
                    matched_positions.add((start, end))
                    # Assign confidence based on pattern specificity
                    confidence = 0.7
                    if intent_type in [IntentType.CODE_GENERATE, IntentType.CODE_MODIFY]:
                        confidence = 0.8
                    elif intent_type in [IntentType.CODE_EXPLAIN, IntentType.CODE_DEBUG]:
                        confidence = 0.75
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = Intent(
                            type=intent_type,
                            confidence=confidence,
                            description=f"Rule-matched: {intent_type}",
                        )
    
    return best_match


# ======================================================================
# LLM-based deep understanding
# ======================================================================

INTENT_PROMPT = """You are an intent classifier for a Python coding agent.
Analyse the user message and return a JSON object with these fields:
- type: one of [code_generate, code_modify, code_explain, code_debug, code_review, code_test, question, search, self_improve, memory_manage, system_command, skill_describe, meta_mine, orchestrate, memory_agent, reflect, conversation, unknown]
- confidence: float 0-1
- description: one-line summary of what the user wants
- language: programming language (default "python")
- parameters: dict of extracted parameters (e.g. {{"function_name": "sort_list", "input_type": "list"}})
- sub_tasks: list of sub-task descriptions if the request is complex

Respond ONLY with valid JSON, no markdown fences.

User message: {message}"""


class IntentUnderstanding:
    """Understands and classifies user intent."""

    def __init__(self, model_loader=None):
        self._model = model_loader

    def classify(self, user_message: str, context: Optional[List[Dict]] = None) -> Intent:
        """Classify the user's intent.

        First tries rule-based matching; if confidence is low or ambiguous,
        falls back to LLM-based classification.
        """
        # Fast path: rule-based
        rule_intent = _rule_classify(user_message)
        if rule_intent and rule_intent.confidence >= 0.7:
            logger.info(f"Intent (rule): {rule_intent.type} ({rule_intent.confidence})")
            # Enhance with LLM if model is available
            if self._model:
                return self._llm_enhance(user_message, rule_intent, context)
            return rule_intent

        # Slow path: LLM-based
        if self._model:
            return self._llm_classify(user_message, context)

        # No model available — best-effort rule match or unknown
        if rule_intent:
            return rule_intent
        return Intent(
            type=IntentType.CONVERSATION,
            confidence=0.3,
            description="Could not determine intent without model",
        )

    def _llm_classify(self, message: str, context: Optional[List[Dict]] = None) -> Intent:
        """Use the LLM for deep intent classification."""
        messages = []
        if context:
            messages.extend(context[-4:])  # Last few turns for context
        messages.append({
            "role": "user",
            "content": INTENT_PROMPT.format(message=message),
        })

        try:
            response = self._model.generate(
                messages,
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for classification
            )
            intent = self._parse_intent_json(response)
            logger.info(f"Intent (LLM): {intent.type} ({intent.confidence})")
            return intent
        except Exception as e:
            logger.error(f"LLM intent classification failed: {e}")
            return Intent(type=IntentType.UNKNOWN, confidence=0.1,
                          description=f"Classification error: {e}")

    def _llm_enhance(self, message: str, rule_intent: Intent,
                     context: Optional[List[Dict]] = None) -> Intent:
        """Enhance a rule-classified intent with LLM-extracted parameters."""
        messages = [{
            "role": "user",
            "content": INTENT_PROMPT.format(message=message),
        }]
        try:
            response = self._model.generate(messages, max_new_tokens=512, temperature=0.1)
            llm_intent = self._parse_intent_json(response)
            # Merge: keep higher confidence, combine parameters
            if llm_intent.confidence >= rule_intent.confidence:
                llm_intent.parameters = {**rule_intent.parameters, **llm_intent.parameters}
                return llm_intent
            else:
                rule_intent.parameters = {**rule_intent.parameters, **llm_intent.parameters}
                rule_intent.sub_tasks = llm_intent.sub_tasks or rule_intent.sub_tasks
                return rule_intent
        except Exception:
            return rule_intent

    def _parse_intent_json(self, text: str) -> Intent:
        """Parse LLM response into an Intent object."""
        try:
            data = parse_json_response(text)
            if not isinstance(data, dict):
                raise json.JSONDecodeError("Expected JSON object", text, 0)
            # Validate intent type against enum
            raw_type = data.get("type", IntentType.UNKNOWN)
            try:
                intent_type = IntentType(raw_type)
            except ValueError:
                intent_type = IntentType.UNKNOWN
            return Intent(
                type=intent_type,
                confidence=float(data.get("confidence", 0.5)),
                description=data.get("description", ""),
                language=data.get("language", "python"),
                parameters=data.get("parameters", {}),
                sub_tasks=data.get("sub_tasks", []),
            )
        except json.JSONDecodeError:
            logger.warning("Could not parse intent JSON: %s", text[:200])
            return Intent(type=IntentType.UNKNOWN, confidence=0.2,
                          description="Failed to parse LLM intent response")

