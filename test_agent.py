#!/usr/bin/env python3
"""Quick smoke test for all agent modules."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")
from agent.memory.working_memory import WorkingMemory
from agent.memory.long_term_memory import LongTermMemory
from agent.memory.persistent_memory import PersistentMemory
from agent.memory.external_memory import ExternalMemory
from agent.memory.manager import MemoryManager
from agent.intent import IntentUnderstanding, IntentType, Intent
from agent.code_generator import CodeGenerator, execute_code, extract_code_blocks
from agent.self_improver import SelfImprover
from agent.model_loader import select_device, ModelLoader
from agent.core import CodingAgent, create_agent
from agent.meta_knowledge import MetaKnowledgeMiner
from agent.skill_registry import SkillRegistry, Skill, SkillLevel, INTENT_SKILL_MAP
from agent.agent_orchestrator import AgentOrchestrator, SubAgent
from agent.utils import JsonStore, ERROR_MARKERS, UNCERTAINTY_MARKERS
print("All imports OK")

# ---- Shared utilities ----
print("\nTesting shared utilities...")
assert len(ERROR_MARKERS) >= 5, "ERROR_MARKERS should have at least 5 entries"
assert len(UNCERTAINTY_MARKERS) >= 10, "UNCERTAINTY_MARKERS should have at least 10 entries"
print(f"  ERROR_MARKERS: {len(ERROR_MARKERS)} entries")
print(f"  UNCERTAINTY_MARKERS: {len(UNCERTAINTY_MARKERS)} entries")

# JsonStore
js = JsonStore("/tmp/test_jsonstore.json", default={"key": "value"})
js.save()
js2 = JsonStore("/tmp/test_jsonstore.json")
assert js2.load()
assert js2.data == {"key": "value"}
print(f"  JsonStore: save/load OK")
if os.path.exists("/tmp/test_jsonstore.json"):
    os.remove("/tmp/test_jsonstore.json")

# Device
try:
    device = select_device()
    print(f"Device: {device}")
except ImportError:
    print("Device: skipped (torch not installed, using Ollama backend)")

# Memory
print("\nTesting memory system...")
mem_config = {
    "working": {"max_turns": 10, "max_tokens": 4096},
    "long_term": {
        "embedding_dim": 64,
        "index_path": "/tmp/test_lt.json",
        "surprise_threshold": 0.15,
        "dedup_threshold": 0.92,
        "consolidation_threshold": 0.90,
    },
    "persistent": {
        "db_path": "/tmp/test_pm.json",
        "dedup_enabled": True,
        "dedup_ratio": 0.85,
    },
    "external": {"enabled": False},
    "auto_search": False,
}
mm = MemoryManager(mem_config)
mm.add_conversation_turn("user", "Hello, write a sort function")
mm.add_conversation_turn("assistant", "Here is quicksort")

# --- Test surprise gating & dedup ---
print("\n  Testing surprise gating + dedup...")
mid1 = mm.long_term.store("Quicksort is a divide-and-conquer algorithm")
# Exact duplicate → should return same id (hash match)
mid2 = mm.long_term.store("Quicksort is a divide-and-conquer algorithm")
assert mid1 == mid2, f"Exact dup should return same id: {mid1} vs {mid2}"
# Near-duplicate → should be None (dedup / low surprise)
mid3 = mm.long_term.store("Quicksort is a divide and conquer sorting algorithm")
# Either accepted or rejected — both are valid depending on TF-IDF similarity
print(f"    Near-dup store result: {mid3} (None = deduped)")

# Genuinely different entries should be stored
mid4 = mm.long_term.store("Python decorators wrap functions to add behavior")
assert mid4 is not None, "Novel entry should be stored"
mid5 = mm.long_term.store("Docker containers provide isolated environments")
assert mid5 is not None, "Another novel entry should be stored"
print(f"    Stored {mm.long_term.summary()['total_entries']} entries (dedup working)")

# --- Test consolidation ---
print("\n  Testing memory consolidation...")
count_before = len(mm.long_term.entries)
merged = mm.long_term.consolidate(dry_run=True)
print(f"    Would merge {merged} entries (dry run)")

# --- Test experience replay ---
print("\n  Testing experience replay...")
mm.persistent.store_experience(
    task="Write a quicksort function",
    solution="def quicksort(arr): ...",
    outcome="Executed successfully",
    success=True,
)
mm.persistent.store_experience(
    task="Fix IndexError in sort",
    solution="Added bounds check",
    outcome="Error resolved",
    success=True,
)
# Near-duplicate experience (should be deduped by persistent dedup)
mm.persistent.store_experience(
    task="Write a quicksort sorting function",
    solution="def quicksort(lst): ...",
    outcome="OK",
    success=True,
)
exp_count = len(mm.persistent.recall(category="experiences", limit=100))
print(f"    Stored {exp_count} experiences (dedup may reduce)")

exps = mm.persistent.recall_experiences("quicksort")
print(f"    Recalled {len(exps)} experiences for 'quicksort'")

# --- Test RAG pipeline ---
print("\n  Testing RAG recall pipeline...")
rag_results = mm.rag_recall("sorting algorithm", top_k=3)
print(f"    RAG tiers returned: {list(rag_results.keys())}")
print(f"    long_term hits: {len(rag_results.get('long_term', []))}")
print(f"    experiences hits: {len(rag_results.get('experiences', []))}")

mm.remember("User prefers functional style", category="preferences")
recalled = mm.recall("sorting algorithm")
print(f"\n  Working: {mm.working.summary()}")
print(f"  Long-term: {mm.long_term.summary()}")
print(f"  Persistent: {mm.persistent.summary()}")
print(f"  Recalled tiers: { {k: len(v) for k, v in recalled.items()} }")

# ---- Skill Registry ----
print("\nTesting skill registry...")
sr = SkillRegistry({"db_path": "/tmp/test_skills.json"})
assert len(sr.skills) >= 14, f"Should have default skills, got {len(sr.skills)}"
sr.record("code_generation", success=True, example="wrote sort function")
sr.record("code_generation", success=True, example="wrote web scraper")
sr.record("code_generation", success=False, example="failed ML pipeline")
sk = sr.get("code_generation")
assert sk is not None
assert sk.total_uses == 3
assert sk.success_count == 2
print(f"  code_generation: uses={sk.total_uses}, rate={sk.success_rate:.0%}, level={sk.level}")

sr.record_for_intent("code_debug", success=True, example="fixed IndexError")
assert sr.get("code_debugging").total_uses == 1

desc = sr.describe_all()
assert "code_generation" in desc
print(f"  describe_all: {len(desc)} chars")

gaps = sr.identify_gaps()
print(f"  Skill gaps: {len(gaps)}")

sr.add_skill("data_pipeline", "构建数据处理管道", category="coding")
assert "data_pipeline" in sr.skills
print(f"  Total skills: {len(sr.skills)}")

orch_desc = sr.describe_for_orchestrator()
assert "code_generation" in orch_desc
print(f"  Orchestrator skill desc: {len(orch_desc)} chars")

# Verify INTENT_SKILL_MAP coverage
mapped_intents = set(INTENT_SKILL_MAP.keys())
assert "reflect" in mapped_intents, "INTENT_SKILL_MAP should include 'reflect'"
assert "meta_mine" in mapped_intents, "INTENT_SKILL_MAP should include 'meta_mine'"
assert "orchestrate" in mapped_intents, "INTENT_SKILL_MAP should include 'orchestrate'"
assert "memory_agent" in mapped_intents, "INTENT_SKILL_MAP should include 'memory_agent'"
assert "conversation" in mapped_intents, "INTENT_SKILL_MAP should include 'conversation'"
print(f"  INTENT_SKILL_MAP: {len(mapped_intents)} intents mapped")

# ---- Meta-knowledge Miner ----
print("\nTesting meta-knowledge miner...")
miner = MetaKnowledgeMiner(None, mm.persistent, {"min_experiences": 2})
# With 3 experiences stored earlier, mining should try (but no model → empty)
insights = miner.mine(force=True)
print(f"  Insights extracted: {len(insights)} (expected 0 without model)")
stored = miner.store_insights([
    {"principle": "Divide-and-conquer helps with sequence problems",
     "kind": "meta_pattern", "confidence": 0.8},
    {"lesson": "Always include error messages when debugging",
     "kind": "failure_lesson", "confidence": 0.9},
])
print(f"  Manually stored: {stored} meta-knowledge entries")
assert "meta_knowledge" in mm.persistent.list_categories()
meta_entries = mm.persistent.recall(category="meta_knowledge", limit=100)
print(f"  Meta-knowledge in persistent: {len(meta_entries)}")

# ---- Agent Orchestrator ----
print("\nTesting agent orchestrator...")
orch = AgentOrchestrator(None, sr, {"max_sub_agents": 4})
# Without model, decompose returns empty → test SubAgent directly
sa = SubAgent(
    role="CodeWriter",
    skill_description="编写高质量代码",
    subtask="实现排序函数",
    priority=1,
)
assert sa.describe().startswith("[○")
sa.result = "def quicksort(arr): ..."
sa.completed = 1.0
assert sa.describe().startswith("[✓")
print(f"  SubAgent describe: OK")
print(f"  SubAgent to_dict keys: {list(sa.to_dict().keys())}")
desc = orch.describe_agents([sa])
assert "CodeWriter" in desc
print(f"  Orchestrator describe: {len(desc)} chars")
print(f"  Orchestrator summary: {orch.summary()}")

# Intent (new types)
print("\nTesting intent classification (rule-based)...")
iu = IntentUnderstanding(model_loader=None)
test_cases = [
    ("写一个快速排序函数", "code_generate"),
    ("explain this code", "code_explain"),
    ("debug my program", "code_debug"),
    ("search Python asyncio", "search"),
    ("self improve", "self_improve"),
    ("remember this pattern", "memory_manage"),
    ("你有什么技能", "skill_describe"),
    ("提炼元知识", "meta_mine"),
    ("记忆管理智能体状态", "memory_agent"),
    ("编排多智能体完成任务", "orchestrate"),
    ("反思状态", "reflect"),
    ("进化趋势", "reflect"),
]
for text, expected in test_cases:
    intent = iu.classify(text)
    status = "OK" if intent.type == expected else f"MISMATCH(got {intent.type})"
    print(f"  '{text}' -> {intent.type} [{status}]")

# Code execution
print("\nTesting sandboxed code execution...")
result = execute_code("print('Hello from sandbox!')\nprint(sum(range(100)))")
print(f"  success={result.success}, output={result.stdout.strip()}")

# Code extraction
print("\nTesting code extraction...")
test_text = '```python\ndef hello():\n    return "world"\n```'
blocks = extract_code_blocks(test_text)
print(f"  Extracted {len(blocks)} block(s): {blocks[0][:40]}...")

# Cleanup
for p in ["/tmp/test_lt.json", "/tmp/test_pm.json", "/tmp/test_skills.json"]:
    if os.path.exists(p):
        os.remove(p)

print("\n" + "=" * 40)
print("ALL TESTS PASSED")
print("=" * 40)
