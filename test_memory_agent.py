#!/usr/bin/env python3
"""Targeted tests for MemoryAgent and ErrorRegistry (v0.5.0)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.memory_agent import MemoryAgent, ErrorRegistry
from agent.memory.manager import MemoryManager
from agent.intent import IntentType, IntentUnderstanding

TMP_REG = "/tmp/test_error_registry.json"
TMP_LT = "/tmp/test_ma_lt.json"
TMP_PM = "/tmp/test_ma_pm.json"


def cleanup():
    for p in [TMP_REG, TMP_LT, TMP_PM]:
        if os.path.exists(p):
            os.remove(p)


def test_error_registry():
    print("=== ErrorRegistry ===")
    cleanup()
    reg = ErrorRegistry(TMP_REG)
    reg.record_failure("agent/core.py", "Optimise prompt handling", "-old\n+new", "Smoke test failed")
    reg.record_failure("agent/core.py", "Optimise prompt processing", "-old\n+diff", "SyntaxError")
    assert len(reg.entries) == 2
    print(f"  Entries: {len(reg.entries)}")

    sim = reg.find_similar_failures("agent/core.py", "Optimise prompt handling logic")
    assert len(sim) >= 1
    print(f"  Similar found: {len(sim)}")

    appr = reg.get_tried_approaches("agent/core.py")
    assert len(appr) >= 1
    print(f"  Tried approaches: {len(appr)}")

    reg.record_success("agent/core.py", "Optimise prompt handling")
    # record_success appends a new entry with error=None
    assert len(reg.entries) == 3
    assert reg.entries[-1].get("error") is None
    print(f"  record_success: OK")

    s = reg.summary()
    assert "total_entries" in s
    print(f"  Summary: {s}")

    reg.save()
    assert os.path.exists(TMP_REG)
    reg2 = ErrorRegistry(TMP_REG)
    assert len(reg2.entries) == 3
    print(f"  Persistence: OK ({len(reg2.entries)} entries reloaded)")


def test_memory_agent():
    print("\n=== MemoryAgent ===")
    mem_config = {
        "working": {"max_turns": 10, "max_tokens": 4096},
        "long_term": {
            "embedding_dim": 64,
            "index_path": TMP_LT,
            "surprise_threshold": 0.15,
            "dedup_threshold": 0.92,
            "consolidation_threshold": 0.90,
        },
        "persistent": {
            "db_path": TMP_PM,
            "dedup_enabled": True,
            "dedup_ratio": 0.85,
        },
        "external": {"enabled": False},
        "auto_search": False,
    }
    mm = MemoryManager(mem_config)
    ma_cfg = {
        "error_registry_path": TMP_REG,
        "llm_routing": False,
        "max_exploration_suggestions": 3,
    }
    ma = MemoryAgent(model_loader=None, memory_manager=mm, config=ma_cfg)

    tier_result = ma.smart_route("User prefers type hints on everything")
    tier, category = tier_result
    assert tier in ("working", "long_term", "persistent", "external", "unknown"), f"Unexpected: {tier}"
    print(f"  smart_route -> tier={tier}, category={category}")

    stored = ma.route_and_store("Always add docstrings to public functions")
    print(f"  route_and_store -> tier={stored.get('tier')}, stored={stored.get('stored')}")

    chk = ma.pre_improvement_check("agent/core.py", "Optimise prompt handling", "-old\n+new")
    assert not chk["safe"], "Identical diff should be blocked"
    print(f"  pre_improvement_check (same diff): safe={chk['safe']}, risk={chk['risk_level']}")

    sugg = ma.suggest_exploration("agent/core.py")
    assert isinstance(sugg, list)
    print(f"  suggest_exploration: {len(sugg)} suggestions")

    adv = ma.advise_maintenance()
    assert isinstance(adv, list)
    print(f"  advise_maintenance: {len(adv)} advice items")


def test_intent_memory_agent():
    print("\n=== IntentType.MEMORY_AGENT ===")
    assert hasattr(IntentType, "MEMORY_AGENT")
    iu = IntentUnderstanding(model_loader=None)

    intent = iu.classify("记忆管理智能体状态")
    print(f"  '记忆管理智能体状态' -> {intent.type}")
    assert intent.type == "memory_agent", f"Expected memory_agent, got {intent.type}"

    intent2 = iu.classify("memory agent status")
    print(f"  'memory agent status' -> {intent2.type}")
    assert intent2.type == "memory_agent", f"Expected memory_agent, got {intent2.type}"


if __name__ == "__main__":
    try:
        test_error_registry()
        test_memory_agent()
        test_intent_memory_agent()
        print("\n" + "=" * 40)
        print("ALL MEMORY AGENT TESTS PASSED")
        print("=" * 40)
    finally:
        cleanup()
