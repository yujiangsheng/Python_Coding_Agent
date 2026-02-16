#!/usr/bin/env python3
"""Smoke tests for ReflectionAgent + EvolutionTracker."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP_EVO = "/tmp/test_evolution.json"


def test_imports():
    print("=== Imports ===")
    from agent.reflection_agent import (
        ReflectionAgent, ReflectionRecord, ReflectionLevel,
        QualityScore, EvolutionTracker,
    )
    from agent.intent import IntentType
    assert hasattr(IntentType, "REFLECT")
    print("  All imports OK")


def test_quality_score():
    from agent.reflection_agent import QualityScore
    print("\n=== QualityScore ===")
    qs = QualityScore(correctness=0.9, completeness=0.8, clarity=0.7, efficiency=0.6)
    d = qs.to_dict()
    assert "correctness" in d
    qs2 = QualityScore.from_dict(d)
    assert qs2.correctness == 0.9
    print(f"  to_dict/from_dict: OK")


def test_reflection_record():
    from agent.reflection_agent import ReflectionRecord, ReflectionLevel, QualityScore
    print("\n=== ReflectionRecord ===")
    rec = ReflectionRecord(
        level=ReflectionLevel.RESPONSE,
        quality=QualityScore(correctness=0.8, completeness=0.7,
                             clarity=0.9, efficiency=0.6, overall=0.75),
        strengths=["包含代码"],
        weaknesses=["未处理边界"],
        suggestions=["添加边界检查"],
    )
    d = rec.to_dict()
    assert d["level"] == "response"
    assert d["quality"]["correctness"] == 0.8
    assert rec.timestamp > 0
    print(f"  to_dict: OK, timestamp={rec.timestamp:.0f}")


def test_evolution_tracker():
    from agent.reflection_agent import EvolutionTracker
    print("\n=== EvolutionTracker ===")

    # Clean up first
    if os.path.exists(TMP_EVO):
        os.remove(TMP_EVO)

    et = EvolutionTracker(db_path=TMP_EVO)
    assert len(et.snapshots) == 0

    # Record a few sessions
    et.record_session(
        interaction_count=10, avg_quality=0.6,
        strengths=["代码示例丰富"], weaknesses=["回答偏短"],
        goals=[{"goal": "增加详细解释", "priority": "high"}],
        lessons=["用户偏好详细回答"],
    )
    et.record_session(
        interaction_count=15, avg_quality=0.7,
        strengths=["代码示例丰富", "逻辑清晰"], weaknesses=["边界处理不足"],
        goals=[{"goal": "加强边界检查", "priority": "medium"}],
        lessons=["注意边界条件"],
    )
    et.record_session(
        interaction_count=20, avg_quality=0.8,
        strengths=["代码示例丰富"], weaknesses=["回答偏短"],
        goals=[],
        lessons=["保持进步"],
    )
    assert len(et.snapshots) == 3
    print(f"  Recorded 3 sessions")

    # Trend
    trend = et.get_trend()
    assert trend["sessions_tracked"] == 3
    assert trend["trend"] in ("improving", "stable", "declining")
    print(f"  Trend: {trend['trend']}, avg={trend['avg_quality']:.3f}, recent={trend['recent_avg']:.3f}")

    # Unresolved goals
    goals = et.get_unresolved_goals()
    assert len(goals) >= 1
    print(f"  Unresolved goals: {len(goals)}")

    # Recurring weaknesses
    rw = et.get_recurring_weaknesses(min_count=2)
    assert len(rw) >= 1
    print(f"  Recurring weaknesses: {rw}")

    # Persistence
    et2 = EvolutionTracker(db_path=TMP_EVO)
    assert len(et2.snapshots) == 3
    print(f"  Persistence: OK")

    # Summary
    s = et.summary()
    assert "trend" in s
    print(f"  Summary: {s}")


def test_reflection_agent_rule_based():
    """Test ReflectionAgent instantiation with current simplified API."""
    from agent.reflection_agent import ReflectionAgent
    print("\n=== ReflectionAgent (basic) ===")

    ra = ReflectionAgent(use_llm=False, model=None)
    assert ra._use_llm is False
    assert ra._model is None
    assert ra._evolution is not None
    assert ra._quality_threshold == 0.6
    print("  Instantiation: OK")

    # Cleanup
    if os.path.exists(TMP_EVO):
        os.remove(TMP_EVO)


def test_intent_reflect():
    from agent.intent import IntentUnderstanding, IntentType
    print("\n=== IntentType.REFLECT ===")
    iu = IntentUnderstanding(model_loader=None)

    cases = [
        ("反思状态", "reflect"),
        ("回顾会话反思", "reflect"),
        ("进化趋势", "reflect"),
        ("reflection status", "reflect"),
    ]
    for text, expected in cases:
        intent = iu.classify(text)
        status = "OK" if intent.type == expected else f"MISMATCH(got {intent.type})"
        print(f"  '{text}' -> {intent.type} [{status}]")
        assert intent.type == expected, f"Expected {expected}, got {intent.type}"


if __name__ == "__main__":
    test_imports()
    test_quality_score()
    test_reflection_record()
    test_evolution_tracker()
    test_reflection_agent_rule_based()
    test_intent_reflect()

    print("\n" + "=" * 40)
    print("ALL REFLECTION AGENT TESTS PASSED")
    print("=" * 40)
