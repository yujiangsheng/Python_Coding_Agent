#!/usr/bin/env python3
"""
main.py — Entry point for the Python Coding Agent (PyCoder).

Supports three modes:
  1. Interactive REPL:  python main.py
  2. Single query:      python main.py --query "写一个快速排序"
  3. Self-improve:      python main.py --self-improve
"""

import argparse
import logging
import os
import signal
import sys
import yaml
from typing import Callable, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.core import CodingAgent, create_agent

logger = logging.getLogger(__name__)


# ======================================================================
# Interactive REPL
# ======================================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║              PyCoder — Python Coding Agent               ║
║          Powered by Qwen3-Coder-30B (Ollama local)        ║
║          Author & Maintainer: Jiangsheng Yu              ║
║                                                          ║
║  Commands:                                               ║
║    /status      — Show agent status                       ║
║    /memory      — Show memory summary                     ║
║    /skills      — Show skill registry                     ║
║    /meta        — Run meta-knowledge mining               ║
║    /orchestrate — Multi-agent task (enter task next)       ║
║    /improve     — Run self-improvement cycle              ║
║    /memory-agent— Memory agent status & advice            ║
║    /reflect     — Reflection & evolution status            ║
║    /retrospect  — Session retrospective analysis           ║
║    /save        — Save session & memories                 ║
║    /clear       — Clear working memory                    ║
║    /history     — Show interaction history                 ║
║    /help        — Show this help                          ║
║    /quit        — Save and exit                           ║
╚══════════════════════════════════════════════════════════╝
"""


def run_repl(agent: CodingAgent):
    """Run the interactive read-eval-print loop."""
    print(BANNER)
    print("模型正在加载中，请稍候...\n")

    # Trigger lazy loading of the model
    _ = agent.model
    print(f"\n{agent.status()}\n")
    print("输入你的编程需求（输入 /help 查看命令，/quit 退出）：\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            _graceful_exit(agent)
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if _handle_command(agent, user_input):
                continue
            else:
                break  # /quit

        # Chat
        try:
            response = agent.chat(user_input)
            print(f"\nPyCoder > {response}\n")
        except KeyboardInterrupt:
            print("\n[中断] 当前请求已取消\n")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n[错误] {e}\n")


def _cmd_status(agent: CodingAgent):
    print(f"\n{agent.status()}\n")


def _cmd_memory(agent: CodingAgent):
    summary = agent.memory.summary()
    print(f"\n{yaml.dump(summary, allow_unicode=True)}\n")


def _cmd_improve(agent: CodingAgent):
    print("\n正在运行自我改进循环...\n")
    response = agent.chat("请进行自我改进")
    print(f"\nPyCoder > {response}\n")


def _cmd_skills(agent: CodingAgent):
    print(f"\n{agent.skills.describe_all()}\n")
    gaps = agent.skills.identify_gaps()
    if gaps:
        print("🔍 需要加强的领域:")
        for g in gaps[:5]:
            print(f"  - {g['skill']} [{g['level']}]: {g['reason']}")
        print()


def _cmd_meta(agent: CodingAgent):
    print("\n正在挖掘元知识...\n")
    response = agent.chat("请提炼元知识和元经验")
    print(f"\nPyCoder > {response}\n")


def _cmd_memory_agent(agent: CodingAgent):
    response = agent.chat("记忆管理智能体状态")
    print(f"\nPyCoder > {response}\n")


def _cmd_reflect(agent: CodingAgent):
    response = agent.chat("反思状态")
    print(f"\nPyCoder > {response}\n")


def _cmd_retrospect(agent: CodingAgent):
    print("\n正在进行会话反思回顾...\n")
    response = agent.chat("回顾会话反思")
    print(f"\nPyCoder > {response}\n")


def _cmd_save(agent: CodingAgent):
    agent.save_session()
    print("\n会话已保存。\n")


def _cmd_clear(agent: CodingAgent):
    agent.memory.working.clear()
    print("\n工作记忆已清除。\n")


def _cmd_history(agent: CodingAgent):
    turns = agent.memory.working.get_full_turns()
    if not turns:
        print("\n暂无交互历史。\n")
        return

    print(f"\n最近 {len(turns)} 条记录：")
    for t in turns[-10:]:
        role = t["role"].upper()
        text = t["content"][:120].replace("\n", " ")
        print(f"  [{role}] {text}{'…' if len(t['content']) > 120 else ''}")
    print()


def _cmd_help(_: CodingAgent):
    print(BANNER)


def _command_registry() -> Dict[str, Callable[[CodingAgent], None]]:
    """集中管理 REPL 命令映射，便于扩展与维护。"""
    return {
        "/status": _cmd_status,
        "/memory": _cmd_memory,
        "/improve": _cmd_improve,
        "/skills": _cmd_skills,
        "/meta": _cmd_meta,
        "/memory-agent": _cmd_memory_agent,
        "/reflect": _cmd_reflect,
        "/retrospect": _cmd_retrospect,
        "/save": _cmd_save,
        "/clear": _cmd_clear,
        "/history": _cmd_history,
        "/help": _cmd_help,
    }


def _handle_orchestrate_command(agent: CodingAgent, raw_cmd: str):
    """处理 /orchestrate 及其可选内联任务参数。"""
    task = raw_cmd[len("/orchestrate"):].strip()
    if not task:
        try:
            task = input("请输入复杂任务描述 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return
    if task:
        print("\n正在编排子智能体...\n")
        response = agent.chat(f"请编排多智能体完成：{task}")
        print(f"\nPyCoder > {response}\n")


def _handle_command(agent: CodingAgent, cmd: str) -> bool:
    """Handle slash commands. Returns False if the loop should exit."""
    raw_cmd = cmd.strip()
    normalized_cmd = raw_cmd.lower()

    if normalized_cmd in ("/quit", "/exit"):
        _graceful_exit(agent)
        return False

    if normalized_cmd.startswith("/orchestrate"):
        _handle_orchestrate_command(agent, raw_cmd)
        return True

    handler = _command_registry().get(normalized_cmd)
    if handler:
        handler(agent)
        return True

    print(f"\n未知命令: {normalized_cmd}  （输入 /help 查看可用命令）\n")

    return True


def _graceful_exit(agent: CodingAgent):
    """Save state and exit gracefully."""
    print("正在保存会话...")
    try:
        agent.save_session()
        print("会话已保存。再见！")
    except Exception as e:
        print(f"保存失败: {e}")


# ======================================================================
# Self-improvement mode
# ======================================================================

def run_self_improve(agent: CodingAgent, iterations: int = 3):
    """Run autonomous self-improvement cycles."""
    print("=" * 60)
    print("PyCoder Self-Improvement Mode")
    print("=" * 60)

    for i in range(1, iterations + 1):
        print(f"\n--- Improvement Cycle {i}/{iterations} ---")
        records = agent.improver.run_improvement_cycle()

        for record in records:
            status = "APPLIED" if record.applied else "SKIPPED"
            print(f"  [{status}] {record.description} (confidence={record.confidence:.2f})")
            if record.diff:
                # Show first few lines of diff
                diff_lines = record.diff.split("\n")[:10]
                for line in diff_lines:
                    print(f"    {line}")

        print(f"\n{agent.improver.summary()}")

    agent.save_session()
    print("\n自我改进完成。会话已保存。")


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyCoder — Python Coding Agent powered by Qwen3-Coder"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to execute (non-interactive mode)",
    )
    parser.add_argument(
        "--self-improve", "-s",
        action="store_true",
        help="Run autonomous self-improvement",
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Number of self-improvement iterations (default: 3)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    args = parser.parse_args()

    # Create agent
    agent = create_agent(config_path=args.config)

    # Handle SIGINT gracefully
    def sigint_handler(sig, frame):
        _graceful_exit(agent)
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    if args.query:
        # Single query mode
        response = agent.chat(args.query)
        print(response)
        agent.save_session()

    elif args.self_improve:
        # Self-improvement mode
        run_self_improve(agent, iterations=args.iterations)

    else:
        # Interactive REPL
        run_repl(agent)


if __name__ == "__main__":
    main()
