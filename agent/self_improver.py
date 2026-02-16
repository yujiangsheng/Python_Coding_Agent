"""
self_improver.py — Autonomous self-improvement engine.

The agent can analyse its own source code, identify weaknesses,
generate improvements, test them, and apply changes. Each improvement
cycle is logged for traceability and rollback.
"""

import ast
import difflib
import json
import logging
import os
import shutil
import time
from typing import Dict, Any, List, Optional, Tuple

from agent.code_generator import execute_code, extract_all_code
from agent.utils import parse_json_response, strip_code_fences

logger = logging.getLogger(__name__)


# ======================================================================
# Improvement record
# ======================================================================

class ImprovementRecord:
    """One atomic self-improvement attempt."""

    def __init__(self, target_file: str, description: str):
        self.id = f"imp_{int(time.time())}_{os.getpid()}"
        self.target_file = target_file
        self.description = description
        self.original_code: str = ""
        self.proposed_code: str = ""
        self.diff: str = ""
        self.test_result: Optional[dict] = None
        self.applied: bool = False
        self.confidence: float = 0.0
        self.timestamp: float = time.time()
        self.rollback_path: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target_file": self.target_file,
            "description": self.description,
            "diff": self.diff,
            "test_result": self.test_result,
            "applied": self.applied,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


# ======================================================================
# Self-improvement engine
# ======================================================================

ANALYSIS_PROMPT = """You are reviewing your own source code as an AI coding agent.
Analyse the following module and suggest specific improvements.

IMPORTANT: Do NOT remove or modify any function/method marked with "DO NOT REMOVE".
Keep all existing public methods intact — only add or improve implementations.

Focus on:
1. Code quality, efficiency, and robustness
2. Better error handling and edge cases
3. Smarter algorithms or data structures
4. Better prompts for code generation
5. Missing functionality that would make you more capable

For each suggestion, provide:
- description: what to improve
- priority: high/medium/low
- confidence: 0-1 that the change will improve things
- target_function: the function or class name to modify
- new_code: the specific code modification (provide complete functions/classes)

Module path: {file_path}
Module code:
```python
{code}
```

You MUST respond with ONLY a valid JSON array. Do NOT include any explanation,
markdown fences, or text outside the JSON. The output must start with [ and end with ].

Example of the EXACT output format:
[{{"description": "Add input validation", "priority": "high", "confidence": 0.8, "target_function": "process", "new_code": "def process(data):\\n    if not data:\\n        raise ValueError()\\n    return data"}}]

Output the JSON array now: /no_think"""


VALIDATION_PROMPT = """You are validating a proposed code change to yourself (an AI agent).
Analyse whether this change is safe and beneficial.

Original code:
```python
{original}
```

Proposed change:
```python
{proposed}
```

Change description: {description}

Evaluate:
1. Does the change introduce any bugs?
2. Does it break any existing functionality?
3. Is it actually an improvement?
4. What's the risk level? (low/medium/high)

Return JSON: {{"safe": true/false, "beneficial": true/false, "risk": "low/medium/high", "reasoning": "..."}}
Respond with ONLY the JSON object, no markdown fences or extra text. /no_think"""


class SelfImprover:
    """Autonomous self-improvement engine for the coding agent."""

    def __init__(self, model_loader, config: dict, project_root: str,
                 memory_agent=None, reflection_agent=None):
        self._model = model_loader
        self.config = config
        self.project_root = project_root
        self.memory_agent = memory_agent
        self.reflection_agent = reflection_agent
        self.log_dir = os.path.join(project_root, config.get("log_dir", "data/improvements"))
        self.min_confidence = config.get("min_confidence", 0.7)
        self.max_iterations = config.get("max_iterations", 5)
        self.do_backup = config.get("backup", True)
        self.history: List[ImprovementRecord] = []
        self._load_history()

    # ------------------------------------------------------------------
    # Core improvement cycle
    # ------------------------------------------------------------------

    def run_improvement_cycle(self, target_files: Optional[List[str]] = None) -> List[ImprovementRecord]:
        """Run one full self-improvement cycle.

        1. Analyse own source files
        2. Generate improvement proposals
        3. **Memory Agent check**: block proposals that repeat past failures
        4. Validate proposals
        5. Apply safe, high-confidence changes
        6. Test and rollback if needed
        7. **Record** results in memory agent
        """
        if target_files is None:
            target_files = self._get_agent_source_files()

        all_records = []
        iteration = 0

        # Consult reflection agent for evolution goals to prioritize
        evolution_goals: List[str] = []
        if self.reflection_agent:
            try:
                goals = self.reflection_agent.evolution_goals(limit=5)
                evolution_goals = [g.get("goal", "") for g in goals if g.get("goal")]
                if evolution_goals:
                    logger.info(
                        f"Self-improvement: {len(evolution_goals)} evolution goals "
                        f"from reflection agent"
                    )
            except Exception as e:
                logger.debug(f"Could not fetch evolution goals: {e}")

        for fpath in target_files:
            if iteration >= self.max_iterations:
                break

            logger.info(f"Self-improvement: analysing {fpath}")

            # Fetch exploration suggestions from memory agent
            if self.memory_agent:
                explorations = self.memory_agent.suggest_exploration(fpath)
                if explorations:
                    logger.info(
                        f"Memory Agent suggests {len(explorations)} new directions "
                        f"for {os.path.basename(fpath)}"
                    )

            suggestions = self._analyse_module(fpath)

            # Boost confidence for suggestions aligned with evolution goals
            if evolution_goals and suggestions:
                for s in suggestions:
                    desc = s.get("description", "").lower()
                    for goal in evolution_goals:
                        if any(kw in desc for kw in goal.lower().split() if len(kw) > 2):
                            old_conf = s.get("confidence", 0)
                            s["confidence"] = min(1.0, old_conf + 0.15)
                            logger.info(
                                f"Boosted suggestion '{s.get('description','')[:60]}' "
                                f"confidence {old_conf:.2f} -> {s['confidence']:.2f} "
                                f"(matched evolution goal: {goal[:60]})"
                            )
                            break

            for suggestion in suggestions:
                if iteration >= self.max_iterations:
                    break
                if suggestion.get("confidence", 0) < self.min_confidence:
                    logger.debug(f"Skipping low-confidence suggestion: {suggestion.get('description', '?')}")
                    continue

                record = self._apply_suggestion(fpath, suggestion)
                all_records.append(record)
                iteration += 1

        # Save history
        self._save_history()
        return all_records

    def _analyse_module(self, file_path: str) -> List[dict]:
        """Analyse a source module and return improvement suggestions."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []

        prompt = ANALYSIS_PROMPT.format(file_path=file_path, code=code)
        try:
            response = self._model.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=2048,
                temperature=0.3,
            )
            logger.debug(
                "Raw LLM response for %s (first 300 chars): %s",
                os.path.basename(file_path), response[:300],
            )
            suggestions = parse_json_response(response)
            if isinstance(suggestions, list):
                # Filter out non-dict items that slipped through
                suggestions = [s for s in suggestions if isinstance(s, dict)]
                logger.info(f"Got {len(suggestions)} improvement suggestions for {file_path}")
                return suggestions
            logger.warning("Expected JSON array, got: %s", type(suggestions).__name__)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(
                f"Failed to parse improvement suggestions: {e}\n"
                f"  Response preview: {response[:200] if 'response' in dir() else '(no response)'}"
            )
        return []

    def _apply_suggestion(self, file_path: str, suggestion: dict) -> ImprovementRecord:
        """Attempt to apply a single improvement suggestion."""
        record = ImprovementRecord(
            target_file=file_path,
            description=suggestion.get("description", "Unknown improvement"),
        )
        record.confidence = suggestion.get("confidence", 0.5)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_code = f.read()
            record.original_code = original_code

            # Generate the improved code
            proposed_code = self._generate_improvement(file_path, original_code, suggestion)
            if not proposed_code:
                record.test_result = {"error": "No code generated"}
                self.history.append(record)
                return record
            record.proposed_code = proposed_code

            # Generate diff
            record.diff = self._make_diff(original_code, proposed_code, file_path)

            # --- Memory Agent pre-check: block repeated mistakes ---
            if self.memory_agent:
                check = self.memory_agent.pre_improvement_check(
                    file_path, record.description, record.diff,
                )
                if not check["safe"]:
                    record.test_result = {
                        "error": f"Memory Agent blocked: {check.get('suggestion', 'repeated failure pattern')}",
                        "risk_level": check["risk_level"],
                    }
                    logger.warning(
                        f"Memory Agent blocked improvement: {record.description} "
                        f"(risk={check['risk_level']})"
                    )
                    self.history.append(record)
                    return record

            # Validate the change
            is_valid = self._validate_change(original_code, proposed_code, suggestion)
            if not is_valid:
                record.test_result = {"error": "Validation rejected the change"}
                self.history.append(record)
                return record

            # Syntax check
            if not self._syntax_check(proposed_code):
                record.test_result = {"error": "Syntax error in proposed code"}
                self.history.append(record)
                return record

            # Structural integrity check: ensure no "DO NOT REMOVE" methods vanished
            if not self._structural_integrity_check(original_code, proposed_code):
                record.test_result = {"error": "Structural integrity: protected methods removed"}
                logger.warning(f"Blocked: proposed code removes protected methods in {file_path}")
                self.history.append(record)
                return record

            # Backup
            if self.do_backup:
                record.rollback_path = self._backup_file(file_path)

            # Apply the change
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(proposed_code)
            record.applied = True

            # Test by importing (basic smoke test)
            test_ok = self._smoke_test(file_path)
            record.test_result = {"success": test_ok}

            if not test_ok:
                # Rollback
                logger.warning(f"Smoke test failed — rolling back {file_path}")
                self._rollback(file_path, record.rollback_path)
                record.applied = False
                record.test_result = {"error": "Smoke test failed, rolled back"}
                # Record failure in memory agent
                if self.memory_agent:
                    self.memory_agent.record_improvement_result(
                        file_path, record.description, record.diff,
                        success=False, error_reason="Smoke test failed",
                    )
            else:
                logger.info(f"✓ Applied improvement to {file_path}: {record.description}")
                # Record success in memory agent
                if self.memory_agent:
                    self.memory_agent.record_improvement_result(
                        file_path, record.description, record.diff,
                        success=True,
                    )

        except Exception as e:
            record.test_result = {"error": str(e)}
            logger.error(f"Improvement failed: {e}")
            # Rollback if we have a backup
            if record.rollback_path and os.path.exists(record.rollback_path):
                self._rollback(file_path, record.rollback_path)
                record.applied = False
            # Record failure in memory agent
            if self.memory_agent:
                self.memory_agent.record_improvement_result(
                    file_path, record.description, record.diff or "",
                    success=False, error_reason=str(e),
                )

        self.history.append(record)
        return record

    def _generate_improvement(self, file_path: str, original_code: str,
                              suggestion: dict) -> Optional[str]:
        """Use LLM to generate the improved version of the code."""
        new_code = suggestion.get("new_code", "")
        target_fn = suggestion.get("target_function", "")

        if new_code and target_fn:
            # Surgical replacement: replace just the target function/class
            return self._surgical_replace(original_code, target_fn, new_code)

        # Full rewrite approach
        prompt = (
            f"Apply this improvement to the module.\n\n"
            f"Improvement: {suggestion.get('description', '')}\n\n"
            f"Original code:\n```python\n{original_code}\n```\n\n"
            f"Return the COMPLETE updated module code in ```python``` fences. "
            f"Preserve all existing functionality. Only make the described improvement."
        )
        try:
            response = self._model.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=4096,
                temperature=0.2,
            )
            code = extract_all_code(response)
            return code if code.strip() else None
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None

    def _surgical_replace(self, original: str, target_name: str, new_code: str) -> str:
        """Replace a specific function or class in the source code, preserving indentation."""
        try:
            tree = ast.parse(original)
        except SyntaxError:
            return original

        lines = original.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == target_name:
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1

                    # Detect the original indentation level
                    original_indent = len(lines[start]) - len(lines[start].lstrip())

                    # Detect the indentation of the new_code
                    new_code_lines = new_code.split("\n")
                    non_empty = [l for l in new_code_lines if l.strip()]
                    new_indent = len(non_empty[0]) - len(non_empty[0].lstrip()) if non_empty else 0

                    # Re-indent new_code to match the original indentation
                    indent_diff = original_indent - new_indent
                    adjusted = []
                    for line in new_code_lines:
                        if line.strip():
                            if indent_diff > 0:
                                adjusted.append(" " * indent_diff + line)
                            elif indent_diff < 0:
                                adjusted.append(line[-indent_diff:] if line[:abs(indent_diff)].strip() == "" else line)
                            else:
                                adjusted.append(line)
                        else:
                            adjusted.append(line)

                    new_lines = lines[:start] + adjusted + lines[end:]
                    result = "\n".join(new_lines)
                    # Verify the replacement didn't break syntax
                    try:
                        ast.parse(result)
                    except SyntaxError as e:
                        logger.warning(
                            f"Surgical replacement of '{target_name}' produced "
                            f"invalid syntax: {e} — returning original"
                        )
                        return original
                    return result

        # Target not found — return original unchanged
        logger.warning(f"Target '{target_name}' not found in source")
        return original

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_change(self, original: str, proposed: str, suggestion: dict) -> bool:
        """Use LLM to validate a proposed change is safe."""
        prompt = VALIDATION_PROMPT.format(
            original=original[:3000],
            proposed=proposed[:3000],
            description=suggestion.get("description", ""),
        )
        try:
            response = self._model.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=512,
                temperature=0.1,
            )
            result = parse_json_response(response)
            safe = result.get("safe", False)
            beneficial = result.get("beneficial", False)
            risk = result.get("risk", "high")

            logger.info(f"Validation: safe={safe}, beneficial={beneficial}, risk={risk}")
            return safe and beneficial and risk != "high"
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def _syntax_check(self, code: str) -> bool:
        """Check if the code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax error: {e}")
            return False

    def _smoke_test(self, file_path: str) -> bool:
        """Basic sanity test: try importing the module."""
        # Create a test script that imports the module
        module_path = os.path.relpath(file_path, self.project_root)
        module_name = module_path.replace(os.sep, ".").replace(".py", "")

        test_code = f"""
import sys
sys.path.insert(0, {repr(self.project_root)})
try:
    import importlib
    mod = importlib.import_module({repr(module_name)})
    print("IMPORT_OK")
except Exception as e:
    print(f"IMPORT_FAIL: {{e}}")
    sys.exit(1)
"""
        result = execute_code(test_code, timeout=15)
        return result.success and "IMPORT_OK" in result.stdout

    # ------------------------------------------------------------------
    # Backup / rollback
    # ------------------------------------------------------------------

    def _backup_file(self, file_path: str) -> str:
        """Create a timestamped backup of a file."""
        os.makedirs(self.log_dir, exist_ok=True)
        basename = os.path.basename(file_path)
        backup_name = f"{basename}.{int(time.time())}.bak"
        backup_path = os.path.join(self.log_dir, backup_name)
        shutil.copy2(file_path, backup_path)
        return backup_path

    def _rollback(self, file_path: str, backup_path: str):
        """Restore a file from backup."""
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            logger.info(f"Rolled back {file_path} from {backup_path}")

    @staticmethod
    def _structural_integrity_check(original: str, proposed: str) -> bool:
        """Ensure no 'DO NOT REMOVE' marked methods are deleted in proposed code.

        Scans original for ``def <name>`` that appear near a DO NOT REMOVE
        comment, then verifies they still exist in proposed code.
        """
        import re
        # Find all method names that follow a DO NOT REMOVE marker within 3 lines
        protected: set = set()
        lines = original.splitlines()
        for i, line in enumerate(lines):
            if "DO NOT REMOVE" in line:
                # Scan next 5 lines for def statements
                for j in range(i, min(i + 6, len(lines))):
                    m = re.match(r'\s+def (\w+)\s*\(', lines[j])
                    if m:
                        protected.add(m.group(1))
        if not protected:
            return True  # nothing to protect
        for name in protected:
            pattern = rf'\bdef {name}\s*\('
            if not re.search(pattern, proposed):
                logger.warning(f"Protected method '{name}' missing in proposed code")
                return False
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_agent_source_files(self) -> List[str]:
        """Get all Python source files of the agent itself."""
        agent_dir = os.path.join(self.project_root, "agent")
        files = []
        for root, _, filenames in os.walk(agent_dir):
            for fn in sorted(filenames):
                if fn.endswith(".py") and not fn.startswith("__"):
                    files.append(os.path.join(root, fn))
        return files

    def _make_diff(self, original: str, proposed: str, file_path: str) -> str:
        """Generate a unified diff."""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"a/{os.path.basename(file_path)}",
            tofile=f"b/{os.path.basename(file_path)}",
        )
        return "".join(diff)

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _save_history(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "history.json")
        data = [r.to_dict() for r in self.history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def _load_history(self):
        path = os.path.join(self.log_dir, "history.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data:
                    rec = ImprovementRecord(
                        target_file=entry.get("target_file", ""),
                        description=entry.get("description", ""),
                    )
                    rec.id = entry.get("id", rec.id)
                    rec.diff = entry.get("diff", "")
                    rec.test_result = entry.get("test_result")
                    rec.applied = entry.get("applied", False)
                    rec.confidence = entry.get("confidence", 0.0)
                    rec.timestamp = entry.get("timestamp", 0.0)
                    self.history.append(rec)
                logger.info(f"Loaded {len(self.history)} improvement records")
            except Exception as e:
                logger.error(f"Failed to load improvement history: {e}")

    def get_stats(self) -> dict:
        total = len(self.history)
        applied = sum(1 for r in self.history if r.applied)
        return {
            "total_attempts": total,
            "applied": applied,
            "success_rate": round(applied / total, 2) if total > 0 else 0,
            "log_dir": self.log_dir,
        }

    def summary(self) -> dict:
        stats = self.get_stats()
        stats["summary_text"] = (
            f"Self-Improvement Stats: {stats['applied']}/{stats['total_attempts']} "
            f"applied ({stats['success_rate']*100:.0f}% success rate)"
        )
        return stats
