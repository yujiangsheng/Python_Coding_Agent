"""
code_generator.py — Code generation, execution, and validation.

Generates code via the LLM, executes it in a sandboxed subprocess,
captures output, and validates correctness.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, Any, Optional, Tuple

from agent.exceptions import GenerationError

logger = logging.getLogger(__name__)

__all__ = [
    "CodeGenerator",
    "ExecutionResult",
    "execute_code",
    "extract_code_blocks",
    "extract_all_code",
]


# ======================================================================
# Code extraction helpers
# ======================================================================

def extract_code_blocks(text: str, language: str = "python") -> list:
    """Extract fenced code blocks from LLM output."""
    # Prevent regex DoS by limiting text size
    if len(text) > 100000:
        logger.warning("Very long text detected in extract_code_blocks, may cause performance issues")
        return []
    
    # Validate input
    if not isinstance(text, str):
        logger.warning("Invalid input type for extract_code_blocks, expected str")
        return []
    
    # Match fenced code blocks with more precise pattern
    pattern = rf'```(?:{language})?\s*\n(.*?)```'
    blocks = re.findall(pattern, text, re.DOTALL)
    
    if not blocks:
        # Try to detect raw code (indented blocks or no fences)
        lines = text.strip().split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            # Check if line is part of code block
            if (line.startswith(("import ", "from ", "def ", "class ", "    ", "\t")) or 
                (in_code and line.strip() != "")):
                code_lines.append(line)
                in_code = True
            elif in_code and line.strip() == "":
                code_lines.append(line)
            elif in_code:
                in_code = False
        if code_lines:
            blocks = ["\n".join(code_lines)]
    
    # Additional cleanup: remove leading/trailing empty lines
    cleaned_blocks = []
    for block in blocks:
        lines = block.strip().split("\n")
        # Remove leading empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            cleaned_blocks.append("\n".join(lines))
    
    # Limit number of blocks to prevent excessive processing
    return cleaned_blocks[:10]  # Max 10 code blocks


def extract_all_code(text: str) -> str:
    """Extract and concatenate all code from LLM output."""
    blocks = extract_code_blocks(text)
    return "\n\n".join(blocks) if blocks else text


# ======================================================================
# Execution
# ======================================================================

class ExecutionResult:
    """Result of code execution."""

    def __init__(self, code: str, stdout: str = "", stderr: str = "",
                 returncode: int = -1, duration: float = 0.0,
                 error: Optional[str] = None):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.duration = duration
        self.error = error

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.error

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "duration": round(self.duration, 3),
            "error": self.error,
        }

    def summary(self) -> str:
        if self.success:
            out = self.stdout.strip()
            if len(out) > 500:
                out = out[:500] + "… (truncated)"
            return f"✓ Success ({self.duration:.2f}s)\n{out}" if out else f"✓ Success ({self.duration:.2f}s, no output)"
        else:
            err = self.stderr.strip() or self.error or "Unknown error"
            if len(err) > 500:
                err = err[:500] + "… (truncated)"
            return f"✗ Failed (rc={self.returncode}, {self.duration:.2f}s)\n{err}"


def execute_code(code: str, timeout: int = 30, max_output: int = 10000) -> ExecutionResult:
    """Execute Python code in a sandboxed environment.

    Args:
        code: The Python code to execute
        timeout: Execution timeout in seconds
        max_output: Maximum output size to capture

    Returns:
        ExecutionResult object with status and output
    """
    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        start_time = time.time()
        # Execute with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(temp_file)
        )
        duration = time.time() - start_time
        
        # Check if execution was successful
        success = result.returncode == 0
        
        # Log execution details
        if success:
            logger.debug(f"Code executed successfully. Output length: {len(result.stdout)}")
        else:
            logger.warning(f"Code execution failed. Return code: {result.returncode}")
            logger.debug(f"Error output: {(result.stderr or '')[:200]}")

        return ExecutionResult(
            code=code,
            stdout=result.stdout[:max_output],
            stderr=result.stderr[:max_output],
            returncode=result.returncode,
            error=result.stderr[:max_output] if result.stderr else None,
            duration=duration
        )
        
    except subprocess.TimeoutExpired:
        logger.error(f"Code execution timed out after {timeout} seconds")
        return ExecutionResult(
            code=code,
            error=f"Execution timed out after {timeout} seconds",
            returncode=124,
        )
    except Exception as e:
        logger.error(f"Unexpected error during code execution: {str(e)}")
        return ExecutionResult(code=code, error=str(e), returncode=1)
    finally:
        try:
            os.unlink(temp_file)
        except OSError:
            pass


# ======================================================================
# Code Generator
# ======================================================================

GENERATE_SYSTEM = """You are an expert Python programmer. You write clean, efficient,
well-documented code. Follow these rules:
1. Always include docstrings and type hints.
2. Handle errors gracefully.
3. Write runnable, self-contained code unless building on existing code.
4. If the task requires multiple files, clearly separate them with comments.
5. Use modern Python idioms and best practices.
6. When fixing bugs, explain the root cause and the fix.

Respond with the code inside ```python``` fences."""

DEBUG_SYSTEM = """You are an expert Python debugger.
Given failing code and traceback, output ONLY corrected Python code in a single ```python``` block.
Do not include explanations, markdown text, or analysis outside the code block.
Preserve required public function/class names and signatures."""


class CodeGenerator:
    """Generates, executes, and validates code using the LLM."""

    def __init__(self, model_loader, config: dict):
        self._model = model_loader
        self.timeout = config.get("timeout", 30)
        self.max_output = config.get("max_output_chars", 10000)
        self.history: list = []  # Track generation history

    def generate(self, request: str, context: Optional[list] = None,
                 system: str = GENERATE_SYSTEM) -> Tuple[str, str]:
        """Generate code for a request.

        Returns:
            Tuple of (raw_response, extracted_code)
        """
        if request is None:
            raise ValueError("Request cannot be None")
        if not isinstance(request, str):
            raise TypeError("Request must be a string")
        if not request.strip():
            raise ValueError("Request cannot be empty")

        messages = [{"role": "system", "content": system}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": request})

        try:
            response = self._model.generate(messages)
        except Exception as e:
            raise GenerationError(f"Model generation failed: {e}") from e

        if response is None:
            raise GenerationError("Model returned None response")

        code_blocks = extract_code_blocks(response)
        code = "\n\n".join(code_blocks) if code_blocks else ""

        self.history.append({
            "timestamp": time.time(),
            "request": request[:2000],
            "response": str(response)[:5000],
            "code": code[:5000],
            "has_code": bool(code.strip()),
        })
        if len(self.history) > 200:
            self.history = self.history[-200:]

        return response, code

    def generate_and_run(self, request: str, context: Optional[list] = None,
                         auto_fix: bool = True, max_retries: int = 3) -> Dict[str, Any]:
        """Generate code, execute it, and optionally auto-fix errors.

        Returns a dict with keys: response, code, result, iterations.
        """
        try:
            response, code = self.generate(request, context)
        except Exception as e:
            logger.error(f"Error in generate_and_run: {e}")
            return {
                "response": f"Generation failed: {e}",
                "code": "",
                "result": ExecutionResult(code="", error=f"Generation failed: {e}"),
                "iterations": [],
            }

        iterations = [{"code": code, "response": response}]

        if not code.strip():
            return {
                "response": response,
                "code": code,
                "result": ExecutionResult(code=code, error="No code generated"),
                "iterations": iterations,
            }

        result = execute_code(code, timeout=self.timeout, max_output=self.max_output)
        retry = 0
        while not result.success and auto_fix and retry < max_retries:
            retry += 1
            logger.info(f"Auto-fix attempt {retry}/{max_retries} for request: {request[:100]}...")
            logger.debug(f"Error details: {result.stderr or result.error}")

            fix_request = self._build_fix_request(code, result)
            try:
                response, code = self.generate(fix_request, system=DEBUG_SYSTEM)
            except Exception as e:
                result = ExecutionResult(code=code, error=f"Auto-fix generation failed: {e}")
                break

            if not code.strip():
                result = ExecutionResult(code=code, error="Auto-fix failed to generate new code")
                break

            result = execute_code(code, timeout=self.timeout, max_output=self.max_output)
            iterations.append({"code": code, "response": response, "result": result.to_dict()})

        return {
            "response": response,
            "code": code,
            "result": result,
            "iterations": iterations,
        }

    def review_code(self, code: str) -> str:
        """Review code for quality, bugs, and improvements."""
        messages = [
            {"role": "system", "content": "You are an expert code reviewer. Analyse the code for:\n"
             "1. Bugs and potential issues\n2. Performance concerns\n"
             "3. Code style and readability\n4. Security issues\n"
             "5. Suggested improvements\n\nBe specific and constructive."},
            {"role": "user", "content": f"Review this code:\n```python\n{code}\n```"},
        ]
        return self._model.generate(messages)

    def explain_code(self, code: str) -> str:
        """Explain what a piece of code does."""
        messages = [
            {"role": "system", "content": "Explain the given code clearly and thoroughly. "
             "Include: purpose, logic flow, key data structures, and any non-obvious behavior."},
            {"role": "user", "content": f"Explain this code:\n```python\n{code}\n```"},
        ]
        return self._model.generate(messages)

    def write_tests(self, code: str) -> Tuple[str, str]:
        """Generate tests for the given code."""
        messages = [
            {"role": "system", "content": "Write comprehensive pytest tests for the given code. "
             "Include edge cases, error cases, and typical usage. Return only code."},
            {"role": "user", "content": f"Write tests for:\n```python\n{code}\n```"},
        ]
        response = self._model.generate(messages)
        test_code = extract_all_code(response)
        return response, test_code

    def _build_fix_request(self, code: str, result: ExecutionResult) -> str:
        """Build a fix request from failed execution."""
        if not result:
            return f"The following code has an error. Fix it.\n\nCode:\n\`\`\`python\n{code}\n\`\`\`\n\nProvide the complete corrected code."
    
        error_output = result.stderr or result.error or "Unknown error"
        # Handle case where error_output might be None
        if error_output is None:
            error_output = "Unknown error"
    
        return (
            f"The following code has an error. Fix it.\n\n"
            f"Code:\n\`\`\`python\n{code}\n\`\`\`\n\n"
            f"Error output:\n\`\`\`\n{error_output}\n\`\`\`\n\n"
            f"Provide the complete corrected code."
        )
