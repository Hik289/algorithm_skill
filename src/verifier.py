"""
Code verifier: execute generated code against test cases with timeout.
"""
import signal
import ast
import re
import traceback
from typing import Any, List, Tuple


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try fenced code block
    patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"```py\n(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.DOTALL)
        if m:
            return m.group(1).strip()
    # If no fenced block, return the whole response stripped
    return response.strip()


def run_code_with_tests(code: str, problem: dict, timeout_sec: int = 5) -> dict:
    """
    Execute code against all test cases.
    Returns:
      {
        "compile_ok": bool,
        "passed": int,
        "total": int,
        "pass_rate": float,
        "errors": list of str,
      }
    """
    result = {
        "compile_ok": False,
        "passed": 0,
        "total": len(problem["tests"]),
        "pass_rate": 0.0,
        "errors": [],
    }

    # Try to compile
    try:
        code_clean = extract_code(code)
        compiled = compile(code_clean, "<string>", "exec")
        result["compile_ok"] = True
    except SyntaxError as e:
        result["errors"].append(f"SyntaxError: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"CompileError: {e}")
        return result

    # Run each test
    for i, test in enumerate(problem["tests"]):
        args, expected = test
        if not isinstance(args, tuple):
            args = (args,)

        try:
            # Set up execution namespace
            ns = {}
            exec(compiled, ns)
            func = ns.get("solution")
            if func is None:
                result["errors"].append(f"Test {i}: 'solution' function not found")
                continue

            # Run with timeout
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_sec)
            try:
                actual = func(*args)
            finally:
                signal.alarm(0)

            # Check result
            if _check_result(actual, expected, problem.get("check", "exact"), args, problem):
                result["passed"] += 1
            else:
                result["errors"].append(
                    f"Test {i}: expected {expected!r}, got {actual!r}"
                )
        except TimeoutError:
            result["errors"].append(f"Test {i}: TLE (>{timeout_sec}s)")
        except Exception as e:
            result["errors"].append(f"Test {i}: RuntimeError: {e}")

    result["pass_rate"] = result["passed"] / result["total"] if result["total"] > 0 else 0.0
    return result


def _check_result(actual, expected, check_type: str, args, problem) -> bool:
    """Check if actual result matches expected using the problem's check type."""
    if check_type == "exact":
        return actual == expected
    elif check_type == "sorted":
        # For Two Sum: sort both
        if actual is None:
            return False
        try:
            return sorted(actual) == sorted(expected)
        except Exception:
            return actual == expected
    elif check_type == "sorted_lists":
        # K closest points: sort each inner list, then sort outer
        if actual is None:
            return False
        try:
            a = sorted([sorted(x) for x in actual])
            e = sorted([sorted(x) for x in expected])
            return a == e
        except Exception:
            return False
    elif check_type == "peak_valid":
        # Any valid peak index is acceptable
        if actual is None:
            return False
        try:
            nums = args[0]
            idx = int(actual)
            n = len(nums)
            left_ok = (idx == 0) or (nums[idx] > nums[idx - 1])
            right_ok = (idx == n - 1) or (nums[idx] > nums[idx + 1])
            return left_ok and right_ok
        except Exception:
            return False
    elif check_type == "subset_valid":
        if actual is None:
            return False
        try:
            a = sorted([sorted(x) for x in actual])
            e = sorted([sorted(x) for x in expected])
            return a == e
        except Exception:
            return False
    else:
        return actual == expected
