"""
Runner for multi-test corpora (Task 5 post-cutoff, Task 6 cross-platform).

Each problem has a list of tests [{"input": str, "output": str}, ...].
A problem is "correct" iff all tests pass.
"""
import argparse, json, os, sys, time, subprocess, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import call_llm_with_usage, BACKBONE_CONFIGS


DIRECT_T = """You are an expert competitive programmer.

Problem: {description}
Constraints: {constraints}

Write a complete Python program that reads from stdin and writes the answer
to stdout. Wrap your solution in ```python ... ```.
"""

COT_T = """You are an expert competitive programmer.

Problem: {description}
Constraints: {constraints}

Step 1: identify the algorithmic family.
Step 2: state the time and space complexity you will aim for.
Step 3: write a complete Python program that reads from stdin and writes the
answer to stdout. Wrap your final solution in ```python ... ```.
"""

ALGOSKILL_T = """You are an expert competitive programmer using the AlgoSkill
methodology: explicitly select and apply algorithmic skills before coding.

Problem: {description}
Constraints: {constraints}

Apply the following typed skills in order:
1. CONSTRAINT_READING — note input bounds and feasible complexity class.
2. PROBLEM_ABSTRACTION — identify the canonical algorithmic family.
3. STATE_DESIGN — describe the state / data structure used.
4. COMPLEXITY_REFINEMENT — confirm optimal time complexity.
5. CODE_GENERATION — write a complete Python program. Wrap final solution
   in ```python ... ```.
"""

ALGOSKILL_GREEDY_T = """You are an expert competitive programmer using the
AlgoSkill methodology with greedy (single-step) skill selection.

Problem: {description}
Constraints: {constraints}

Step 1: PROBLEM_ABSTRACTION — identify the canonical algorithmic family and
note feasible complexity class given the constraints.
Step 2: CODE_GENERATION — write a complete Python program. Wrap final
solution in ```python ... ```.
"""


REFLEXION_T = """You are an expert competitive programmer.

Problem: {description}
Constraints: {constraints}

Write a complete Python program that reads from stdin and writes the
answer to stdout. Wrap your solution in ```python ... ```.
"""

REFLEXION_REFLECT_T = """The following solution to the problem below FAILED tests:

Problem: {description}
Constraints: {constraints}

Failed input:
{failed_input}
Expected output:
{failed_expected}
Actual output:
{failed_actual}

My previous code:
```python
{prev_code}
```

Reflect briefly (2-3 sentences) on why it might have failed, then write a
new corrected solution. Return your reflection followed by the new
```python``` code block.
"""

SELFREFINE_T = """You are an expert competitive programmer.

Problem: {description}
Constraints: {constraints}

Write a complete Python program that reads from stdin and writes the answer
to stdout. Wrap your solution in ```python ... ```.
"""

SELFREFINE_CRIT_T = """You wrote the following solution to the problem below.
Critique it: identify potential bugs, edge cases it might miss, or
inefficiencies. Be concise.

Problem: {description}
Constraints: {constraints}

Solution:
```python
{prev_code}
```
"""

SELFREFINE_REVISE_T = """Given the critique below, write an improved solution.
Return ONLY a ```python``` code block.

Problem: {description}
Constraints: {constraints}

Previous solution:
```python
{prev_code}
```

Critique:
{critique}
"""


METHOD_TEMPLATES = {
    "direct": DIRECT_T,
    "cot": COT_T,
    "algoskill": ALGOSKILL_T,
    "algoskill_g": ALGOSKILL_GREEDY_T,
    "reflexion": REFLEXION_T,      # uses iterative loop in run_one
    "selfrefine": SELFREFINE_T,    # uses iterative loop in run_one
}


def extract_code(text):
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()


def run_code(code, stdin_data, timeout=8.0):
    try:
        p = subprocess.run([sys.executable, "-c", code],
                           input=stdin_data, capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout.strip(), p.returncode, p.stderr[:300]
    except subprocess.TimeoutExpired:
        return "", "TLE", ""
    except Exception as e:
        return "", "ERR", str(e)[:200]


def verify(code, prob):
    passed = 0
    total = 0
    details = []
    for tc in prob.get("tests", []):
        total += 1
        actual, rc, err = run_code(code, tc["input"])
        expected = tc["output"].strip()
        a_norm = " ".join(actual.split())
        e_norm = " ".join(expected.split())
        ok = (a_norm == e_norm)
        if ok: passed += 1
        details.append({"expected": expected[:200], "actual": actual[:200], "ok": ok})
    return {"passed": passed, "total": total, "details": details,
            "all_passed": (passed == total and total > 0)}


def _verify_collect_failed(code, prob):
    """Verify and also return first failed test details (for reflexion)."""
    vres = verify(code, prob)
    failed = None
    if not vres["all_passed"] and vres.get("details"):
        for tc, det in zip(prob.get("tests", []), vres["details"]):
            if not det.get("ok"):
                failed = {"input": tc["input"], "expected": det["expected"],
                          "actual": det["actual"]}
                break
    return vres, failed


def run_one(prob, method, backbone, temperature=0.5, max_rounds=3):
    t0 = time.time()
    tok_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if method in ("direct", "cot", "algoskill", "algoskill_g"):
        template = METHOD_TEMPLATES[method]
        prompt = template.format(
            description=prob["description"],
            constraints=prob.get("constraints", ""),
        )
        try:
            out = call_llm_with_usage(prompt, backbone=backbone,
                                      temperature=temperature, max_tokens=8000)
        except Exception as e:
            return {"correct": False, "error": str(e)[:300],
                    "tokens": tok_total,
                    "elapsed": time.time()-t0, "code":"", "llm_response":""}
        for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
        code = extract_code(out["text"])
        vres = verify(code, prob)
        return {
            "correct": vres["all_passed"],
            "passed_count": vres["passed"],
            "total_tests": vres["total"],
            "code": code[:3000],
            "llm_response": out["text"][:1500],
            "verify": {"passed": vres["passed"], "total": vres["total"]},
            "tokens": tok_total,
            "elapsed": time.time() - t0,
        }

    if method == "reflexion":
        # Try; if fail, reflect on first failed test and retry, up to max_rounds.
        prompt = REFLEXION_T.format(
            description=prob["description"],
            constraints=prob.get("constraints", ""),
        )
        try:
            out = call_llm_with_usage(prompt, backbone=backbone,
                                      temperature=temperature, max_tokens=8000)
        except Exception as e:
            return {"correct": False, "error": str(e)[:300],
                    "tokens": tok_total, "elapsed": time.time()-t0,
                    "code":"", "llm_response":""}
        for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
        code = extract_code(out["text"])
        vres, failed = _verify_collect_failed(code, prob)
        for r in range(max_rounds - 1):
            if vres["all_passed"]:
                break
            if not failed:
                break
            reflect_prompt = REFLEXION_REFLECT_T.format(
                description=prob["description"],
                constraints=prob.get("constraints", ""),
                failed_input=failed["input"][:500],
                failed_expected=failed["expected"][:200],
                failed_actual=failed["actual"][:200],
                prev_code=code[:2500],
            )
            try:
                out = call_llm_with_usage(reflect_prompt, backbone=backbone,
                                          temperature=temperature,
                                          max_tokens=8000)
            except Exception as e:
                break
            for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
            code = extract_code(out["text"])
            vres, failed = _verify_collect_failed(code, prob)
        return {
            "correct": vres["all_passed"],
            "passed_count": vres["passed"],
            "total_tests": vres["total"],
            "code": code[:3000],
            "llm_response": out["text"][:1500],
            "verify": {"passed": vres["passed"], "total": vres["total"]},
            "tokens": tok_total,
            "elapsed": time.time() - t0,
        }

    if method == "selfrefine":
        prompt = SELFREFINE_T.format(
            description=prob["description"],
            constraints=prob.get("constraints", ""),
        )
        try:
            out = call_llm_with_usage(prompt, backbone=backbone,
                                      temperature=temperature, max_tokens=8000)
        except Exception as e:
            return {"correct": False, "error": str(e)[:300],
                    "tokens": tok_total, "elapsed": time.time()-t0,
                    "code":"", "llm_response":""}
        for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
        code = extract_code(out["text"])
        vres = verify(code, prob)
        for _ in range(max_rounds - 1):
            if vres["all_passed"]:
                break
            crit_prompt = SELFREFINE_CRIT_T.format(
                description=prob["description"],
                constraints=prob.get("constraints", ""),
                prev_code=code[:2500],
            )
            try:
                crit_out = call_llm_with_usage(crit_prompt, backbone=backbone,
                                               temperature=0.5, max_tokens=1024)
            except Exception:
                break
            for k in tok_total: tok_total[k] += crit_out["tokens"].get(k, 0)
            revise_prompt = SELFREFINE_REVISE_T.format(
                description=prob["description"],
                constraints=prob.get("constraints", ""),
                prev_code=code[:2500],
                critique=crit_out["text"][:1500],
            )
            try:
                out = call_llm_with_usage(revise_prompt, backbone=backbone,
                                          temperature=0.5, max_tokens=8000)
            except Exception:
                break
            for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
            code = extract_code(out["text"])
            vres = verify(code, prob)
        return {
            "correct": vres["all_passed"],
            "passed_count": vres["passed"],
            "total_tests": vres["total"],
            "code": code[:3000],
            "llm_response": out["text"][:1500],
            "verify": {"passed": vres["passed"], "total": vres["total"]},
            "tokens": tok_total,
            "elapsed": time.time() - t0,
        }

    raise ValueError(f"unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--method", required=True, choices=list(METHOD_TEMPLATES))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONE_CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    corpus = json.load(open(args.corpus))
    if isinstance(corpus, dict):
        items = list(corpus.values())
    else:
        items = corpus
    if args.limit:
        items = items[:args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    print(f"[multitest] method={args.method} backbone={args.backbone} "
          f"corpus={len(items)} resume={len(results)}", flush=True)

    for prob in items:
        pid = prob["id"]
        if pid in results:
            continue
        r = run_one(prob, args.method, args.backbone)
        results[pid] = {
            "platform": prob.get("platform", "?"),
            "theme": prob.get("theme", ""),
            "tags": prob.get("tags", ""),
            **r,
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        if len(results) % 5 == 0 or len(results) <= 5:
            n_correct = sum(1 for v in results.values() if v.get("correct"))
            print(f"  [{len(results)}/{len(items)}] {pid} correct={r['correct']} "
                  f"tests={r.get('passed_count', '?')}/{r.get('total_tests', '?')} {r['elapsed']:.1f}s "
                  f"({n_correct}/{len(results)})", flush=True)

    n_correct = sum(1 for v in results.values() if v.get("correct"))
    print(f"\n[multitest] {args.method}@{args.backbone}: "
          f"{n_correct}/{len(results)} = {n_correct/len(results)*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
