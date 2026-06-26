"""
Task 4: rule-based benchmark runner.

Runs Direct LLM / CoT / AlgoSkill on the 200-problem rule_based_corpus.
For each (problem, method, backbone) record:
- correct: bool
- code: generated Python program
- estimated_time_complexity: LLM-judge based, kept simple as a separate post-step
  (here we only verify correctness; T-opt judging is a post-processing step)

Output: raw_data_v2/results_rule_based/<method>_<backbone>.json
"""
import argparse, json, os, sys, time, subprocess, traceback, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import call_llm_with_usage, BACKBONE_CONFIGS


# ── Prompts ──────────────────────────────────────────────────────────────────
DIRECT_T = """You are an expert competitive programmer.

Problem: {description}
Constraints: {constraints}

Write a complete Python program that reads from stdin and writes the answer
to stdout. Wrap your solution in ```python ... ```.

Input format:
{input_example}
"""

COT_T = """You are an expert competitive programmer.

Problem: {description}
Constraints: {constraints}

Step 1: identify the algorithmic family.
Step 2: state the time and space complexity you will aim for.
Step 3: write a complete Python program that reads from stdin and writes the
answer to stdout. Wrap your final solution in ```python ... ```.

Input format:
{input_example}
"""

ALGOSKILL_T = """You are an expert competitive programmer using the AlgoSkill
methodology: explicitly select and apply algorithmic skills before coding.

Problem: {description}
Constraints: {constraints}

Apply the following typed skills in order:
1. CONSTRAINT_READING — note input bounds and feasible complexity class.
2. PROBLEM_ABSTRACTION — identify the canonical algorithmic family this
   problem belongs to.
3. STATE_DESIGN — describe the state / data structure used.
4. COMPLEXITY_REFINEMENT — confirm your solution achieves the optimal time
   complexity.
5. CODE_GENERATION — write a complete Python program that reads from stdin
   and writes the answer to stdout. Wrap your final solution in ```python ... ```.

Input format:
{input_example}
"""


ALGOSKILL_G_T = """You are an expert competitive programmer using AlgoSkill
methodology with greedy (single-step) skill selection.

Problem: {description}
Constraints: {constraints}

Step 1: PROBLEM_ABSTRACTION — identify the canonical algorithmic family and
note feasible complexity class given the constraints.
Step 2: CODE_GENERATION — write a complete Python program that reads from
stdin and writes to stdout. Wrap final solution in ```python ... ```.

Input format:
{input_example}
"""

REFLEXION_T = """You are an expert competitive programmer.

Problem: {description}
Constraints: {constraints}

Write a complete Python program that reads from stdin and writes the answer
to stdout. Wrap your solution in ```python ... ```.

Input format:
{input_example}
"""

REFLEXION_REFLECT_T = """The following solution to the problem below FAILED its test:

Problem: {description}
Constraints: {constraints}

Test input:
{failed_input}
Expected output: {failed_expected}
Actual output: {failed_actual}

My previous solution:
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

Write a complete Python program that reads from stdin and writes the
answer to stdout. Wrap your solution in ```python ... ```.

Input format:
{input_example}
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
    "algoskill_g": ALGOSKILL_G_T,
    "reflexion": REFLEXION_T,
    "selfrefine": SELFREFINE_T,
}


def extract_code(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()


def run_code(code: str, stdin_data: str, timeout: float = 5.0):
    try:
        p = subprocess.run([sys.executable, "-c", code],
                           input=stdin_data, capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout.strip(), p.returncode, p.stderr[:300]
    except subprocess.TimeoutExpired:
        return "", "TLE", ""
    except Exception as e:
        return "", "ERR", str(e)[:200]


def verify(code: str, prob: dict) -> dict:
    """Run code on prob['input'], compare to prob['expected_output']."""
    actual, rc, err = run_code(code, prob["input"])
    expected = prob["expected_output"].strip()
    # Normalize whitespace (tolerate trailing newline / extra spaces)
    a_norm = " ".join(actual.split())
    e_norm = " ".join(expected.split())
    ok = a_norm == e_norm
    return {"correct": ok, "actual": actual[:500], "expected": expected[:500],
            "rc": rc, "err": err[:200]}


def _run_one_iterative(prob: dict, method: str, backbone: str,
                       temperature: float = 0.5, max_rounds: int = 3):
    """Run reflexion/selfrefine with iterative refinement."""
    template = METHOD_TEMPLATES[method]
    prompt = template.format(
        description=prob["description"],
        constraints=prob["constraints"],
        input_example=prob.get("input", "")[:300],
    )
    t0 = time.time()
    tok_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        out = call_llm_with_usage(prompt, backbone=backbone,
                                  temperature=temperature, max_tokens=2500)
    except Exception as e:
        return {"correct": False, "error": str(e)[:300],
                "tokens": tok_total, "elapsed": time.time()-t0,
                "code": "", "llm_response": ""}
    for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
    code = extract_code(out["text"])
    vres = verify(code, prob)
    for _ in range(max_rounds - 1):
        if vres.get("correct"):
            break
        if method == "reflexion":
            reflect_prompt = REFLEXION_REFLECT_T.format(
                description=prob["description"],
                constraints=prob["constraints"],
                failed_input=prob["input"][:500],
                failed_expected=vres.get("expected","")[:200],
                failed_actual=vres.get("actual","")[:200],
                prev_code=code[:2500],
            )
            try:
                out = call_llm_with_usage(reflect_prompt, backbone=backbone,
                                          temperature=temperature, max_tokens=2500)
            except Exception:
                break
            for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
            code = extract_code(out["text"])
            vres = verify(code, prob)
        elif method == "selfrefine":
            crit_prompt = SELFREFINE_CRIT_T.format(
                description=prob["description"],
                constraints=prob["constraints"],
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
                constraints=prob["constraints"],
                prev_code=code[:2500],
                critique=crit_out["text"][:1500],
            )
            try:
                out = call_llm_with_usage(revise_prompt, backbone=backbone,
                                          temperature=0.5, max_tokens=2500)
            except Exception:
                break
            for k in tok_total: tok_total[k] += out["tokens"].get(k, 0)
            code = extract_code(out["text"])
            vres = verify(code, prob)
    return {
        "correct": vres.get("correct", False),
        "code": code[:3000],
        "llm_response": out["text"][:1500],
        "verify": {"correct": vres.get("correct", False),
                   "actual": vres.get("actual","")[:200],
                   "expected": vres.get("expected","")[:200]},
        "tokens": tok_total,
        "elapsed": time.time() - t0,
    }


def run_one(prob: dict, method: str, backbone: str, temperature: float = 0.5,
            max_rounds: int = 3):
    """Single-shot for direct/cot/algoskill/algoskill_g; iterative for
    reflexion/selfrefine."""
    if method in ("reflexion", "selfrefine"):
        return _run_one_iterative(prob, method, backbone, temperature, max_rounds)
    template = METHOD_TEMPLATES[method]
    prompt = template.format(
        description=prob["description"],
        constraints=prob["constraints"],
        input_example=prob["input"][:300],
    )
    t0 = time.time()
    try:
        out = call_llm_with_usage(prompt, backbone=backbone,
                                  temperature=temperature, max_tokens=2500)
    except Exception as e:
        return {
            "correct": False, "error": str(e)[:300],
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "elapsed": time.time() - t0,
            "code": "", "llm_response": "",
        }
    code = extract_code(out["text"])
    vres = verify(code, prob)
    return {
        "correct": bool(vres["correct"]),
        "code": code[:3000],
        "llm_response": out["text"][:1500],
        "verify": vres,
        "tokens": out["tokens"],
        "elapsed": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="rule_based_corpus.json")
    ap.add_argument("--method", required=True, choices=list(METHOD_TEMPLATES))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONE_CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap on # problems")
    args = ap.parse_args()

    with open(args.corpus) as f:
        corpus = json.load(f)
    if args.limit:
        corpus = corpus[:args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    print(f"[task4] method={args.method} backbone={args.backbone} "
          f"corpus={len(corpus)} resume_existing={len(results)}", flush=True)

    for prob in corpus:
        pid = prob["id"]
        if pid in results:
            continue
        r = run_one(prob, args.method, args.backbone)
        results[pid] = {
            "family": prob["family"],
            "opt_time": prob["opt_time"],
            "opt_space": prob["opt_space"],
            **r,
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        if len(results) % 5 == 0 or len(results) <= 5:
            n_correct = sum(1 for v in results.values() if v.get("correct"))
            print(f"  [{len(results)}/{len(corpus)}] {pid} correct={r['correct']} "
                  f"tok={r['tokens']['total_tokens']} {r['elapsed']:.1f}s "
                  f"(rolling {n_correct}/{len(results)})", flush=True)

    n_correct = sum(1 for v in results.values() if v.get("correct"))
    print(f"\n[task4] {args.method}@{args.backbone}: {n_correct}/{len(results)} "
          f"correct = {n_correct/len(results)*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
