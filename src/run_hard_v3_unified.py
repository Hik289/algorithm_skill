"""
Unified Hard Bench multi-sample runner. Methods:
- direct_v3: 5 independent samples, pass@1 and pass@5 (from original run_hard_benchmark.py)
- algoskill_v3: 10 trajectories with skill scaffolding + repair (from original)
- algoskill_g_v3: 5 samples greedy AlgoSkill (2-step prompt × 5 samples)
- cot_v3: 5 CoT samples, pass@5
- reflexion_v3: 5 attempts each with up to 3 refl rounds (pass@5)
- selfrefine_v3: 5 attempts each with up to 3 refine rounds (pass@5)
"""
import argparse, json, os, sys, time, random, re

# PD = repository root (parent of src/). Override with DATA_DIR env var
# to point at a different release / data location.
PD = os.environ.get("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PD, "src"))

from llm_client import call_llm_with_usage, BACKBONE_CONFIGS
import run_hard_benchmark as orig
# Override PROBLEMS
orig.PROBLEMS = list(json.load(open(os.path.join(PD, "data", "hard_bench_corpus.json"))).values())


def make_llm_fn(backbone):
    def llm_fn(prompt):
        out = call_llm_with_usage(prompt, backbone=backbone,
                                  temperature=0.8, max_tokens=4096)
        return out["text"], out["tokens"]
    return llm_fn


COT_T = """You are an expert competitive programmer.

Problem: {name}
{desc}
Constraints: {constr}

Step 1: identify the algorithmic family.
Step 2: state the time and space complexity you will aim for.
Step 3: write a complete Python program that reads from stdin and writes the answer
to stdout. Wrap your final solution in ```python ... ```.
"""

REFLEXION_T = """You are an expert competitive programmer.

Problem: {name}
{desc}
Constraints: {constr}

Write a complete Python program that reads from stdin and writes the answer
to stdout. Wrap your solution in ```python ... ```.
"""

REFLEXION_REFLECT_T = """The following solution to the problem below FAILED tests:

Problem: {name}
{desc}

Failed input:
{inp}
Expected: {exp}
Got: {got}

My previous code:
```python
{prev_code}
```

Reflect briefly (2-3 sentences) on why it might have failed, then write a
new corrected solution. Return your reflection followed by the new
```python``` code block.
"""

ALGOSKILL_G_T = """You are an expert competitive programmer using AlgoSkill methodology with greedy (single-step) skill selection.

Problem: {name}
{desc}
Constraints: {constr}

Step 1: PROBLEM_ABSTRACTION — identify canonical algorithmic family.
Step 2: CODE_GENERATION — write complete Python program reading from stdin.

Wrap final solution in ```python ... ```.
"""


def run_cot_v3(prob, llm_fn, n=5):
    prompt = COT_T.format(name=prob["name"], desc=prob["description"],
                          constr=prob["constraints"])
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    p1 = pk = False
    best_passed = 0
    best_code = ""
    for i in range(n):
        text, t = llm_fn(prompt)
        for k in tok: tok[k] += t.get(k, 0)
        code = orig.extract(text)
        if code:
            res = orig.verify(code, prob)
            ok = res["passed"] == res["total"] and res["total"] > 0
            if res["passed"] > best_passed:
                best_passed = res["passed"]; best_code = code
            if i == 0 and ok: p1 = True
            if ok: pk = True
        time.sleep(0.3)
    return {"pass_at_1": p1, "pass_at_k": pk, "best_passed": best_passed,
            "total_tests": len(prob["tests"]), "total_tokens": tok,
            "code": best_code[:3000]}


def run_algoskill_g_v3(prob, llm_fn, n=5):
    prompt = ALGOSKILL_G_T.format(name=prob["name"], desc=prob["description"],
                                  constr=prob["constraints"])
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    p1 = pk = False
    best_passed = 0; best_code = ""
    for i in range(n):
        text, t = llm_fn(prompt)
        for k in tok: tok[k] += t.get(k, 0)
        code = orig.extract(text)
        if code:
            res = orig.verify(code, prob)
            ok = res["passed"] == res["total"] and res["total"] > 0
            if res["passed"] > best_passed:
                best_passed = res["passed"]; best_code = code
            if i == 0 and ok: p1 = True
            if ok: pk = True
        time.sleep(0.3)
    return {"pass_at_1": p1, "pass_at_k": pk, "best_passed": best_passed,
            "total_tests": len(prob["tests"]), "total_tokens": tok,
            "code": best_code[:3000]}


def run_reflexion_v3(prob, llm_fn, n=5, max_rounds=3):
    """5 outer attempts each with up to max_rounds reflexion iterations."""
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    p1 = pk = False
    best_passed = 0; best_code = ""
    for i in range(n):
        prompt = REFLEXION_T.format(name=prob["name"], desc=prob["description"],
                                    constr=prob["constraints"])
        text, t = llm_fn(prompt)
        for k in tok: tok[k] += t.get(k, 0)
        code = orig.extract(text)
        attempt_passed = 0; attempt_ok = False
        for round_idx in range(max_rounds):
            if code:
                res = orig.verify(code, prob)
                if res["passed"] > attempt_passed:
                    attempt_passed = res["passed"]
                ok = res["passed"] == res["total"] and res["total"] > 0
                if ok:
                    attempt_ok = True
                    break
                # Reflect
                if res.get("failed"):
                    fc = res["failed"][0]
                    refl_prompt = REFLEXION_REFLECT_T.format(
                        name=prob["name"], desc=prob["description"],
                        inp=fc["input"][:300], exp=fc["expected"][:200],
                        got=fc["actual"][:200], prev_code=code[:2000])
                    text, t = llm_fn(refl_prompt)
                    for k in tok: tok[k] += t.get(k, 0)
                    code = orig.extract(text)
            else:
                break
        if attempt_passed > best_passed:
            best_passed = attempt_passed
            best_code = code if code else best_code
        if i == 0 and attempt_ok: p1 = True
        if attempt_ok: pk = True
        time.sleep(0.3)
    return {"pass_at_1": p1, "pass_at_k": pk, "best_passed": best_passed,
            "total_tests": len(prob["tests"]), "total_tokens": tok,
            "code": best_code[:3000]}


SELFREFINE_CRIT_T = """You wrote the following solution. Critique it: identify
potential bugs, edge cases, or inefficiencies. Be concise.

Problem: {name}
{desc}

Solution:
```python
{prev_code}
```
"""

SELFREFINE_REV_T = """Given the critique, write an improved solution as a complete
Python program. Wrap in ```python ... ```.

Problem: {name}
{desc}

Previous solution:
```python
{prev_code}
```

Critique:
{critique}
"""


def run_selfrefine_v3(prob, llm_fn, n=5, max_rounds=3):
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    p1 = pk = False
    best_passed = 0; best_code = ""
    for i in range(n):
        prompt = REFLEXION_T.format(name=prob["name"], desc=prob["description"],
                                    constr=prob["constraints"])
        text, t = llm_fn(prompt)
        for k in tok: tok[k] += t.get(k, 0)
        code = orig.extract(text)
        attempt_passed = 0; attempt_ok = False
        for round_idx in range(max_rounds):
            if code:
                res = orig.verify(code, prob)
                if res["passed"] > attempt_passed:
                    attempt_passed = res["passed"]
                ok = res["passed"] == res["total"] and res["total"] > 0
                if ok:
                    attempt_ok = True
                    break
                # Critique
                crit_p = SELFREFINE_CRIT_T.format(
                    name=prob["name"], desc=prob["description"],
                    prev_code=code[:2000])
                crit_text, ct = llm_fn(crit_p)
                for k in tok: tok[k] += ct.get(k, 0)
                # Revise
                rev_p = SELFREFINE_REV_T.format(
                    name=prob["name"], desc=prob["description"],
                    prev_code=code[:2000], critique=crit_text[:1500])
                text, t = llm_fn(rev_p)
                for k in tok: tok[k] += t.get(k, 0)
                code = orig.extract(text)
            else:
                break
        if attempt_passed > best_passed:
            best_passed = attempt_passed
            best_code = code if code else best_code
        if i == 0 and attempt_ok: p1 = True
        if attempt_ok: pk = True
        time.sleep(0.3)
    return {"pass_at_1": p1, "pass_at_k": pk, "best_passed": best_passed,
            "total_tests": len(prob["tests"]), "total_tokens": tok,
            "code": best_code[:3000]}


# patch orig.run_algoskill to also store best_code in return dict
_orig_alg = orig.run_algoskill
def algoskill_with_code(prob, llm_fn, n_traj=10, seed=42):
    rng = random.Random(seed)
    seqs = rng.sample(orig.SEQS, min(n_traj, len(orig.SEQS)))
    while len(seqs) < n_traj:
        seqs.append(rng.choice(orig.SEQS))
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    best_passed = 0; best_code = ""; pk = False; p1 = False
    ctx = f"Problem: {prob['name']}\n{prob['description']}\nConstraints: {prob['constraints']}"
    for ti, seq in enumerate(seqs):
        notes = []
        for skill in seq:
            if skill not in orig.SKILL_P: continue
            sp = f"{ctx}\n\n[{skill.upper()}]: {orig.SKILL_P[skill]}\nPrevious analysis:\n" + "\n".join(notes[-2:])
            try:
                txt, t = llm_fn(sp)
                for k in tok: tok[k] += t.get(k, 0)
                notes.append(f"[{skill}]: {txt[:300]}")
            except Exception: pass
        fp = orig.FINAL_T.format(name=prob["name"], desc=prob["description"],
                                  constr=prob["constraints"], notes="\n".join(notes))
        try:
            txt, t = llm_fn(fp); 
            for k in tok: tok[k] += t.get(k, 0)
            code = orig.extract(txt)
        except: code = None
        if code:
            res = orig.verify(code, prob)
            psd = res["passed"]
            ok = psd == res["total"] and res["total"] > 0
            if not ok and res.get("failed"):
                fc = res["failed"][0]
                rp = orig.REPAIR_T.format(name=prob["name"], inp=fc["input"],
                                          exp=fc["expected"], got=fc["actual"])
                try:
                    rtxt, rt = llm_fn(rp)
                    for k in tok: tok[k] += rt.get(k, 0)
                    rcode = orig.extract(rtxt)
                    if rcode:
                        rres = orig.verify(rcode, prob)
                        if rres["passed"] >= psd:
                            code, psd = rcode, rres["passed"]
                            ok = psd == res["total"]
                except: pass
            if psd > best_passed:
                best_passed = psd; best_code = code
            if ok:
                pk = True
                if ti == 0: p1 = True
    return {"pass_at_1": p1, "pass_at_k": pk, "best_passed": best_passed,
            "total_tests": len(prob["tests"]), "total_tokens": tok,
            "code": best_code[:3000] if best_code else ""}


def run_direct_v3_with_code(prob, llm_fn, n=5):
    """Patched: same logic as orig.run_direct but stores best_code so T-opt judge works."""
    prompt = orig.DIRECT_TMPL.format(name=prob["name"], desc=prob["description"],
                                      constr=prob["constraints"])
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    p1 = pk = False
    best_passed = 0
    best_code = ""
    for i in range(n):
        text, t = llm_fn(prompt)
        for k in tok: tok[k] += t.get(k,0)
        code = orig.extract(text)
        if code:
            res = orig.verify(code, prob)
            ok = res["passed"] == res["total"] and res["total"] > 0
            if res["passed"] > best_passed:
                best_passed = res["passed"]; best_code = code
            if i == 0 and ok: p1 = True
            if ok: pk = True
        time.sleep(0.5)
    return {"pass_at_1": p1, "pass_at_k": pk, "best_passed": best_passed,
            "total_tests": len(prob["tests"]), "total_tokens": tok,
            "code": best_code[:3000]}


METHODS = {
    "direct_v3": run_direct_v3_with_code,
    "algoskill_v3": algoskill_with_code,
    "algoskill_g_v3": run_algoskill_g_v3,
    "cot_v3": run_cot_v3,
    "reflexion_v3": run_reflexion_v3,
    "selfrefine_v3": run_selfrefine_v3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONE_CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_traj", type=int, default=10,
                    help="Number of AlgoSkill trajectories (only affects algoskill_v3). Default 10 = original paper. Use 5 for half-cost runs.")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    saved = json.load(open(args.out)) if os.path.exists(args.out) else {}
    print(f"[hard-v3] method={args.method} backbone={args.backbone} resume={len(saved)}")
    llm_fn = make_llm_fn(args.backbone)
    runner = METHODS[args.method]

    for prob in orig.PROBLEMS:
        pid = prob["id"]
        if pid in saved:
            print(f"  skip {pid}")
            continue
        print(f"  [{pid}] {prob['name'][:40]}", end=" ", flush=True)
        t0 = time.time()
        try:
            if args.method == "algoskill_v3":
                r = runner(prob, llm_fn, n_traj=args.n_traj)
            else:
                r = runner(prob, llm_fn)
            r["elapsed"] = round(time.time() - t0, 1)
            # Compat with multitest schema:
            r["correct"] = bool(r.get("pass_at_k"))
            r["passed_count"] = r.get("best_passed", 0)
            r["total_tests"] = r.get("total_tests", 0)
            r["tokens"] = r.get("total_tokens", {})
            r["code"] = r.get("code", "")
            saved[pid] = r
            print(f"p@1={r.get('pass_at_1')} p@k={r.get('pass_at_k')} "
                  f"best={r.get('best_passed','?')}/{r.get('total_tests','?')} "
                  f"tok={r.get('total_tokens',{}).get('total_tokens',0)} {r['elapsed']}s")
        except Exception as e:
            import traceback; traceback.print_exc()
            saved[pid] = {"correct": False, "error": str(e)[:300]}
            print(f"ERROR: {e}")
        with open(args.out, "w") as f:
            json.dump(saved, f, indent=2)

    n_p1 = sum(1 for v in saved.values() if v.get("pass_at_1"))
    n_pk = sum(1 for v in saved.values() if v.get("pass_at_k"))
    print(f"\n[hard-v3 final] {args.method}@{args.backbone}: "
          f"p@1={n_p1}/{len(saved)} p@5={n_pk}/{len(saved)}")


if __name__ == "__main__":
    main()
