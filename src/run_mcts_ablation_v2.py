"""
Task 1: rerun mcts_ablation with the configured default backend.

Conditions: Greedy_Policy, Beam_Search, MCTS_NoPolicy, MCTS_WithPolicy
Problems: 20 (from problems.py)
Samples per (cond, problem): default 5

Output JSON schema matches old mcts_ablation_results.json so downstream
analysis scripts keep working.
"""
import os, sys, json, time, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from problems import PROBLEMS
from verifier import run_code_with_tests
from algoskill import (
    AlgoSkillMCTS, AlgoSkillGreedy,
    run_algoskill_full, run_algoskill_greedy,
)
from ablations import run_beam_search, run_mcts_no_policy


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def evaluate(method_fn, problem, n_samples, per_problem_timeout=1200):
    """Run method_fn(problem, n_samples). If it exceeds per_problem_timeout
    seconds, we record what was completed and move on (signal-based timeout
    is unsafe with threads; we use a simple wallclock check inside the call
    via a SIGALRM hack).
    """
    import signal

    class TimeoutError_(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutError_("per_problem timeout")

    t0 = time.time()
    # Install SIGALRM (POSIX only; safe in main thread of CPython)
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(per_problem_timeout)
    code_list = []
    try:
        try:
            code_list = method_fn(problem, n_samples=n_samples)
        except TimeoutError_:
            print(f"    !! timeout after {per_problem_timeout}s, "
                  f"recording partial result", flush=True)
        except Exception as e:
            traceback.print_exc()
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return {
                "pass_rate_avg": 0.0, "pass1": 0.0, "pass5": 0.0,
                "compile_rate": 0.0, "elapsed": time.time() - t0,
                "n_passed_all": 0, "n_samples": 0, "per_sample": [],
                "error": str(e),
            }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    results = []
    for code in code_list:
        if not code or not code.strip():
            results.append({"passed": 0, "total": len(problem["tests"]),
                            "pass_rate": 0.0, "compile_ok": False})
            continue
        vr = run_code_with_tests(code, problem)
        results.append(vr)
    n = len(results)
    c = sum(1 for r in results if r["pass_rate"] == 1.0)
    pass_rates = [r["pass_rate"] for r in results]
    return {
        "pass_rate_avg": float(np.mean(pass_rates)) if pass_rates else 0.0,
        "pass1": pass_at_k(n, c, 1) if n else 0.0,
        "pass5": pass_at_k(n, c, min(5, n)) if n else 0.0,
        "compile_rate": sum(1 for r in results if r["compile_ok"]) / n if n else 0.0,
        "n_passed_all": c,
        "n_samples": n,
        "elapsed": time.time() - t0,
        "per_sample": results,
    }


CONDITIONS = {
    "Greedy_Policy": run_algoskill_greedy,
    "Beam_Search": run_beam_search,
    "MCTS_NoPolicy": run_mcts_no_policy,
    "MCTS_WithPolicy": run_algoskill_full,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--problems", default="all",
                    help="all | smoke | comma list of pids e.g. P01,P02")
    ap.add_argument("--conditions", default="all",
                    help="all | comma list e.g. Greedy_Policy,MCTS_WithPolicy")
    ap.add_argument("--out", required=True, help="output JSON file")
    ap.add_argument("--backbone", default="default")
    args = ap.parse_args()

    os.environ.setdefault("ALGOSKILL_BACKBONE", args.backbone)

    # Select problems
    if args.problems == "all":
        probs = PROBLEMS
    elif args.problems == "smoke":
        probs = PROBLEMS[:1]
    else:
        pid_set = set(args.problems.split(","))
        probs = [p for p in PROBLEMS if p["id"] in pid_set]

    # Select conditions
    if args.conditions == "all":
        conds = list(CONDITIONS.keys())
    else:
        conds = args.conditions.split(",")

    print(f"Conditions: {conds}", flush=True)
    print(f"Problems  : {[p['id'] for p in probs]} ({len(probs)})", flush=True)
    print(f"n_samples : {args.n_samples}", flush=True)
    print(f"backbone  : {args.backbone}", flush=True)
    print(f"out       : {args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Resume support
    if os.path.exists(args.out):
        with open(args.out) as f:
            all_results = json.load(f)
        print(f"Resuming with {sum(len(v) for v in all_results.values())} "
              f"existing cells", flush=True)
    else:
        all_results = {}

    for cond in conds:
        if cond not in CONDITIONS:
            print(f"  skip unknown condition {cond}", flush=True)
            continue
        method_fn = CONDITIONS[cond]
        all_results.setdefault(cond, {})

        for prob in probs:
            pid = prob["id"]
            if pid in all_results[cond]:
                print(f"  {cond} {pid} [already done] "
                      f"pass1={all_results[cond][pid]['pass1']:.2f}", flush=True)
                continue
            t0 = time.time()
            print(f"  {cond} {pid} starting...", flush=True)
            try:
                metrics = evaluate(method_fn, prob, args.n_samples)
            except Exception as e:
                traceback.print_exc()
                metrics = {"error": str(e), "elapsed": time.time() - t0}
            elapsed = metrics.get("elapsed", time.time() - t0)
            print(f"  {cond} {pid}  pass1={metrics.get('pass1',0):.2f} "
                  f"pass5={metrics.get('pass5',0):.2f} "
                  f"compile={metrics.get('compile_rate',0):.2f} "
                  f"npassed={metrics.get('n_passed_all',0)}/{metrics.get('n_samples',0)} "
                  f"elapsed={elapsed:.0f}s", flush=True)
            all_results[cond][pid] = metrics
            # Save incrementally
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=2)

    print("Done. Summary:", flush=True)
    for cond, v in all_results.items():
        if not v:
            continue
        p1s = [c.get("pass1", 0) for c in v.values()]
        p5s = [c.get("pass5", 0) for c in v.values()]
        print(f"  {cond}: pass@1={np.mean(p1s)*100:.1f}% pass@5={np.mean(p5s)*100:.1f}% "
              f"(over {len(v)} problems)", flush=True)


if __name__ == "__main__":
    main()
