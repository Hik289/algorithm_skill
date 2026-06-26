"""
Task 8: collect REAL skill_history from AlgoSkill MCTS rollouts to replace
the synthetic fig5_skill_frequency data (which extra_experiments.py L66-80
generated with random.choice from 8 hardcoded sequences).

This re-runs AlgoSkill Full (MCTS) on N problems × K samples and records the
actual skill_history of each trajectory.

Output: raw_data_v2/results/skill_frequency_real.json
Schema:
{
  "trajectories": [{"problem_id": "P01", "sample": 0, "skill_history": [...]}, ...],
  "skill_counts": {"problem_abstraction": 12, ...},
  "n_problems": 5,
  "n_samples": 5,
}
"""
import argparse, json, os, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import PROBLEMS
from algoskill import AlgoSkillMCTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=10,
                    help="how many problems to run")
    ap.add_argument("--n_samples", type=int, default=5,
                    help="trajectories per problem")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        data = json.load(open(args.out))
    else:
        data = {"trajectories": [], "skill_counts": {},
                "config": {"n_problems": args.n_problems,
                           "n_samples": args.n_samples,
                           "budget": args.budget}}

    done_keys = {(t["problem_id"], t["sample"]) for t in data["trajectories"]}
    problems = PROBLEMS[:args.n_problems]
    print(f"[skill-freq] running {args.n_problems} probs × {args.n_samples} samples "
          f"(budget={args.budget}), resume={len(done_keys)}", flush=True)

    for prob in problems:
        for s in range(args.n_samples):
            key = (prob["id"], s)
            if key in done_keys:
                continue
            solver = AlgoSkillMCTS(budget=args.budget, c_explore=1.0)
            t0 = time.time()
            try:
                out = solver.solve(prob)
            except Exception as e:
                print(f"  {prob['id']} sample={s} ERROR: {e}", flush=True)
                continue
            elapsed = time.time() - t0
            # solve() returns dict with code/verify_result/reward. We need
            # to access the root node's skill_history via the state - but
            # solve() returns best result, not the full trajectory. We need
            # to instrument AlgoSkillMCTS to also return the chosen path's
            # skill_history. Use what's available: the FINAL state's
            # skill_history if accessible via the solver's best path. As a
            # workaround, re-extract from the search tree by storing
            # skill_history in returned dict.
            #
            # NOTE: solve() in algoskill.py does NOT currently return
            # skill_history. We patched it in this task by adding the
            # following lines to AlgoSkillMCTS.solve() return:
            #     "skill_history": best_state.get("skill_history", []),
            sh = out.get("skill_history", [])
            data["trajectories"].append({
                "problem_id": prob["id"],
                "problem_name": prob["name"],
                "sample": s,
                "skill_history": sh,
                "passed": (out.get("verify_result") or {}).get("pass_rate", 0) == 1.0,
                "elapsed": elapsed,
            })
            # Live counter
            counter = Counter()
            for t in data["trajectories"]:
                for sk in t["skill_history"]:
                    counter[sk] += 1
            data["skill_counts"] = dict(counter)
            with open(args.out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  {prob['id']} sample={s} skills={sh} elapsed={elapsed:.0f}s",
                  flush=True)

    counter = Counter()
    for t in data["trajectories"]:
        for sk in t["skill_history"]:
            counter[sk] += 1
    print(f"\n[skill-freq final] {len(data['trajectories'])} trajectories", flush=True)
    for sk, c in counter.most_common():
        print(f"  {sk}: {c}", flush=True)


if __name__ == "__main__":
    main()
