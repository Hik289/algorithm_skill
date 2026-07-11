"""
Task 6 v2: HARDER cross-platform corpus generator.

Path A iteration after v1 calibration miss (direct baseline above target range).

Key changes vs v1:
- Stronger prompt: explicit "must be Codeforces Div1 / AtCoder ABC F / LeetCode
  Hard tier", "brute force O(N²) at N=10^5 must TLE within 2s timeout",
  "include adversarial corner cases", "force a NON-OBVIOUS algorithm".
- Add explicit "hard" skeletons that require harder data structures
  (segment tree, persistent DS, suffix array, SCC, network flow, etc.)
- The reference solution MUST use the hard algorithm not brute force.
- Increase verifier timeout to 8s; include 1 stress test in the 5 tests
  with maximum N (10^5) to filter out problems where brute force passes.

Output: raw_data_v2/cross_platform_corpus_v2.json
"""
import argparse, json, os, sys, time, random, re, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import call_llm_with_usage


CF_SKELETONS = [
    {"theme": "DP on subsets with non-trivial state (n<=20, 3 dims)", "level": "Div2-D/Div1-A",
     "tags": "bitmask_dp,hard"},
    {"theme": "Segment tree with lazy propagation for range update range query",
     "level": "Div1-A", "tags": "segtree,lazy"},
    {"theme": "Persistent data structure or offline queries with rollback DSU",
     "level": "Div1-B", "tags": "persistent,dsu_rollback"},
    {"theme": "Heavy-Light Decomposition queries on tree",
     "level": "Div1-A", "tags": "hld,tree"},
    {"theme": "Strongly Connected Components + condensation DP",
     "level": "Div1-A", "tags": "scc,graph_dp"},
    {"theme": "Mo's algorithm for offline range queries",
     "level": "Div1-A", "tags": "mo,offline"},
    {"theme": "Centroid decomposition for tree path counting",
     "level": "Div1-A", "tags": "centroid,tree"},
    {"theme": "Convex hull trick / Li Chao tree for DP optimization",
     "level": "Div1-B", "tags": "cht,dp_opt"},
    {"theme": "Matrix exponentiation for linear recurrence with N up to 10^18",
     "level": "Div1-A", "tags": "matrix_exp,modular"},
    {"theme": "Suffix array + LCP for substring queries",
     "level": "Div1-B", "tags": "sa,lcp,string"},
    {"theme": "Min-cost max-flow on small graph (n<=30)",
     "level": "Div1-A", "tags": "mcmf,flow"},
    {"theme": "Game theory: Sprague-Grundy with non-trivial nim variant",
     "level": "Div1-A", "tags": "game,grundy"},
    {"theme": "Digit DP with multiple states (sum / parity / max constraint)",
     "level": "Div2-E/Div1-A", "tags": "digit_dp,hard"},
    {"theme": "2-SAT with implication graph + SCC",
     "level": "Div1-A", "tags": "2sat,scc"},
]

ATC_SKELETONS = [
    {"theme": "Educational DP: complex bitmask state (e.g., TSP-like with weights)",
     "level": "ABC-F/ARC-D", "tags": "bitmask_dp"},
    {"theme": "Range tree with point query / sweep line",
     "level": "ABC-F", "tags": "sweepline,segtree"},
    {"theme": "Combinatorics with mod 10^9+7 and inclusion-exclusion (n up to 10^5)",
     "level": "ABC-F/ARC-C", "tags": "math,incexc"},
    {"theme": "Dijkstra on extended state graph (e.g., k-th shortest path or 2D state)",
     "level": "ARC-C", "tags": "dijkstra,state"},
    {"theme": "LCA + heavy path decomposition with path sum queries",
     "level": "ARC-D", "tags": "lca,hld"},
    {"theme": "Counting with FFT / NTT polynomial multiplication",
     "level": "ARC-D", "tags": "fft,polynomial"},
    {"theme": "Persistent segment tree for online k-th queries",
     "level": "ARC-D", "tags": "persistent_segtree"},
    {"theme": "Knapsack with multiple constraints and capacity 10^5",
     "level": "ABC-F", "tags": "knapsack,multi_dim"},
    {"theme": "Bipartite matching with Hopcroft-Karp",
     "level": "ARC-C", "tags": "matching,bipartite"},
    {"theme": "Aho-Corasick automaton for multi-pattern matching",
     "level": "ARC-D", "tags": "aho_corasick,string"},
    {"theme": "BIT (Fenwick) for inversion counting variants",
     "level": "ABC-F", "tags": "bit,inversions"},
    {"theme": "Treap or balanced BST for order statistics",
     "level": "ARC-D", "tags": "balanced_bst"},
    {"theme": "Min-cut max-flow with capacity scaling",
     "level": "ARC-D", "tags": "flow,scaling"},
    {"theme": "Tree DP with rerooting (all-vertices answer in O(N))",
     "level": "ABC-F", "tags": "tree_dp,rerooting"},
]

LC_SKELETONS = [
    {"theme": "Array: find optimal partition into K subarrays with multiple constraints",
     "level": "Hard+", "tags": "array,dp_partition"},
    {"theme": "String: minimum operations to transform with non-trivial cost",
     "level": "Hard+", "tags": "string,edit_distance"},
    {"theme": "Tree: query k-th ancestor with binary lifting (n up to 10^5)",
     "level": "Hard", "tags": "tree,binary_lift"},
    {"theme": "Graph: shortest path with state on visited subset",
     "level": "Hard+", "tags": "tsp,bitmask"},
    {"theme": "DP: 3D state with N,M,K up to 100 each, must memoize efficiently",
     "level": "Hard+", "tags": "dp,3d"},
    {"theme": "Greedy with priority queue + lazy deletion (k-events problem)",
     "level": "Hard", "tags": "greedy,heap_lazy"},
    {"theme": "Interval scheduling with weighted intervals + DP",
     "level": "Hard", "tags": "interval,dp"},
    {"theme": "Sliding window with multiple monotone deques",
     "level": "Hard", "tags": "sliding_window,deque"},
    {"theme": "Backtracking with bitmask + memoization for n<=20",
     "level": "Hard+", "tags": "bitmask_dp"},
    {"theme": "Hash map + Z-function for cyclic substring queries",
     "level": "Hard+", "tags": "z_function,cyclic"},
    {"theme": "Trie + DP for word-counting variants with n<=10^4",
     "level": "Hard", "tags": "trie,dp"},
    {"theme": "Union-Find with weighted ranks (offline)",
     "level": "Hard+", "tags": "dsu,weighted"},
]


GEN_PROMPT = """You are an EXPERT competitive programmer authoring a problem
for a HARD difficulty benchmark. Your goal: design a problem that REQUIRES
the non-obvious algorithm in the theme, and is HARD ENOUGH that:

(1) Naive brute force (e.g. O(N^2) for N=10^5) will TLE in 2 seconds.
(2) A correct solution needs the specific algorithm in the theme.
(3) Edge cases include: empty/single-element, max-size N (~10^5), adversarial
    structure (e.g. unbalanced tree, all-equal values, off-by-one).

Theme: "{theme}"
Style: {platform} (level: {level})
Tags: {tags}

CRITICAL: This must be hard enough that an LLM "directly generating Python"
without thinking carefully would likely:
- pick a wrong algorithm family (O(N^2) when O(N log N) is needed), or
- get edge cases wrong, or
- produce code that TLEs the largest test.

Output format (STRICT):

###STATEMENT###
[Self-contained problem statement, 150-300 words. Include input format, output
format, constraints, and 1-2 small worked examples inline (small N, easy to
verify by hand). The constraints MUST include a hard upper bound (e.g.
N <= 10^5) that forces the non-obvious algorithm.]

###CONSTRAINTS###
[One line: e.g. "1 <= N <= 10^5; 1 <= a[i] <= 10^9; 1 <= Q <= 10^5"]

###REFERENCE_CODE###
```python
import sys
from typing import *
input = sys.stdin.readline
def solve():
    # IMPLEMENT THE OPTIMAL ALGORITHM (must use the theme's data structure
    # or technique; brute force is unacceptable here).
    pass
solve()
```

###TESTS###
[Exactly 5 distinct test cases, one per line, using "|" to separate input
lines. Make tests progressively harder:
  Test 1: trivial (single element / empty)
  Test 2: small example matching the worked example
  Test 3: medium (N ~ 100, varied input)
  Test 4: large (N ~ 1000, adversarial structure)
  Test 5: stress test (N = max constraint, e.g. 10^5)
Do NOT include expected outputs; they'll be computed from your reference.]
"""


def _parse_response(text):
    def section(name):
        m = re.search(rf"###{name}###\s*\n(.*?)(?:\n###|\Z)", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    stmt = section("STATEMENT")
    constr = section("CONSTRAINTS")
    code_block = section("REFERENCE_CODE")
    tests_block = section("TESTS")
    code = ""
    m = re.search(r"```python\s*\n(.*?)```", code_block, re.DOTALL)
    if m:
        code = m.group(1).strip()
    elif code_block.startswith("```"):
        m = re.search(r"```\s*\n(.*?)```", code_block, re.DOTALL)
        if m: code = m.group(1).strip()
    else:
        code = code_block.strip()
    tests = []
    for line in tests_block.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        stdin = "\n".join(part.strip() for part in line.split("|"))
        tests.append(stdin)
    return stmt, constr, code, tests


def _run_code(code, stdin_data, timeout=8.0):
    try:
        p = subprocess.run([sys.executable, "-c", code],
                           input=stdin_data, capture_output=True, text=True,
                           timeout=timeout)
        return p.stdout.strip(), p.returncode, p.stderr[:300]
    except subprocess.TimeoutExpired:
        return "", "TLE", ""
    except Exception as e:
        return "", "ERR", str(e)[:200]


def build_one(platform, skeleton, oracle="judge"):
    if platform == "CF":
        platform_name = "Codeforces"
    elif platform == "ATC":
        platform_name = "AtCoder"
    elif platform == "LC":
        platform_name = "LeetCode Hard"
    else:
        platform_name = platform
    prompt = GEN_PROMPT.format(
        platform=platform_name,
        level=skeleton["level"],
        theme=skeleton["theme"],
        tags=skeleton["tags"],
    )
    out = call_llm_with_usage(prompt, backbone=oracle,
                              temperature=0.9, max_tokens=4000)
    text = out["text"]
    stmt, constr, code, raw_tests = _parse_response(text)
    verified_tests = []
    if code:
        for stdin_data in raw_tests:
            stdout, rc, err = _run_code(code, stdin_data, timeout=8.0)
            if rc == 0 and stdout:
                verified_tests.append({"input": stdin_data, "output": stdout})
    return {
        "platform": platform,
        "platform_name": platform_name,
        "theme": skeleton["theme"],
        "tags": skeleton["tags"],
        "level": skeleton["level"],
        "description": stmt,
        "constraints": constr,
        "reference_code": code[:5000],
        "tests": verified_tests,
        "n_tests_kept": len(verified_tests),
        "tokens": out["tokens"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cf", type=int, default=100)
    ap.add_argument("--n_atc", type=int, default=100)
    ap.add_argument("--n_lc", type=int, default=75)
    ap.add_argument("--oracle", default="judge")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        out = json.load(open(args.out))
        print(f"resume: {len(out)} existing", flush=True)
    else:
        out = {}

    plan = (
        [("CF", CF_SKELETONS, args.n_cf)] +
        [("ATC", ATC_SKELETONS, args.n_atc)] +
        [("LC", LC_SKELETONS, args.n_lc)]
    )
    total_target = args.n_cf + args.n_atc + args.n_lc

    for platform, skeletons, n_total in plan:
        have = sum(1 for v in out.values() if v.get("platform") == platform)
        i = have + 1
        attempts = 0
        while have < n_total and attempts < n_total * 4:
            attempts += 1
            sk = rng.choice(skeletons)
            pid = f"{platform}_{i:03d}"
            if pid in out:
                i += 1; continue
            t0 = time.time()
            try:
                v = build_one(platform, sk, args.oracle)
            except Exception as e:
                print(f"  {pid} ERR: {e}", flush=True); continue
            elapsed = time.time() - t0
            v["id"] = pid; v["elapsed"] = elapsed
            if v["n_tests_kept"] < 3:
                print(f"  {pid} REJECT tests={v['n_tests_kept']} "
                      f"theme='{sk['theme'][:40]}' ({elapsed:.0f}s)", flush=True)
                continue
            out[pid] = v; have += 1
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  [{platform} {have}/{n_total} total {len(out)}/{total_target}] "
                  f"{pid} tests={v['n_tests_kept']} {elapsed:.0f}s", flush=True)
            i += 1

    n_kept = sum(1 for v in out.values() if v.get("n_tests_kept", 0) >= 3)
    print(f"\nDone. {len(out)} problems, {n_kept} usable.", flush=True)


if __name__ == "__main__":
    main()
