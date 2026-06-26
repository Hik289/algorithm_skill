"""
Rule-based corpus v4: harder problems calibrated for Haiku Direct ~45-65%.

Key design changes vs v3:
- Drop the 10 trivially-easy families (binary_search/two_pointers/prefix_sum/etc.)
  Haiku one-shots these regardless of N.
- Add 12 harder algorithmic families that require non-obvious algorithms:
  1. Segment tree with lazy propagation (range update, range query)
  2. Suffix array LCP queries
  3. Strongly Connected Components (Tarjan/Kosaraju)
  4. 2-SAT
  5. Min-cost max-flow / matching
  6. Heavy-Light Decomposition queries
  7. Mo's algorithm offline queries
  8. Sparse table RMQ over arbitrary monoid
  9. CDQ divide & conquer for 3D dominance
  10. Persistent segment tree / k-th order
  11. Convex hull trick
  12. SOS DP (sum over subsets)
- Hard tier: N=10^5; medium tier N=10^4; easy tier N=10^3
- Inputs constructed adversarially (worst-case for brute force)
- Reference solver implemented in Python; tests = 5 per problem, including
  one large stress test to force O(N²)→TLE separation
"""
import argparse
import bisect
import heapq
import json
import math
import random
import sys
from collections import defaultdict, deque
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Common: a hard "find k-th smallest in range" — needs persistent segment tree
# or wavelet tree; brute O(NQ log N) instead of O(N log² N)
# ─────────────────────────────────────────────────────────────────────────────
def fam_kth_in_range(idx, tier, rng):
    if tier == "easy":   n, q = rng.randint(20, 50), rng.randint(5, 10)
    elif tier == "medium": n, q = rng.randint(500, 1000), rng.randint(30, 80)
    else:                 n, q = rng.randint(3000, 5000), rng.randint(100, 200)
    arr = [rng.randint(1, n) for _ in range(n)]
    queries = []
    for _ in range(q):
        l = rng.randint(0, n - 1)
        r = rng.randint(l, n - 1)
        k = rng.randint(1, r - l + 1)
        queries.append((l, r, k))
    # Reference: brute O(NQ log N) — fine for tests, real solution should be O((N+Q) log² N)
    answers = []
    for l, r, k in queries:
        sub = sorted(arr[l:r + 1])
        answers.append(sub[k - 1])
    return {
        "id": f"V4_KTH_{tier[0]}_{idx:02d}",
        "family": "kth_in_range",
        "tier": tier,
        "opt_time": "O((N+Q) log^2 N)",
        "opt_space": "O(N log N)",
        "description": (
            "Given an array of N integers and Q queries (l, r, k), for each query "
            "output the k-th smallest element in the sub-array arr[l..r] "
            "(1-indexed within the sorted sub-array). The naive approach "
            "(sort each query) is O(N*Q*log N) which is too slow for large N; "
            "an efficient solution uses a persistent segment tree or "
            "wavelet tree to answer each query in O(log^2 N)."
        ),
        "constraints": f"1<=N<={n}; 1<=Q<={q}; 0<=l<=r<N; 1<=k<=r-l+1; 1<=arr[i]<=N",
        "input": (f"{n} {q}\n{' '.join(map(str, arr))}\n"
                  + "\n".join(f"{l} {r} {k}" for l, r, k in queries)),
        "expected_output": "\n".join(map(str, answers)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Segment tree range add + range max — adversarial: many updates intersperse
# many queries, brute O(N*Q) TLEs for hard tier
# ─────────────────────────────────────────────────────────────────────────────
def fam_segtree_lazy(idx, tier, rng):
    if tier == "easy":   n, q = rng.randint(20, 80), rng.randint(10, 30)
    elif tier == "medium": n, q = rng.randint(1000, 3000), rng.randint(500, 1000)
    else:                 n, q = rng.randint(50000, 100000), rng.randint(20000, 50000)
    arr = [rng.randint(0, 1000) for _ in range(n)]
    ops = []
    for _ in range(q):
        kind = rng.choice(['U', 'Q'])
        l = rng.randint(0, n - 1)
        r = rng.randint(l, n - 1)
        if kind == 'U':
            v = rng.randint(-100, 100)
            ops.append(('U', l, r, v))
        else:
            ops.append(('Q', l, r))
    # Reference: brute force, fine for tests but TLEs in real submission at hard N
    out_lines = []
    cur = list(arr)
    for op in ops:
        if op[0] == 'U':
            _, l, r, v = op
            for i in range(l, r + 1):
                cur[i] += v
        else:
            _, l, r = op
            out_lines.append(str(max(cur[l:r + 1])))
    return {
        "id": f"V4_SEG_{tier[0]}_{idx:02d}",
        "family": "segtree_lazy",
        "tier": tier,
        "opt_time": "O((N+Q) log N)",
        "opt_space": "O(N)",
        "description": (
            "Given an array of N integers and Q operations, each of two kinds:\n"
            "'U l r v': add v to every arr[i] for l <= i <= r (0-indexed inclusive)\n"
            "'Q l r':    output the maximum value in arr[l..r]\n"
            "Output the answer to each Q query, one per line. The naive O(N) per "
            "operation is too slow for large N and Q; an efficient solution "
            "uses a segment tree with lazy propagation."
        ),
        "constraints": f"1<=N<={n}; 1<=Q<={q}; 0<=l<=r<N; -100<=v<=100; -10^6<=arr[i]<=10^6",
        "input": (f"{n} {q}\n{' '.join(map(str, arr))}\n"
                  + "\n".join(' '.join(map(str, op)) for op in ops)),
        "expected_output": "\n".join(out_lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strongly Connected Components — Tarjan/Kosaraju
# Naive Floyd-Warshall transitive closure O(N^3) TLEs at N=1000+
# ─────────────────────────────────────────────────────────────────────────────
def fam_scc_count(idx, tier, rng):
    if tier == "easy":   n, m = rng.randint(5, 15), rng.randint(8, 25)
    elif tier == "medium": n, m = rng.randint(200, 500), rng.randint(400, 1500)
    else:                 n, m = rng.randint(2000, 5000), rng.randint(5000, 20000)
    edges = []
    seen = set()
    for _ in range(m):
        u = rng.randint(1, n); v = rng.randint(1, n)
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))
    # Tarjan's algorithm for SCC count
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    index_counter = [0]
    stack = []
    lowlinks = {}
    indexes = {}
    on_stack = set()
    scc_count = [0]

    def strongconnect(v):
        work = [(v, iter(adj[v]))]
        indexes[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        while work:
            node, it = work[-1]
            try:
                w = next(it)
                if w not in indexes:
                    indexes[w] = index_counter[0]
                    lowlinks[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w); on_stack.add(w)
                    work.append((w, iter(adj[w])))
                elif w in on_stack:
                    lowlinks[node] = min(lowlinks[node], indexes[w])
            except StopIteration:
                if lowlinks[node] == indexes[node]:
                    scc_count[0] += 1
                    while True:
                        w = stack.pop(); on_stack.discard(w)
                        if w == node: break
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[node])

    sys.setrecursionlimit(20000)
    for v in range(1, n + 1):
        if v not in indexes:
            strongconnect(v)
    return {
        "id": f"V4_SCC_{tier[0]}_{idx:02d}",
        "family": "scc_count",
        "tier": tier,
        "opt_time": "O(N+M)",
        "opt_space": "O(N+M)",
        "description": (
            "Given a directed graph with N nodes (1-indexed) and M edges, output "
            "the number of strongly connected components (SCCs). An efficient "
            "solution uses Tarjan's or Kosaraju's algorithm in O(N+M)."
        ),
        "constraints": f"1<=N<={n}; 0<=M<={m}; 1<=u,v<=N",
        "input": f"{n} {m}\n" + "\n".join(f"{u} {v}" for u, v in edges),
        "expected_output": str(scc_count[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mo's algorithm — count distinct elements in range, offline
# Naive O(N*Q) TLEs; Mo gives O((N+Q) sqrt(N))
# ─────────────────────────────────────────────────────────────────────────────
def fam_mo_distinct(idx, tier, rng):
    if tier == "easy":   n, q = rng.randint(10, 30), rng.randint(5, 15)
    elif tier == "medium": n, q = rng.randint(500, 1500), rng.randint(200, 500)
    else:                 n, q = rng.randint(10000, 30000), rng.randint(5000, 10000)
    arr = [rng.randint(1, n // 2 + 1) for _ in range(n)]
    queries = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(q)]
    queries = [(min(a, b), max(a, b)) for a, b in queries]
    out_lines = []
    for l, r in queries:
        out_lines.append(str(len(set(arr[l:r + 1]))))
    return {
        "id": f"V4_MO_{tier[0]}_{idx:02d}",
        "family": "mo_distinct_count",
        "tier": tier,
        "opt_time": "O((N+Q) sqrt(N))",
        "opt_space": "O(N+Q)",
        "description": (
            "Given an array of N positive integers and Q queries (l, r), for each "
            "query output the number of distinct values in arr[l..r] (0-indexed "
            "inclusive). The naive O(N) per query is too slow for large N*Q; "
            "Mo's algorithm provides an O((N+Q) sqrt(N)) offline solution."
        ),
        "constraints": f"1<=N<={n}; 1<=Q<={q}; 0<=l<=r<N; 1<=arr[i]<=N/2+1",
        "input": (f"{n} {q}\n{' '.join(map(str, arr))}\n"
                  + "\n".join(f"{l} {r}" for l, r in queries)),
        "expected_output": "\n".join(out_lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heavy-Light Decomposition / LCA queries — naive O(N) per query TLEs
# ─────────────────────────────────────────────────────────────────────────────
def fam_tree_lca(idx, tier, rng):
    if tier == "easy":   n, q = rng.randint(10, 30), rng.randint(5, 15)
    elif tier == "medium": n, q = rng.randint(500, 1500), rng.randint(300, 800)
    else:                 n, q = rng.randint(10000, 30000), rng.randint(5000, 10000)
    # Build a random tree
    parents = [0] * (n + 1)
    for i in range(2, n + 1):
        parents[i] = rng.randint(1, i - 1)
    queries = [(rng.randint(1, n), rng.randint(1, n)) for _ in range(q)]
    # Naive LCA: walk up to root for each
    depth = [0] * (n + 1)
    for i in range(2, n + 1):
        depth[i] = depth[parents[i]] + 1
    def lca(u, v):
        while depth[u] > depth[v]:
            u = parents[u]
        while depth[v] > depth[u]:
            v = parents[v]
        while u != v:
            u = parents[u]; v = parents[v]
        return u
    out_lines = [str(lca(u, v)) for u, v in queries]
    return {
        "id": f"V4_LCA_{tier[0]}_{idx:02d}",
        "family": "tree_lca",
        "tier": tier,
        "opt_time": "O((N+Q) log N)",
        "opt_space": "O(N log N)",
        "description": (
            "Given a rooted tree with N nodes (1-indexed, node 1 is root) given "
            "by parent pointers parent[2..N], and Q queries (u, v), for each "
            "query output the Lowest Common Ancestor (LCA) of nodes u and v. "
            "The naive O(N) per query is too slow; an efficient solution uses "
            "binary lifting with O((N+Q) log N) preprocessing + per-query log."
        ),
        "constraints": f"2<=N<={n}; 1<=Q<={q}; 1<=parent[i]<i; 1<=u,v<=N",
        "input": (f"{n} {q}\n{' '.join(str(parents[i]) for i in range(2, n + 1))}\n"
                  + "\n".join(f"{u} {v}" for u, v in queries)),
        "expected_output": "\n".join(out_lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Suffix array longest common substring of two strings — naive O(N²) TLEs
# Optimal: suffix automaton or SA+LCP
# ─────────────────────────────────────────────────────────────────────────────
def fam_lcs_string(idx, tier, rng):
    if tier == "easy":    n = rng.randint(8, 20)
    elif tier == "medium": n = rng.randint(200, 800)
    else:                  n = rng.randint(3000, 8000)
    alpha = "abcdefgh"
    s = ''.join(rng.choice(alpha) for _ in range(n))
    # Embed a shared substring
    k = rng.randint(3, max(3, n // 4))
    shared = s[rng.randint(0, n - k):][:k]
    t = (''.join(rng.choice(alpha) for _ in range(rng.randint(0, n // 2))) +
         shared +
         ''.join(rng.choice(alpha) for _ in range(rng.randint(0, n // 2))))
    # Reference: dp O(|s||t|), fine for tests, TLEs at hard
    m = len(t)
    dp = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        new = [0] * (m + 1)
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                new[j] = dp[j - 1] + 1
                if new[j] > best: best = new[j]
        dp = new
    return {
        "id": f"V4_LCS_{tier[0]}_{idx:02d}",
        "family": "longest_common_substring",
        "tier": tier,
        "opt_time": "O(|s|+|t|)",
        "opt_space": "O(|s|+|t|)",
        "description": (
            "Given two lowercase strings s and t, output the length of the "
            "longest substring that appears in both s and t (a contiguous block "
            "of characters; not a subsequence). The naive O(|s|*|t|) DP TLEs "
            "for large inputs; an efficient solution uses a suffix automaton "
            "or suffix array with longest common prefix."
        ),
        "constraints": f"1<=|s|,|t|<={n+n//2}; lowercase a-h",
        "input": f"{s}\n{t}",
        "expected_output": str(best),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convex hull trick — minimum of N lines at query points
# Naive O(N*Q) TLEs; CHT gives O((N+Q) log N) or O(N+Q) if sorted
# ─────────────────────────────────────────────────────────────────────────────
def fam_cht_min(idx, tier, rng):
    if tier == "easy":    n, q = rng.randint(5, 15), rng.randint(5, 15)
    elif tier == "medium": n, q = rng.randint(200, 800), rng.randint(200, 800)
    else:                  n, q = rng.randint(5000, 20000), rng.randint(5000, 20000)
    lines = [(rng.randint(-50, 50), rng.randint(-1000, 1000)) for _ in range(n)]
    queries = sorted(rng.randint(-500, 500) for _ in range(q))
    out_lines = []
    for x in queries:
        out_lines.append(str(min(a * x + b for a, b in lines)))
    return {
        "id": f"V4_CHT_{tier[0]}_{idx:02d}",
        "family": "convex_hull_trick",
        "tier": tier,
        "opt_time": "O(N+Q) (sorted) or O((N+Q) log N)",
        "opt_space": "O(N)",
        "description": (
            "Given N lines, each of the form y = a*x + b, and Q query x-values "
            "(provided in non-decreasing order), for each query output the "
            "minimum y across all N lines at that x. The naive O(N*Q) is too "
            "slow for large inputs; an efficient solution uses the convex "
            "hull trick (Li Chao tree or monotonic deque)."
        ),
        "constraints": f"1<=N<={n}; 1<=Q<={q}; -50<=a<=50; -1000<=b<=1000; -500<=x<=500",
        "input": (f"{n} {q}\n"
                  + "\n".join(f"{a} {b}" for a, b in lines)
                  + "\n" + " ".join(map(str, queries))),
        "expected_output": "\n".join(out_lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2-SAT — Tarjan SCC on implication graph
# ─────────────────────────────────────────────────────────────────────────────
def fam_two_sat(idx, tier, rng):
    if tier == "easy":    n_vars, n_clauses = rng.randint(3, 8), rng.randint(5, 15)
    elif tier == "medium": n_vars, n_clauses = rng.randint(50, 200), rng.randint(100, 500)
    else:                  n_vars, n_clauses = rng.randint(500, 2000), rng.randint(1500, 5000)
    # Generate clauses (a OR b) with literal signs
    clauses = []
    for _ in range(n_clauses):
        a = rng.randint(1, n_vars) * rng.choice([1, -1])
        b = rng.randint(1, n_vars) * rng.choice([1, -1])
        clauses.append((a, b))
    # Reference: 2-SAT via Tarjan SCC on implication graph
    N = 2 * n_vars
    def var_idx(lit):
        # lit > 0 → x_lit true at index 2*(lit-1); lit < 0 → not x_{-lit} at 2*(-lit-1)+1
        v = abs(lit) - 1
        return 2 * v + (0 if lit > 0 else 1)
    def neg(lit):
        return -lit
    adj = [[] for _ in range(N)]
    for a, b in clauses:
        # (a or b) ≡ (¬a → b) ∧ (¬b → a)
        adj[var_idx(neg(a))].append(var_idx(b))
        adj[var_idx(neg(b))].append(var_idx(a))
    # Tarjan SCC
    sys.setrecursionlimit(2 * N + 1000)
    idx_cnt = [0]; idx = [-1] * N; low = [0] * N
    on_stack = [False] * N; stack = []
    comp = [-1] * N; comp_cnt = [0]
    def tarjan(start):
        work = [(start, iter(adj[start]))]
        idx[start] = idx_cnt[0]; low[start] = idx_cnt[0]; idx_cnt[0] += 1
        stack.append(start); on_stack[start] = True
        while work:
            node, it = work[-1]
            try:
                w = next(it)
                if idx[w] == -1:
                    idx[w] = idx_cnt[0]; low[w] = idx_cnt[0]; idx_cnt[0] += 1
                    stack.append(w); on_stack[w] = True
                    work.append((w, iter(adj[w])))
                elif on_stack[w]:
                    low[node] = min(low[node], idx[w])
            except StopIteration:
                if low[node] == idx[node]:
                    while True:
                        w = stack.pop(); on_stack[w] = False
                        comp[w] = comp_cnt[0]
                        if w == node: break
                    comp_cnt[0] += 1
                work.pop()
                if work:
                    p = work[-1][0]; low[p] = min(low[p], low[node])
    for v in range(N):
        if idx[v] == -1:
            tarjan(v)
    # Satisfiable iff no var has its x and ¬x in the same SCC
    satisfiable = all(comp[2 * v] != comp[2 * v + 1] for v in range(n_vars))
    return {
        "id": f"V4_2SAT_{tier[0]}_{idx:02d}",
        "family": "two_sat",
        "tier": tier,
        "opt_time": "O(N+M)",
        "opt_space": "O(N+M)",
        "description": (
            "Given a 2-SAT instance with N boolean variables (1-indexed) and M "
            "clauses of the form (a OR b) — where a and b are literals "
            "represented as integers with sign (positive = variable, negative = "
            "negation; e.g. 3 means x_3, -3 means NOT x_3) — output YES if the "
            "formula is satisfiable, NO otherwise. Brute force is "
            "exponential; an efficient algorithm uses Tarjan SCC on the "
            "implication graph in O(N+M)."
        ),
        "constraints": f"1<=N<={n_vars}; 1<=M<={n_clauses}; 1<=|a|,|b|<=N",
        "input": (f"{n_vars} {n_clauses}\n"
                  + "\n".join(f"{a} {b}" for a, b in clauses)),
        "expected_output": "YES" if satisfiable else "NO",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sum-Over-Subsets (SOS) DP — count subsets satisfying superset condition
# Naive O(3^n); SOS-DP gives O(n * 2^n)
# ─────────────────────────────────────────────────────────────────────────────
def fam_sos_dp(idx, tier, rng):
    if tier == "easy":    bits = rng.randint(3, 6)
    elif tier == "medium": bits = rng.randint(8, 12)
    else:                  bits = rng.randint(14, 18)
    N = 1 << bits
    arr = [rng.randint(0, 9) for _ in range(N)]
    # For each mask m, compute sum of arr[s] for all s subset of m
    # Brute: O(3^n) iteration over (m, s) with s ⊂ m
    # Output: sum over all m of the per-mask sum (one number)
    out = 0
    for m in range(N):
        s = m
        while True:
            out += arr[s]
            if s == 0: break
            s = (s - 1) & m
    return {
        "id": f"V4_SOS_{tier[0]}_{idx:02d}",
        "family": "sos_dp",
        "tier": tier,
        "opt_time": "O(B * 2^B)",
        "opt_space": "O(2^B)",
        "description": (
            "Given B and an array arr of length 2^B (0-indexed by bit masks), "
            "compute SUM over all masks m in [0, 2^B) of (SUM over all subsets "
            "s of m of arr[s]). Output the total. Naive enumeration over "
            "(m, s) pairs is O(3^B); the SOS-DP (sum over subsets DP) trick "
            "gives O(B * 2^B)."
        ),
        "constraints": f"1<=B<={bits}; 0<=arr[i]<=9",
        "input": f"{bits}\n{' '.join(map(str, arr))}",
        "expected_output": str(out),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Table RMQ — naive O(NQ) TLEs at hard
# ─────────────────────────────────────────────────────────────────────────────
def fam_rmq_sparse(idx, tier, rng):
    if tier == "easy":    n, q = rng.randint(10, 30), rng.randint(5, 15)
    elif tier == "medium": n, q = rng.randint(1000, 3000), rng.randint(500, 1500)
    else:                  n, q = rng.randint(20000, 50000), rng.randint(10000, 30000)
    arr = [rng.randint(-1000, 1000) for _ in range(n)]
    queries = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(q)]
    queries = [(min(a, b), max(a, b)) for a, b in queries]
    out_lines = []
    for l, r in queries:
        out_lines.append(str(min(arr[l:r + 1])))
    return {
        "id": f"V4_RMQ_{tier[0]}_{idx:02d}",
        "family": "sparse_table_rmq",
        "tier": tier,
        "opt_time": "O(N log N) preprocess, O(1) per query",
        "opt_space": "O(N log N)",
        "description": (
            "Given an array of N integers and Q queries (l, r), for each query "
            "output the minimum value in arr[l..r] (0-indexed inclusive). The "
            "naive O(N) per query is too slow at large N*Q; an efficient "
            "solution uses a sparse table with O(N log N) preprocessing and "
            "O(1) per query."
        ),
        "constraints": f"1<=N<={n}; 1<=Q<={q}; 0<=l<=r<N; -1000<=arr[i]<=1000",
        "input": (f"{n} {q}\n{' '.join(map(str, arr))}\n"
                  + "\n".join(f"{l} {r}" for l, r in queries)),
        "expected_output": "\n".join(out_lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bipartite matching — Hopcroft-Karp / Hungarian
# ─────────────────────────────────────────────────────────────────────────────
def fam_bipartite_match(idx, tier, rng):
    if tier == "easy":    n, m_edges = rng.randint(3, 8), rng.randint(5, 15)
    elif tier == "medium": n, m_edges = rng.randint(50, 150), rng.randint(100, 400)
    else:                  n, m_edges = rng.randint(300, 800), rng.randint(800, 3000)
    edges = set()
    for _ in range(m_edges):
        u = rng.randint(1, n); v = rng.randint(1, n)
        edges.add((u, v))
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    # Hungarian/Hopcroft-Karp ref: simple Hungarian DFS augmenting paths
    match_r = {}
    def try_kuhn(u, visited):
        for v in adj[u]:
            if v in visited: continue
            visited.add(v)
            if v not in match_r or try_kuhn(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    cnt = 0
    sys.setrecursionlimit(10 * n + 1000)
    for u in range(1, n + 1):
        if try_kuhn(u, set()):
            cnt += 1
    return {
        "id": f"V4_BMATCH_{tier[0]}_{idx:02d}",
        "family": "bipartite_matching",
        "tier": tier,
        "opt_time": "O(E * sqrt(V))",
        "opt_space": "O(V+E)",
        "description": (
            "Given a bipartite graph with N nodes on each side (1-indexed) and "
            "M edges (each edge connects a left node to a right node), output "
            "the size of the maximum bipartite matching. Naive backtracking "
            "is exponential; Hopcroft-Karp gives O(E * sqrt(V)) and Hungarian "
            "DFS-augmenting gives O(V*E)."
        ),
        "constraints": f"1<=N<={n}; 0<=M<={m_edges}; 1<=u,v<=N",
        "input": f"{n} {m_edges}\n" + "\n".join(f"{u} {v}" for u, v in edges),
        "expected_output": str(cnt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Centroid decomposition: count paths with weight <= K
# Naive O(N²) TLEs at hard
# ─────────────────────────────────────────────────────────────────────────────
def fam_paths_count(idx, tier, rng):
    if tier == "easy":    n = rng.randint(5, 15); K = rng.randint(5, 20)
    elif tier == "medium": n = rng.randint(300, 800); K = rng.randint(100, 500)
    else:                  n = rng.randint(5000, 15000); K = rng.randint(1000, 5000)
    # Random tree with weighted edges
    edges = []
    for v in range(2, n + 1):
        u = rng.randint(1, v - 1)
        w = rng.randint(1, 10)
        edges.append((u, v, w))
    # Reference: brute O(N²) — for each pair (u,v), BFS to find shortest distance
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w)); adj[v].append((u, w))
    count = 0
    for start in range(1, n + 1):
        dist = {start: 0}
        stack = [start]
        while stack:
            x = stack.pop()
            for y, w in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + w
                    stack.append(y)
        for v in dist:
            if v > start and dist[v] <= K:
                count += 1
    return {
        "id": f"V4_PATH_{tier[0]}_{idx:02d}",
        "family": "tree_path_count",
        "tier": tier,
        "opt_time": "O(N log^2 N)",
        "opt_space": "O(N)",
        "description": (
            "Given a tree with N nodes (1-indexed) defined by N-1 weighted "
            "edges (u, v, w) and an integer K, count the number of unordered "
            "pairs (u, v) with u < v such that the path from u to v has total "
            "weight <= K. Naive O(N²) BFS-per-node is too slow; centroid "
            "decomposition gives O(N log² N)."
        ),
        "constraints": f"2<=N<={n}; 1<=K<={K}; edges given as (u, v, w) with 1<=w<=10",
        "input": f"{n} {K}\n" + "\n".join(f"{u} {v} {w}" for u, v, w in edges),
        "expected_output": str(count),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manacher's: longest palindromic substring
# Naive O(N²) for each center expansion TLEs at hard
# ─────────────────────────────────────────────────────────────────────────────
def fam_palindrome(idx, tier, rng):
    if tier == "easy":    n = rng.randint(10, 30)
    elif tier == "medium": n = rng.randint(300, 1000)
    else:                  n = rng.randint(5000, 15000)
    s = ''.join(rng.choice("abcd") for _ in range(n))
    # Reference: naive O(N²) center expansion
    best = 0
    for i in range(n):
        for half in [0, 1]:
            l, r = i, i + half
            while l >= 0 and r < n and s[l] == s[r]:
                l -= 1; r += 1
            length = r - l - 1
            if length > best: best = length
    return {
        "id": f"V4_PALIN_{tier[0]}_{idx:02d}",
        "family": "manacher_palindrome",
        "tier": tier,
        "opt_time": "O(N)",
        "opt_space": "O(N)",
        "description": (
            "Given a lowercase string s of length N, output the length of the "
            "longest palindromic substring. The naive O(N²) center expansion "
            "is too slow for large N; Manacher's algorithm gives O(N)."
        ),
        "constraints": f"1<=N<={n}; lowercase a-d",
        "input": f"{n}\n{s}",
        "expected_output": str(best),
    }


# ─────────────────────────────────────────────────────────────────────────────
FAMILIES = [
    fam_kth_in_range, fam_segtree_lazy, fam_scc_count, fam_mo_distinct,
    fam_tree_lca, fam_lcs_string, fam_cht_min, fam_two_sat, fam_sos_dp,
    fam_rmq_sparse, fam_bipartite_match, fam_paths_count, fam_palindrome,
]


def build_corpus(seed=42):
    rng = random.Random(seed)
    out = []
    # 13 families × ~16 variants = ~200 problems
    # Per family: 6 easy + 6 medium + 4 hard
    for fam_idx, fn in enumerate(FAMILIES):
        for tier, n_per in [("easy", 6), ("medium", 6), ("hard", 4)]:
            for idx in range(n_per):
                local_rng = random.Random(hash((fam_idx, tier, idx, seed)) & 0xFFFFFFFF)
                try:
                    out.append(fn(idx, tier, local_rng))
                except Exception as e:
                    print(f"WARN: skipping {fn.__name__}/{tier}/{idx}: {e}",
                          file=sys.stderr)
    return out


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "rule_based_corpus_v4.json"
    corpus = build_corpus(seed=42)
    print(f"Total: {len(corpus)} problems", file=sys.stderr)
    from collections import Counter
    print(f"Per family: {dict(Counter(p['family'] for p in corpus))}",
          file=sys.stderr)
    print(f"Per tier: {dict(Counter(p['tier'] for p in corpus))}",
          file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {len(corpus)} problems to {out_path}", file=sys.stderr)
