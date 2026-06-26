"""
Algorithmic skill definitions and prompt templates for AlgoSkill.
Each skill is: {name, precondition_check, prompt_template}
"""

SKILL_PROMPTS = {
    "problem_abstraction": {
        "description": "Map the raw problem to a structured problem type (array, graph, DP, etc.) and task type (search, optimization, counting, etc.).",
        "prompt": """\
You are an expert algorithm designer applying the PROBLEM ABSTRACTION skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan (may be empty):
{current_plan}

Apply the problem abstraction skill:
1. Identify the data structure type (array, string, graph, tree, interval, matrix, etc.)
2. Identify the task type (search, counting, optimization, decision, matching, partition, etc.)
3. Identify key patterns or structures in the problem
4. Suggest 1-2 algorithmic directions based on the abstraction

Output a structured analysis. Be concise but precise.
""",
    },

    "constraint_reading": {
        "description": "Read input constraints to determine feasible complexity class.",
        "prompt": """\
You are an expert algorithm designer applying the CONSTRAINT READING skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the constraint reading skill:
1. Parse all input size constraints (n, m, k, etc.)
2. Map each constraint to a feasible algorithm complexity:
   - n ≤ 20: O(2^n) or O(n*2^n) — exponential OK
   - n ≤ 500: O(n^3) — cubic OK
   - n ≤ 5000: O(n^2) — quadratic OK
   - n ≤ 10^5: O(n log n) or O(n) — near-linear required
   - n ≤ 10^9: O(log n) or O(1) — logarithmic or closed-form required
3. Identify the target complexity budget
4. List algorithm families that fit

Output the constraint analysis and feasible algorithm families.
""",
    },

    "brute_force": {
        "description": "Construct a correct brute-force baseline and identify its bottleneck.",
        "prompt": """\
You are an expert algorithm designer applying the BRUTE FORCE FIRST skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the brute force skill:
1. Design the simplest possible correct algorithm (even if slow)
2. State its time complexity
3. Identify the bottleneck — what makes it slow
4. Suggest how the bottleneck could be addressed

This gives us a correct reference to optimize from.
""",
    },

    "state_design": {
        "description": "Design DP states or search states to capture sufficient information.",
        "prompt": """\
You are an expert algorithm designer applying the STATE DESIGN skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the state design skill for dynamic programming or search:
1. Ask: "What information is sufficient to make future decisions?"
2. Define the DP state (e.g., dp[i], dp[i][j], dp[mask], dp[i][remaining])
3. Write the recurrence or transition
4. State base cases
5. State the final answer expression
6. Estimate the total number of states × transitions → complexity

Be precise about indices and what each state variable represents.
""",
    },

    "monotonicity_detection": {
        "description": "Detect monotone feasibility predicate to enable binary search.",
        "prompt": """\
You are an expert algorithm designer applying the MONOTONICITY DETECTION skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the monotonicity detection skill:
1. Check if the problem has a decision version: F(x) = "Is x achievable/feasible?"
2. Check if F is monotone: if F(x)=true then F(x')=true for all x'>x (or x'<x)
3. If monotone, propose binary search on the answer
4. Define the feasibility check function clearly
5. State what to binary search on and the search range

If the problem does NOT have monotone structure, explain why and suggest alternatives.
""",
    },

    "data_structure_substitution": {
        "description": "Replace repeated scans with an appropriate data structure.",
        "prompt": """\
You are an expert algorithm designer applying the DATA STRUCTURE SUBSTITUTION skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the data structure substitution skill:
1. Identify repeated operations in the current plan:
   - Range sum/min/max queries → segment tree or Fenwick tree
   - Minimum/maximum tracking with insert/delete → heap
   - Connectivity queries → Union-Find
   - Nearest smaller/larger element → monotonic stack
   - Order statistics (k-th smallest) → balanced BST or sorted list
   - Substring search → trie or KMP
   - Frequency counting → hash map
2. For each identified operation, propose the appropriate data structure
3. Show the improved complexity

Suggest the best data structure substitution and explain the complexity improvement.
""",
    },

    "greedy_exchange_argument": {
        "description": "Justify a greedy choice via exchange argument.",
        "prompt": """\
You are an expert algorithm designer applying the EXCHANGE ARGUMENT (greedy justification) skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the exchange argument skill:
1. Propose a natural greedy rule (e.g., always pick the smallest, earliest deadline, etc.)
2. Attempt the exchange argument:
   - Suppose an optimal solution O does NOT follow the greedy rule at some step
   - Show we can swap elements to make O follow the greedy rule without increasing cost
   - Conclude the greedy solution is optimal
3. If the exchange argument FAILS, identify a counterexample and reject the greedy approach
4. Propose the correct algorithm (DP or other) if greedy fails

Be rigorous about the exchange proof.
""",
    },

    "counterexample_construction": {
        "description": "Generate edge cases and adversarial inputs to test the current approach.",
        "prompt": """\
You are an expert algorithm designer applying the COUNTEREXAMPLE CONSTRUCTION skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the counterexample construction skill:
1. Generate 5-8 tricky test cases including:
   - Empty/minimum input (n=0, n=1)
   - All equal elements
   - Already sorted (ascending and descending)
   - All maximum values
   - Duplicate values
   - Negative values (if applicable)
   - Disconnected components (for graph problems)
   - Wraparound or boundary conditions
2. For EACH test case, trace through the current plan and check if it gives the right answer
3. If a test case breaks the plan, identify the bug and propose a fix

List the counterexamples and their expected outputs.
""",
    },

    "complexity_refinement": {
        "description": "Improve a correct but slow algorithm by identifying and fixing the bottleneck.",
        "prompt": """\
You are an expert algorithm designer applying the COMPLEXITY REFINEMENT skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the complexity refinement skill:
1. Identify the most expensive operation in the current algorithm
2. Apply one or more of these optimizations:
   - Precomputation (prefix sums, suffix arrays)
   - Memoization (cache subproblems)
   - Monotonic queue optimization (DP with sliding window)
   - Divide and conquer optimization
   - Convex hull trick
   - Better sorting or indexing
3. Show the new complexity after the optimization
4. Verify correctness is maintained

Target: reduce from O(n^2) to O(n log n), or O(n^3) to O(n^2), etc.
""",
    },

    "analogy_mapping": {
        "description": "Map the problem to a known algorithmic template.",
        "prompt": """\
You are an expert algorithm designer applying the ANALOGY MAPPING skill.

Problem:
{problem_desc}
Constraints: {constraints}

Current plan:
{current_plan}

Apply the analogy mapping skill:
1. Search your knowledge for known algorithmic templates this problem resembles:
   - Shortest path (Dijkstra, BFS, Bellman-Ford)
   - Knapsack (0/1, unbounded, bounded)
   - Interval scheduling/covering
   - Topological sort / dependency resolution
   - Max-flow / min-cut
   - Matching (bipartite, general)
   - Sliding window / two pointers
   - Divide and conquer (merge sort style)
2. For the best matching template, show explicitly how problem variables map to template variables
3. Write the adapted algorithm

Be explicit about the mapping (e.g., "this problem's 'capacity' maps to knapsack's 'weight'").
""",
    },

    "code_generation": {
        "description": "Generate the final clean Python implementation.",
        "prompt": """\
You are an expert competitive programmer. Based on the algorithm plan below, write a clean Python implementation.

Problem:
{problem_desc}
Constraints: {constraints}
Function signature: {function_signature}

Algorithm plan:
{current_plan}

Requirements:
1. Write ONLY the function (no extra classes unless needed)
2. The function must be named exactly 'solution'
3. Handle all edge cases
4. The code must be correct and efficient
5. Add brief inline comments for clarity

Output the complete Python function in a ```python code block.
""",
    },
}

# Skill ordering heuristics: which skills to try in sequence
SKILL_SEQUENCES = {
    "dp": ["problem_abstraction", "constraint_reading", "brute_force", "state_design", "counterexample_construction", "code_generation"],
    "graph": ["problem_abstraction", "constraint_reading", "analogy_mapping", "counterexample_construction", "code_generation"],
    "greedy": ["problem_abstraction", "constraint_reading", "greedy_exchange_argument", "counterexample_construction", "code_generation"],
    "binary_search": ["problem_abstraction", "constraint_reading", "monotonicity_detection", "code_generation"],
    "data_structure": ["problem_abstraction", "constraint_reading", "data_structure_substitution", "code_generation"],
    "general": ["problem_abstraction", "constraint_reading", "brute_force", "analogy_mapping", "complexity_refinement", "counterexample_construction", "code_generation"],
}

ALL_SKILLS = list(SKILL_PROMPTS.keys())
