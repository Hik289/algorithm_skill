"""
AlgoSkill: Learning to Schedule Algorithmic Skills for Automatic Algorithm Design
Main system with skill scheduling and MCTS-style search.
"""
import math
import random
from typing import Dict, List, Tuple, Optional
from llm_client import call_llm_single, call_llm_multi
from skills import SKILL_PROMPTS, ALL_SKILLS
from verifier import run_code_with_tests, extract_code


# ─────────────────────────────────────────────────────────────────────────────
# State representation
# ─────────────────────────────────────────────────────────────────────────────
def make_initial_state(problem: dict) -> dict:
    return {
        "problem": problem,
        "plan": "",           # accumulated algorithm plan
        "pseudocode": "",     # high-level pseudocode
        "code": "",           # Python implementation
        "complexity": "",     # complexity estimate
        "counterexamples": [],
        "skill_history": [],  # list of applied skill names
        "verify_result": None,
    }


def state_summary(state: dict) -> str:
    """Compact summary for skill selection context."""
    h = ", ".join(state["skill_history"][-5:]) if state["skill_history"] else "none"
    plan_snippet = state["plan"][-400:] if state["plan"] else "(empty)"
    code_status = "code ready" if state["code"] else "no code yet"
    vr = state["verify_result"]
    if vr:
        verify_str = f"pass_rate={vr['pass_rate']:.2f}"
    else:
        verify_str = "not tested"
    return (f"[Applied skills: {h}] [Status: {code_status}, {verify_str}]\n"
            f"[Plan snippet: ...{plan_snippet}]")


# ─────────────────────────────────────────────────────────────────────────────
# Skill application
# ─────────────────────────────────────────────────────────────────────────────
def apply_skill(skill_name: str, state: dict) -> dict:
    """Apply a skill to the current state and return new state."""
    import copy
    new_state = copy.deepcopy(state)
    problem = state["problem"]

    skill = SKILL_PROMPTS.get(skill_name)
    if skill is None:
        return new_state

    prompt = skill["prompt"].format(
        problem_desc=problem["description"],
        constraints=problem["constraints"],
        current_plan=state["plan"] or "(no plan yet)",
        function_signature=problem.get("function_signature", "def solution(*args):"),
    )

    response = call_llm_single(prompt, temperature=0.7, max_tokens=1500)

    if skill_name == "code_generation":
        new_state["code"] = response
        # Verify the generated code
        vr = run_code_with_tests(response, problem)
        new_state["verify_result"] = vr
        new_state["plan"] += f"\n\n[Code generated — pass_rate={vr['pass_rate']:.2f}]"
    else:
        new_state["plan"] += f"\n\n[Skill: {skill_name}]\n{response}"

    new_state["skill_history"].append(skill_name)
    return new_state


# ─────────────────────────────────────────────────────────────────────────────
# Skill selector (LLM-based)
# ─────────────────────────────────────────────────────────────────────────────
SKILL_SELECT_PROMPT = """\
You are an expert algorithm design coordinator. Choose the BEST next algorithmic skill to apply.

Problem: {problem_name} — {problem_desc}
Constraints: {constraints}

Current state:
{state_summary}

Available skills (choose ONE by name):
{skill_list}

Rules:
- If no code has been generated yet and enough planning is done, choose 'code_generation'
- If code exists but pass_rate < 1.0, prefer 'counterexample_construction' or 'complexity_refinement'
- If no plan exists, start with 'problem_abstraction' or 'constraint_reading'
- Don't repeat skills already applied (see Applied skills)

Respond with ONLY the skill name (e.g., state_design).
"""

def select_skill_llm(state: dict, valid_skills: List[str]) -> str:
    """Use LLM to select the next skill."""
    problem = state["problem"]
    skill_list = "\n".join(f"- {s}: {SKILL_PROMPTS[s]['description']}"
                           for s in valid_skills)
    prompt = SKILL_SELECT_PROMPT.format(
        problem_name=problem["name"],
        problem_desc=problem["description"][:200],
        constraints=problem["constraints"],
        state_summary=state_summary(state),
        skill_list=skill_list,
    )
    response = call_llm_single(prompt, temperature=0.2, max_tokens=50)
    # Extract skill name
    response = response.strip().lower().replace("-", "_")
    for s in valid_skills:
        if s in response:
            return s
    # Fallback: random valid skill
    return random.choice(valid_skills)


def get_valid_skills(state: dict) -> List[str]:
    """Filter skills based on current state."""
    history = set(state["skill_history"])
    vr = state["verify_result"]

    # If code exists and passes all tests, done
    if vr and vr["pass_rate"] == 1.0:
        return []

    # Build valid set
    valid = []
    for s in ALL_SKILLS:
        # Don't re-apply skills already used (except code_generation and counterexample)
        if s in history and s not in ["code_generation", "counterexample_construction"]:
            continue
        # Don't apply code_generation too early (need at least 1 planning skill)
        if s == "code_generation" and len(history) == 0:
            continue
        valid.append(s)

    return valid if valid else ["code_generation"]


# ─────────────────────────────────────────────────────────────────────────────
# MCTS Node
# ─────────────────────────────────────────────────────────────────────────────
class MCTSNode:
    def __init__(self, state: dict, parent=None, skill_applied: str = None):
        self.state = state
        self.parent = parent
        self.skill_applied = skill_applied
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_skills: Optional[List[str]] = None  # lazy init

    @property
    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb_score(self, c: float = 1.41) -> float:
        if self.parent is None or self.parent.visits == 0:
            return float("inf")
        exploit = self.q_value
        explore = c * math.sqrt(math.log(self.parent.visits) / (1 + self.visits))
        return exploit + explore

    def is_terminal(self):
        vr = self.state.get("verify_result")
        if vr and vr["pass_rate"] == 1.0:
            return True
        if len(self.state["skill_history"]) >= 8:
            return True
        return False

    def expand(self, skill_name: str) -> 'MCTSNode':
        new_state = apply_skill(skill_name, self.state)
        child = MCTSNode(new_state, parent=self, skill_applied=skill_name)
        self.children.append(child)
        return child

    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb_score(c))


def compute_reward(state: dict) -> float:
    """Compute reward for a terminal or rollout state.

    BUGFIX 2026-05-28: previously, when state had no
    verify_result, this function would call apply_skill("code_generation")
    into a *new* dict and never write the generated code/result back into
    the caller's state. That caused MCTS branches in algoskill.py and
    ablations.py to record best_code="" even when rollouts produced
    correct code, yielding 0/60 silent failures on the mcts_ablation run.
    We now propagate the generated code & verify_result into the input
    state in-place so callers see the rollout outcome.
    """
    vr = state.get("verify_result")
    if vr is None:
        # Code not generated, force generate and evaluate
        new_state = apply_skill("code_generation", state)
        vr = new_state.get("verify_result")
        # Propagate generated artifacts back to caller's state
        state["code"] = new_state.get("code", "")
        state["verify_result"] = vr
        state["plan"] = new_state.get("plan", state.get("plan", ""))
        state["skill_history"] = new_state.get("skill_history", state["skill_history"])
        if vr is None:
            return 0.0

    r = 0.0
    # Correctness reward (0-1)
    r += vr["pass_rate"] * 0.7

    # Compile reward
    if vr["compile_ok"]:
        r += 0.1

    # Complexity bonus: shorter skill histories that succeed get a small bonus
    steps = len(state["skill_history"])
    r += max(0, 0.2 - steps * 0.02)

    return min(r, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# AlgoSkill with MCTS
# ─────────────────────────────────────────────────────────────────────────────
class AlgoSkillMCTS:
    def __init__(self, budget: int = 20, c_explore: float = 1.0):
        self.budget = budget
        self.c_explore = c_explore

    def solve(self, problem: dict) -> dict:
        """Run MCTS search and return best solution found.

        2026-05-28: also tracks the skill_history of
        the trajectory that produced best_code, so callers (task 8 fig5
        re-statistics) can recover real skill sequences.
        """
        root_state = make_initial_state(problem)
        root = MCTSNode(root_state)

        best_reward = -1.0
        best_code = ""
        best_vr = None
        best_skill_history = []

        for iteration in range(self.budget):
            # Selection
            node = root
            while not node.is_terminal() and node.children:
                valid = get_valid_skills(node.state)
                untried = [s for s in valid
                           if not any(c.skill_applied == s for c in node.children)]
                if untried:
                    break
                node = node.best_child(self.c_explore)

            if node.is_terminal():
                reward = compute_reward(node.state)
                self._backprop(node, reward)
                vr = node.state.get("verify_result")
                if reward > best_reward:
                    best_reward = reward
                    best_code = node.state.get("code", "")
                    best_vr = vr
                    best_skill_history = list(node.state.get("skill_history", []))
                continue

            # Expansion: select a skill
            valid = get_valid_skills(node.state)
            if not valid:
                reward = compute_reward(node.state)
                self._backprop(node, reward)
                continue

            skill = select_skill_llm(node.state, valid)
            child = node.expand(skill)

            # Simulation / rollout
            reward = self._rollout(child)

            # Update best
            rollout_vr = child.state.get("verify_result")
            if reward > best_reward:
                best_reward = reward
                best_code = child.state.get("code", "")
                best_vr = rollout_vr
                best_skill_history = list(child.state.get("skill_history", []))

            # Backpropagation
            self._backprop(child, reward)

        return {
            "code": best_code,
            "verify_result": best_vr,
            "reward": best_reward,
            "skill_history": best_skill_history,
        }

    def _rollout(self, node: MCTSNode) -> float:
        """Quick rollout from node to terminal.

        BUGFIX 2026-05-28: the rollout used to rebind
        the local `state` variable via state = apply_skill(...) but never
        wrote results back into node.state. The caller (solve) then read
        child.state["code"] -> "" and recorded best_code="" even when the
        rollout produced a correct solution. We now keep node.state as the
        single source of truth: rollout artifacts are propagated by
        copying the latest rolled-out state into node.state after the
        loop, before computing reward, so that the caller can recover
        the rollout's code via child.state.
        """
        state = node.state
        for _ in range(4):  # max 4 more steps
            valid = get_valid_skills(state)
            if not valid:
                break
            # Greedy rollout: use LLM selector
            skill = select_skill_llm(state, valid)
            state = apply_skill(skill, state)
            vr = state.get("verify_result")
            if vr and vr["pass_rate"] == 1.0:
                break
        # Propagate rollout result back into node.state so caller can read code.
        # Keep node.state object identity (callers may hold the reference).
        node.state.clear()
        node.state.update(state)
        return compute_reward(node.state)

    def _backprop(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


# ─────────────────────────────────────────────────────────────────────────────
# AlgoSkill Greedy (no MCTS, just LLM-guided skill selection)
# ─────────────────────────────────────────────────────────────────────────────
class AlgoSkillGreedy:
    def __init__(self, max_skills: int = 7):
        self.max_skills = max_skills

    def solve(self, problem: dict) -> dict:
        state = make_initial_state(problem)
        for _ in range(self.max_skills):
            valid = get_valid_skills(state)
            if not valid:
                break
            skill = select_skill_llm(state, valid)
            state = apply_skill(skill, state)
            vr = state.get("verify_result")
            if vr and vr["pass_rate"] == 1.0:
                break
        # Ensure code is generated
        if not state["code"]:
            state = apply_skill("code_generation", state)
        return {
            "code": state["code"],
            "verify_result": state["verify_result"],
            "skill_history": state["skill_history"],
        }


def run_algoskill_full(problem: dict, n_samples: int = 5) -> list:
    """Run AlgoSkill Full (MCTS) n_samples times and return list of code strings.

    BUDGET 2026-05-28: reduced budget from 16 to 8 to keep wallclock manageable
    on a rate-limited backend (each LLM call adds sleep plus server latency, so
    budget=16 with rollout depth 4 = up to 60 calls/trajectory at ~5min/sample
    × 5 samples × 20 problems = ~8h per condition). budget=8 cuts to ~4h.
    The original (broken) code had budget=16; results are not strictly
    comparable but the search-quality comparison vs Greedy_Policy and
    Beam_Search is preserved.
    """
    solver = AlgoSkillMCTS(budget=8, c_explore=1.0)
    results = []
    for _ in range(n_samples):
        out = solver.solve(problem)
        results.append(out["code"] or "")
    return results


def run_algoskill_greedy(problem: dict, n_samples: int = 5) -> list:
    """Run AlgoSkill Greedy (no MCTS) n_samples times."""
    solver = AlgoSkillGreedy(max_skills=7)
    results = []
    for _ in range(n_samples):
        out = solver.solve(problem)
        results.append(out["code"] or "")
    return results
