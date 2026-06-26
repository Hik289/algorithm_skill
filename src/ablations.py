"""
Ablation variants of AlgoSkill for systematic evaluation.
"""
from algoskill import AlgoSkillGreedy, AlgoSkillMCTS, apply_skill, make_initial_state, get_valid_skills, select_skill_llm
from skills import SKILL_PROMPTS
from verifier import run_code_with_tests


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: Remove specific skill groups
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_GROUPS = {
    "wo_abstraction": ["problem_abstraction"],
    "wo_constraint": ["constraint_reading"],
    "wo_brute_force": ["brute_force"],
    "wo_monotonicity": ["monotonicity_detection"],
    "wo_state_design": ["state_design"],
    "wo_data_structure": ["data_structure_substitution"],
    "wo_counterexample": ["counterexample_construction"],
    "wo_complexity_refinement": ["complexity_refinement"],
}


class AlgoSkillAblated:
    """AlgoSkill with one skill group removed."""
    def __init__(self, removed_skills: list, max_skills: int = 7):
        self.removed_skills = set(removed_skills)
        self.max_skills = max_skills

    def solve(self, problem: dict) -> dict:
        state = make_initial_state(problem)
        for _ in range(self.max_skills):
            valid = [s for s in get_valid_skills(state)
                     if s not in self.removed_skills]
            if not valid:
                valid = ["code_generation"]
            skill = select_skill_llm(state, valid)
            state = apply_skill(skill, state)
            vr = state.get("verify_result")
            if vr and vr["pass_rate"] == 1.0:
                break
        if not state["code"]:
            state = apply_skill("code_generation", state)
        return {
            "code": state["code"],
            "verify_result": state["verify_result"],
            "skill_history": state["skill_history"],
        }


def run_ablation(problem: dict, ablation_name: str, n_samples: int = 5) -> list:
    removed = ABLATION_GROUPS.get(ablation_name, [])
    solver = AlgoSkillAblated(removed_skills=removed, max_skills=7)
    results = []
    for _ in range(n_samples):
        out = solver.solve(problem)
        results.append(out["code"] or "")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: MCTS variants
# ─────────────────────────────────────────────────────────────────────────────
class BeamSearchSkills:
    """Beam search over skill sequences (no UCB, just greedy beam)."""
    def __init__(self, beam_width: int = 3, max_depth: int = 5):
        self.beam_width = beam_width
        self.max_depth = max_depth

    def solve(self, problem: dict) -> dict:
        from algoskill import compute_reward
        beams = [make_initial_state(problem)]

        for depth in range(self.max_depth):
            new_beams = []
            for state in beams:
                valid = get_valid_skills(state)
                if not valid:
                    new_beams.append(state)
                    continue
                # Try up to beam_width skills
                import random
                skills_to_try = valid[:self.beam_width]
                for sk in skills_to_try:
                    new_state = apply_skill(sk, state)
                    new_beams.append(new_state)

            # Score and prune
            scored = [(compute_reward(s), s) for s in new_beams]
            scored.sort(key=lambda x: -x[0])
            beams = [s for _, s in scored[:self.beam_width]]

        best = beams[0]
        if not best.get("code"):
            best = apply_skill("code_generation", best)
        return {
            "code": best.get("code", ""),
            "verify_result": best.get("verify_result"),
        }


class MCTSNoPolicy:
    """MCTS with random skill selection (no learned policy prior)."""
    def __init__(self, budget: int = 16):
        self.budget = budget

    def solve(self, problem: dict) -> dict:
        # BUGFIX 2026-05-28 (ml_engineer_claude): relied on compute_reward's
        # in-place propagation fix in algoskill.py to make
        # child.state.get("code") return non-empty after compute_reward.
        # Also now captures best_vr (was hardcoded None previously).
        import random
        from algoskill import MCTSNode, compute_reward
        root = MCTSNode(make_initial_state(problem))
        best_reward = -1.0
        best_code = ""
        best_vr = None

        for _ in range(self.budget):
            node = root
            while not node.is_terminal() and node.children:
                valid = get_valid_skills(node.state)
                untried = [s for s in valid
                           if not any(c.skill_applied == s for c in node.children)]
                if untried:
                    break
                # Random selection (no UCB policy prior)
                node = random.choice(node.children)

            if node.is_terminal():
                reward = compute_reward(node.state)
            else:
                valid = get_valid_skills(node.state)
                if not valid:
                    reward = compute_reward(node.state)
                else:
                    skill = random.choice(valid)
                    child = node.expand(skill)
                    reward = compute_reward(child.state)
                    if reward > best_reward:
                        best_reward = reward
                        best_code = child.state.get("code", "")
                        best_vr = child.state.get("verify_result")
                    # Backprop
                    n = child
                    while n:
                        n.visits += 1
                        n.total_reward += reward
                        n = n.parent
                    continue

            if reward > best_reward:
                best_reward = reward
                best_code = node.state.get("code", "")
                best_vr = node.state.get("verify_result")

            n = node
            while n:
                n.visits += 1
                n.total_reward += reward
                n = n.parent

        return {"code": best_code, "verify_result": best_vr}


def run_beam_search(problem: dict, n_samples: int = 5) -> list:
    solver = BeamSearchSkills(beam_width=3, max_depth=5)
    results = []
    for _ in range(n_samples):
        out = solver.solve(problem)
        results.append(out["code"] or "")
    return results


def run_mcts_no_policy(problem: dict, n_samples: int = 5) -> list:
    # BUDGET 2026-05-28: reduced from 16 to 8 (see comment in algoskill.py
    # run_algoskill_full). Symmetric reduction so the comparison vs
    # MCTS_WithPolicy is fair.
    solver = MCTSNoPolicy(budget=8)
    results = []
    for _ in range(n_samples):
        out = solver.solve(problem)
        results.append(out["code"] or "")
    return results
