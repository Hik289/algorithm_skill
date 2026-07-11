"""
T-opt / S-opt LLM-as-judge for rule-based corpus v3.

Reads a results JSON (from run_rule_based.py) and for each correctly-solved
problem, asks the judge LLM to identify the asymptotic time and space
complexity of the submitted code, then compares against the family's known
opt_time / opt_space.

Output: same results JSON augmented with per-row "topt", "sopt", "judge_time",
"judge_space" fields.

Methodology matches paper §5.9 / tab:rule_bench: "an LLM judge reads the
generated code, infers its time and space complexities, and compares them
with the known optimal complexities".
"""
import argparse, json, os, sys, re, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import call_llm_with_usage, BACKBONE_CONFIGS


JUDGE_PROMPT = """You are an algorithmic complexity expert. Read the following
Python code and decide its asymptotic worst-case TIME and SPACE complexity in
terms of the problem's input variables.

Problem family: {family}
Stated optimal time complexity: {opt_time}
Stated optimal space complexity: {opt_space}
Variable conventions: N = primary input size; M = secondary if grid/graph;
K = window or top-K parameter if present; S = sum of values; Q = number of
queries.

Code:
```python
{code}
```

Respond with EXACTLY this format (no other text):
TIME: O(...)
SPACE: O(...)
TIME_MATCHES_OPTIMAL: yes|no
SPACE_MATCHES_OPTIMAL: yes|no
EXPLANATION: <one-sentence reason>
"""


def _normalize(s):
    return s.lower().replace(" ", "").replace("*", "")


def judge_one(code, family, opt_time, opt_space, backbone):
    prompt = JUDGE_PROMPT.format(
        code=code[:2500], family=family,
        opt_time=opt_time, opt_space=opt_space)
    out = call_llm_with_usage(prompt, backbone=backbone,
                              temperature=0.0, max_tokens=400)
    text = out["text"]

    def grab(pat):
        m = re.search(pat, text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    time_complexity = grab(r"TIME:\s*(O\([^\n]+?\))")
    space_complexity = grab(r"SPACE:\s*(O\([^\n]+?\))")
    time_matches = grab(r"TIME_MATCHES_OPTIMAL:\s*(yes|no)")
    space_matches = grab(r"SPACE_MATCHES_OPTIMAL:\s*(yes|no)")
    expl = grab(r"EXPLANATION:\s*([^\n]+)")

    return {
        "judge_time": time_complexity,
        "judge_space": space_complexity,
        "topt": time_matches.lower() == "yes",
        "sopt": space_matches.lower() == "yes",
        "judge_explanation": expl[:200],
        "judge_tokens": out["tokens"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="Results JSON from run_rule_based.py")
    ap.add_argument("--corpus", required=True,
                    help="rule_based_corpus_v3.json (has opt_time per problem)")
    ap.add_argument("--judge_backbone", default="judge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = {p["id"]: p for p in json.load(open(args.corpus))}
    results = json.load(open(args.results))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}

    print(f"[topt-judge] backbone={args.judge_backbone} "
          f"results={len(results)} resume={len(out)}", flush=True)

    n_processed = 0
    for pid, row in results.items():
        if pid in out:
            continue
        # Only judge correctly-solved problems (paper's methodology)
        if not row.get("correct"):
            out[pid] = {**row, "topt": None, "sopt": None,
                        "judge_skipped_reason": "incorrect"}
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            continue
        code = row.get("code", "")
        if not code:
            out[pid] = {**row, "topt": None, "sopt": None,
                        "judge_skipped_reason": "empty_code"}
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            continue
        source = corpus.get(pid, {})
        family = source.get("family", row.get("family", "unknown"))
        opt_time = source.get("opt_time", row.get("opt_time", "?"))
        opt_space = source.get("opt_space", row.get("opt_space", "?"))
        try:
            j = judge_one(code, family, opt_time, opt_space,
                          args.judge_backbone)
        except Exception as e:
            j = {"judge_error": str(e)[:200],
                 "topt": None, "sopt": None}
        out[pid] = {**row, **j, "family": family,
                    "opt_time": opt_time, "opt_space": opt_space}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        n_processed += 1
        if n_processed % 10 == 0 or n_processed <= 5:
            n_topt = sum(1 for v in out.values() if v.get("topt"))
            n_correct = sum(1 for v in out.values() if v.get("correct"))
            print(f"  [{n_processed} judged] T-opt {n_topt}/{n_correct} "
                  f"= {n_topt/n_correct*100 if n_correct else 0:.1f}%", flush=True)

    # Summary
    n_correct = sum(1 for v in out.values() if v.get("correct"))
    n_topt = sum(1 for v in out.values() if v.get("topt"))
    n_sopt = sum(1 for v in out.values() if v.get("sopt"))
    print(f"\n[topt-judge final] correct={n_correct}/{len(out)} "
          f"T-opt={n_topt}/{n_correct}={n_topt/n_correct*100 if n_correct else 0:.1f}% "
          f"S-opt={n_sopt}/{n_correct}={n_sopt/n_correct*100 if n_correct else 0:.1f}%",
          flush=True)


if __name__ == "__main__":
    main()
