"""T-opt + S-opt judge for Hard Bench. No reference_code needed.
Judge identifies the OPTIMAL complexity from problem name+description, then
identifies the SUBMITTED code's complexity, then compares.
"""
import argparse, json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import call_llm_with_usage


JUDGE_PROMPT = """You are an algorithmic complexity expert evaluating a
Hard Bench problem solution.

Problem name: {name}
Problem description: {description}
Constraints: {constraints}

SUBMITTED SOLUTION:
```python
{sub_code}
```

Step 1: Identify the OPTIMAL asymptotic worst-case time and space complexity
for this problem (in big-O notation, using N for primary input size, Q for
queries, K for window/top-K param, M for grid columns, etc.).

Step 2: Identify the SUBMITTED code's asymptotic worst-case time and space
complexity.

Now state your final answer in EXACTLY these 4 lines, each starting with
the literal label. Do NOT include any markdown or extra text:

OPT_TIME: O(...)
OPT_SPACE: O(...)
SUB_TIME: O(...)
SUB_SPACE: O(...)
"""


def parse_judge_response(text):
    def grab(pat, t):
        m = re.search(pat, t, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    opt_t = grab(r"OPT[_ ]TIME[:\s]*\**\s*(O\([^\n)]*\))", text)
    opt_s = grab(r"OPT[_ ]SPACE[:\s]*\**\s*(O\([^\n)]*\))", text)
    sub_t = grab(r"SUB[_ ]TIME[:\s]*\**\s*(O\([^\n)]*\))", text)
    sub_s = grab(r"SUB[_ ]SPACE[:\s]*\**\s*(O\([^\n)]*\))", text)
    return opt_t, opt_s, sub_t, sub_s


def normalize_complexity(c):
    if not c: return ""
    s = c.lower().replace(" ", "").replace("o(", "").rstrip(")").strip()
    s = s.replace("²", "^2").replace("³", "^3").replace("·", "")
    s = s.replace("**", "^").replace("(", "").replace(")", "").replace("*","")
    return s


RANK = {
    "1": 0, "logn": 1, "loglogn": 1, "logq": 1,
    "n": 2, "n+q": 2, "n+m": 2, "qlogn": 2, "q": 2,
    "nlogn": 3, "nlogq": 3, "(n+q)logn": 3, "nlogm": 3,
    "n^2": 4, "nm": 4, "nq": 4, "nlog^2n": 4,
    "n^2logn": 5, "n^3": 5, "n^2m": 5,
    "n^4": 6, "n^3logn": 6,
    "2^n": 7, "2^nn": 8, "2^nn^2": 9,
    "n!": 10, "n^n": 11,
}


def rank_complexity(c):
    n = normalize_complexity(c)
    if not n: return None
    if n in RANK: return RANK[n]
    if "2^n" in n: return 7 + (1 if "n^2" in n else 0)
    if "n^4" in n: return 6
    if "n^3" in n: return 5
    if "n^2logn" in n: return 5
    if "n^2" in n or "nm" in n or "nq" in n: return 4
    if "nlogn" in n: return 3
    if "logn" in n and len(n)<=6: return 1
    if n in ("n","q","n+q","n+m"): return 2
    return None


def is_topt(opt, sub):
    on = normalize_complexity(opt); sn = normalize_complexity(sub)
    if not on or not sn: return None
    if on == sn: return True
    ro = rank_complexity(opt); rs = rank_complexity(sub)
    if ro is None or rs is None: return None
    return rs <= ro


def judge_one(name, description, constraints, sub_code, backbone):
    prompt = JUDGE_PROMPT.format(
        name=name[:200], description=description[:1500],
        constraints=constraints[:200], sub_code=sub_code[:2500])
    out = call_llm_with_usage(prompt, backbone=backbone,
                              temperature=0.0, max_tokens=800)
    text = out["text"]
    opt_t, opt_s, sub_t, sub_s = parse_judge_response(text)
    return {
        "judge_raw": text[:3500],
        "judge_opt_time": opt_t, "judge_opt_space": opt_s,
        "judge_sub_time": sub_t, "judge_sub_space": sub_s,
        "topt": is_topt(opt_t, sub_t), "sopt": is_topt(opt_s, sub_s),
        "judge_tokens": out["tokens"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--judge_backbone", default="judge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    corpus = json.load(open(args.corpus))
    if isinstance(corpus, list): corpus = {p["id"]:p for p in corpus}
    results = json.load(open(args.results))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    print(f"[topt-hb] backbone={args.judge_backbone} results={len(results)} resume={len(out)}", flush=True)
    n_proc = 0
    for pid, row in results.items():
        if pid in out: continue
        if not row.get("correct"):
            out[pid] = {**row, "topt": None, "sopt": None,
                        "judge_skipped_reason": "incorrect"}
            with open(args.out,"w") as f: json.dump(out, f, indent=2)
            continue
        sub_code = row.get("code","")
        if not sub_code:
            out[pid] = {**row, "topt": None, "sopt": None,
                        "judge_skipped_reason": "empty_code"}
            with open(args.out,"w") as f: json.dump(out, f, indent=2)
            continue
        src = corpus.get(pid, {})
        try:
            j = judge_one(src.get("name",""), src.get("description",""),
                          src.get("constraints",""), sub_code,
                          args.judge_backbone)
        except Exception as e:
            j = {"judge_error": str(e)[:300], "topt": None, "sopt": None}
        out[pid] = {**row, **j}
        with open(args.out,"w") as f: json.dump(out, f, indent=2)
        n_proc += 1
    n_correct = sum(1 for v in out.values() if v.get("correct"))
    n_t = sum(1 for v in out.values() if v.get("topt") is True)
    n_t_inf = sum(1 for v in out.values() if v.get("topt") is not None)
    n_s = sum(1 for v in out.values() if v.get("sopt") is True)
    n_s_inf = sum(1 for v in out.values() if v.get("sopt") is not None)
    print(f"[topt-hb final] correct={n_correct}/{len(out)} "
          f"T-opt={n_t}/{n_t_inf}={n_t/n_t_inf*100 if n_t_inf else 0:.1f}% "
          f"S-opt={n_s}/{n_s_inf}={n_s/n_s_inf*100 if n_s_inf else 0:.1f}%", flush=True)


if __name__ == "__main__": main()
