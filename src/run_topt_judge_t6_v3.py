"""
T-opt judge v3 for Task 6 cross-platform corpus.

Fixes from v2:
- max_tokens = 800 (v2 was 300 — judge often ran out mid-response)
- prompt has a STRICT FOOTER reinforcement that forces the 4-line format
- store raw judge response in `judge_raw` field (no more silent parse fail)
- lenient parser: also tolerate markdown bullets, "Overall O(...)", and "REF/SUB"
  prefixes without colons, prefer the LAST O(...) in REFERENCE block if no
  strict line
- sanity_check function to spot-check parsed results post-run
"""
import argparse, json, os, sys, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import call_llm_with_usage


JUDGE_PROMPT = """You are an algorithmic complexity expert. Analyze the two
Python programs solving the SAME problem.

Problem context (theme): {theme}
Tags: {tags}

REFERENCE (intended solution):
```python
{ref_code}
```

SUBMISSION:
```python
{sub_code}
```

Think briefly about each program's asymptotic worst-case TIME and SPACE
complexity in big-O notation (use N for primary input, Q for queries, K for
windows/top-K, M for grid/secondary, S for sum of values, B for bit-width).

Now state your final answer in EXACTLY these 4 lines, each starting with
the literal label and nothing else after the value. Do NOT include any
markdown or extra text:

REF_TIME: O(...)
REF_SPACE: O(...)
SUB_TIME: O(...)
SUB_SPACE: O(...)
"""


def parse_judge_response(text):
    """Lenient parser. Tries multiple strategies before giving up."""
    def grab(pat, t):
        m = re.search(pat, t, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    # Strategy 1: strict labels with colon
    ref_t = grab(r"REF[_ ]TIME[:\s]*\**\s*(O\([^\n)]*\))", text)
    ref_s = grab(r"REF[_ ]SPACE[:\s]*\**\s*(O\([^\n)]*\))", text)
    sub_t = grab(r"SUB[_ ]TIME[:\s]*\**\s*(O\([^\n)]*\))", text)
    sub_s = grab(r"SUB[_ ]SPACE[:\s]*\**\s*(O\([^\n)]*\))", text)

    # Strategy 2: REFERENCE/SUBMISSION blocks with "Overall" or "Total" O(...)
    if not ref_t:
        ref_block = ""
        m = re.search(r"\*\*?REF(?:ERENCE)?[^*]*\*\*?(.+?)(?=\*\*?SUB|\Z)",
                      text, re.DOTALL | re.IGNORECASE)
        if m: ref_block = m.group(1)
        if ref_block:
            for pat in [
                r"(?:Overall|Total)[^\n]*?(O\([^\n)]*\))",
                r"Time[^\n]*?(O\([^\n)]*\))",
            ]:
                v = grab(pat, ref_block)
                if v: ref_t = v; break
            if not ref_t:
                # Last O() in the block
                matches = re.findall(r"O\([^\n)]*\)", ref_block)
                if matches: ref_t = matches[-1]
    if not sub_t:
        sub_block = ""
        m = re.search(r"\*\*?SUB(?:MISSION)?[^*]*\*\*?(.+)", text,
                      re.DOTALL | re.IGNORECASE)
        if m: sub_block = m.group(1)
        if sub_block:
            for pat in [
                r"(?:Overall|Total)[^\n]*?(O\([^\n)]*\))",
                r"Time[^\n]*?(O\([^\n)]*\))",
            ]:
                v = grab(pat, sub_block)
                if v: sub_t = v; break
            if not sub_t:
                matches = re.findall(r"O\([^\n)]*\)", sub_block)
                if matches: sub_t = matches[-1]

    return ref_t, ref_s, sub_t, sub_s


def normalize_complexity(c):
    if not c: return ""
    s = c.lower().replace(" ", "")
    s = s.replace("o(", "").rstrip(")").strip()
    s = s.replace("²", "^2").replace("³", "^3").replace("·", "")
    s = s.replace("**", "^").replace("(", "").replace(")", "")
    s = s.replace("*", "").replace("·", "")
    s = s.replace("×", "")
    return s


RANK_TABLE = [
    ("1", 0), ("logn", 1), ("loglogn", 1), ("logq", 1),
    ("n", 2), ("n+q", 2), ("n+m", 2), ("qlogn", 2), ("q", 2),
    ("nlogn", 3), ("nlogq", 3), ("n+qlogn", 3), ("nlogm", 3),
    ("qlog^2n", 3), ("(n+q)logn", 3),
    ("n^2", 4), ("nm", 4), ("nq", 4), ("n*sqrtn", 4), ("nlog^2n", 4),
    ("n^2logn", 5), ("n^3", 5), ("n^2m", 5),
    ("n^4", 6), ("n^3logn", 6),
    ("2^n", 7), ("2^nn", 8), ("2^nn^2", 9),
    ("n!", 10), ("n^n", 11),
]
RANK = dict(RANK_TABLE)


def rank_complexity(c):
    n = normalize_complexity(c)
    if not n:
        return None
    if n in RANK:
        return RANK[n]
    # Containment heuristics
    if "2^n" in n: return 7 + (1 if "n^2" in n else 0)
    if "n^4" in n: return 6
    if "n^3" in n: return 5
    if "n^2logn" in n: return 5
    if "n^2" in n or "nm" in n or "nq" in n: return 4
    if "nlog^2" in n: return 4
    if "nlogn" in n or "nlogq" in n or "qlogn" in n: return 3
    if "logn" in n and len(n) <= 6: return 1
    if n in ("n", "q", "n+q", "n+m"): return 2
    return None


def is_topt(ref, sub):
    rn = normalize_complexity(ref)
    sn = normalize_complexity(sub)
    if not rn or not sn:
        return None
    if rn == sn:
        return True
    rr = rank_complexity(ref)
    rs = rank_complexity(sub)
    if rr is None or rs is None:
        return None
    return rs <= rr


def judge_one(theme, tags, ref_code, sub_code, backbone):
    prompt = JUDGE_PROMPT.format(
        theme=theme[:200], tags=tags[:200],
        ref_code=ref_code[:2500], sub_code=sub_code[:2500],
    )
    out = call_llm_with_usage(prompt, backbone=backbone,
                              temperature=0.0, max_tokens=800)
    text = out["text"]
    ref_t, ref_s, sub_t, sub_s = parse_judge_response(text)

    t_match = is_topt(ref_t, sub_t)
    s_match = is_topt(ref_s, sub_s)

    return {
        "judge_raw": text[:3500],
        "judge_ref_time": ref_t,
        "judge_ref_space": ref_s,
        "judge_sub_time": sub_t,
        "judge_sub_space": sub_s,
        "topt": t_match,
        "sopt": s_match,
        "judge_tokens": out["tokens"],
    }


def sanity_check(out_dict, n=5):
    """Verify that recently judged records have non-None topt."""
    import random
    correct_judged = [(pid, v) for pid, v in out_dict.items()
                      if v.get("correct") and "judge_raw" in v]
    if not correct_judged:
        return False, "no correct-judged records yet"
    sample = random.sample(correct_judged, min(n, len(correct_judged)))
    non_none = sum(1 for _, v in sample if v.get("topt") is not None)
    return non_none >= max(1, n // 2), \
        f"sanity: {non_none}/{len(sample)} non-None"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--judge_backbone", default="claude_haiku")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0,
                    help="If > 0, judge only first N and exit (for testing)")
    args = ap.parse_args()

    corpus = json.load(open(args.corpus))
    if isinstance(corpus, list):
        corpus = {p["id"]: p for p in corpus}
    results = json.load(open(args.results))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}

    print(f"[topt-t6-v3] backbone={args.judge_backbone} "
          f"results={len(results)} resume={len(out)} "
          f"smoke={args.smoke}", flush=True)

    n_proc = 0
    for pid, row in results.items():
        if args.smoke and n_proc >= args.smoke:
            break
        if pid in out:
            continue
        if not row.get("correct"):
            out[pid] = {**row, "topt": None, "sopt": None,
                        "judge_skipped_reason": "incorrect"}
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            continue
        sub_code = row.get("code", "")
        if not sub_code:
            out[pid] = {**row, "topt": None, "sopt": None,
                        "judge_skipped_reason": "empty_code"}
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            continue
        source = corpus.get(pid, {})
        ref_code = source.get("reference_code", "")
        theme = source.get("theme", "")
        tags = source.get("tags", "")
        try:
            j = judge_one(theme, tags, ref_code, sub_code, args.judge_backbone)
        except Exception as e:
            j = {"judge_error": str(e)[:300], "topt": None, "sopt": None}
        out[pid] = {**row, **j, "theme": theme, "tags": tags}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        n_proc += 1
        if n_proc % 20 == 0 or n_proc <= 5:
            n_topt = sum(1 for v in out.values() if v.get("topt"))
            n_inferable = sum(1 for v in out.values()
                               if v.get("topt") is not None)
            n_correct = sum(1 for v in out.values() if v.get("correct"))
            print(f"  [{n_proc} judged] T-opt {n_topt}/{n_inferable}"
                  f" = {n_topt/n_inferable*100 if n_inferable else 0:.1f}% "
                  f"(of {n_correct} correct)", flush=True)

    n_correct = sum(1 for v in out.values() if v.get("correct"))
    n_topt = sum(1 for v in out.values() if v.get("topt"))
    n_inferable = sum(1 for v in out.values() if v.get("topt") is not None)
    n_sopt = sum(1 for v in out.values() if v.get("sopt"))
    n_sinferable = sum(1 for v in out.values() if v.get("sopt") is not None)
    ok, msg = sanity_check(out, n=5)
    print(f"\n[topt-t6-v3 final] correct={n_correct}/{len(out)} "
          f"T-opt={n_topt}/{n_inferable}"
          f"={n_topt/n_inferable*100 if n_inferable else 0:.1f}% "
          f"S-opt={n_sopt}/{n_sinferable}"
          f"={n_sopt/n_sinferable*100 if n_sinferable else 0:.1f}% "
          f"[{msg}]", flush=True)


if __name__ == "__main__":
    main()
