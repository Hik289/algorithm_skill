"""
Unified LLM client for AlgoSkill rebuild rerun (2026-05-28).

Supports:
- Anthropic Haiku via standard API (set ANTHROPIC_API_BASE env to override; default https://api.anthropic.com)
- Groq (Qwen/Llama family)  (sleep 13s/call, verify non-empty)
- OpenAI (gpt-4o-mini, gpt-4o, gpt-4.1)  (when key set)
- Gemini (when key set)

Drop-in compatible with old llm_client API:
- call_llm(prompt, model=..., temperature=, max_tokens=, n=)
- call_llm_single(prompt, ...) -> str
- call_llm_multi(prompt, n=, ...) -> list[str]

Active backbone is selected via env var ALGOSKILL_BACKBONE.
Recognized backbones:
- "claude_haiku" (default, via proxy)
- "qwen3_32b"
- "llama33_70b"
- "llama31_8b"
- "llama4_scout"
- "gpt4o_mini" (needs OPENAI_API_KEY)
- "gpt4o" / "gpt41"
- "gemini25flash" (needs GEMINI_API_KEY)
"""
import os
import time
import json
import requests
from typing import List, Dict, Optional

# ── Backbone registry ────────────────────────────────────────────────────────
BACKBONE_CONFIGS = {
    "claude_haiku": {
        "provider": "anthropic_proxy",
        "model": "claude-haiku-4-5",
        "base_url": os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com"),
        # Director said sleep ~13s; empirical 20-rapid-call test got 0 failures
        # at ~3s/call. Use 3s as conservative middle ground with adaptive
        # backoff on 429/5xx via the retry layer.
        "sleep_after": 3.0,
    },
    "qwen3_32b": {
        "provider": "groq",
        "model": "qwen/qwen3-32b",
        "sleep_after": 13.0,
    },
    "llama33_70b": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "sleep_after": 13.0,
    },
    "llama31_8b": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "sleep_after": 13.0,
    },
    "llama4_scout": {
        "provider": "groq",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "sleep_after": 13.0,
    },
    "gpt4o_mini": {"provider": "openai", "model": "gpt-4o-mini", "sleep_after": 0.0},
    "gpt4o":      {"provider": "openai", "model": "gpt-4o",      "sleep_after": 0.0},
    "gpt41":      {"provider": "openai", "model": "gpt-4.1",     "sleep_after": 0.0},
    "gemini25flash": {"provider": "gemini", "model": "gemini-2.5-flash", "sleep_after": 0.0},
    # AWS Bedrock backbones (long-lived Bedrock API key auth, Converse endpoint).
    # NB: AWS Bedrock does NOT host GPT-4o; only `openai.gpt-oss-*` open-weight.
    "bedrock_gpt_oss_120b": {
        "provider": "bedrock",
        "model": "openai.gpt-oss-120b-1:0",
        "region": "us-east-1",
        "sleep_after": 0.0,
    },
    "bedrock_gpt_oss_20b": {
        "provider": "bedrock",
        "model": "openai.gpt-oss-20b-1:0",
        "region": "us-east-1",
        "sleep_after": 0.0,
    },
    "bedrock_llama33_70b": {
        "provider": "bedrock",
        # Requires cross-region inference profile (on-demand throughput
        # not supported for raw model id).
        "model": "us.meta.llama3-3-70b-instruct-v1:0",
        "region": "us-east-1",
        "sleep_after": 0.0,
    },
    "bedrock_claude_haiku": {
        "provider": "bedrock",
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "region": "us-east-1",
        "sleep_after": 0.0,
    },
    "bedrock_claude_sonnet45": {
        "provider": "bedrock",
        # Requires cross-region inference profile for on-demand throughput.
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "region": "us-east-1",
        "sleep_after": 0.0,
    },
    # GPT-5.5 via the new Bedrock /openai/v1/responses endpoint
    # (different host + URL shape from Converse API).
    "bedrock_gpt55": {
        "provider": "bedrock_responses",
        "model": "openai.gpt-5.5",
        # 16k+ output budget is needed because the model spends most of its
        # output budget on reasoning tokens before emitting code.
        "default_max_output_tokens": 16384,
        "sleep_after": 0.0,
    },
    "bedrock_gpt54": {
        "provider": "bedrock_responses",
        "model": "openai.gpt-5.4",
        "default_max_output_tokens": 4096,
        "sleep_after": 0.0,
    },
}

DEFAULT_BACKBONE = os.environ.get("ALGOSKILL_BACKBONE", "claude_haiku")

# ── API keys ─────────────────────────────────────────────────────────────────
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_PROXY_KEY = os.environ.get("ANTHROPIC_PROXY_KEY", "placeholder")
BEDROCK_KEY = os.environ.get("BEDROCK_API_KEY", "")

# ── Backbone routing ─────────────────────────────────────────────────────────
def _ensure_keys(provider: str):
    if provider == "groq" and not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    if provider == "bedrock" and not BEDROCK_KEY:
        raise RuntimeError("BEDROCK_API_KEY not set")
    if provider == "bedrock_responses" and not BEDROCK_KEY:
        raise RuntimeError("BEDROCK_API_KEY not set (shared with bedrock_responses)")
    if provider == "openai" and not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    if provider == "gemini" and not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

# ── Anthropic proxy ──────────────────────────────────────────────────────────
def _call_anthropic_proxy(model: str, prompt: str, temperature: float,
                          max_tokens: int, base_url: str) -> Dict:
    url = f"{base_url}/v1/messages"
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_PROXY_KEY,
        "connection": "close",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Use fresh session per call (avoid stale keepalive after 502s from proxy)
    # Tighter (connect=10s, read=120s) so a stuck conn fails fast.
    with requests.Session() as sess:
        r = sess.post(url, headers=headers, json=body,
                      timeout=(10, 120))
    r.raise_for_status()
    d = r.json()
    text = ""
    for block in d.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    usage = d.get("usage", {})
    return {
        "text": text,
        "tokens": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }

# ── Groq ─────────────────────────────────────────────────────────────────────
def _call_groq(model: str, prompt: str, temperature: float, max_tokens: int) -> Dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        # Raise so caller retries; include status text for diagnosis
        r.raise_for_status()
    d = r.json()
    choices = d.get("choices", [])
    if not choices:
        raise RuntimeError(f"Groq returned no choices: {d}")
    text = choices[0].get("message", {}).get("content", "")
    if not text or not text.strip():
        raise RuntimeError(f"Groq returned empty text for {model}: {json.dumps(d)[:300]}")
    usage = d.get("usage", {})
    return {
        "text": text,
        "tokens": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }

# ── OpenAI / Gemini ──────────────────────────────────────────────────────────
def _call_openai(model: str, prompt: str, temperature: float, max_tokens: int) -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = r.choices[0].message.content or ""
    return {
        "text": text,
        "tokens": {
            "prompt_tokens": getattr(r.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(r.usage, "completion_tokens", 0),
            "total_tokens": getattr(r.usage, "total_tokens", 0),
        },
    }

def _call_bedrock(model: str, prompt: str, temperature: float,
                  max_tokens: int, region: str = "us-east-1") -> Dict:
    """Bedrock Converse API call with long-lived API key (Bearer auth).
    Handles Anthropic, Llama, OpenAI gpt-oss models uniformly.
    """
    url = (f"https://bedrock-runtime.{region}.amazonaws.com/"
           f"model/{model}/converse")
    body = {
        "messages": [{"role": "user",
                      "content": [{"text": prompt}]}],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    r = requests.post(url,
                      headers={"Authorization": f"Bearer {BEDROCK_KEY}",
                               "content-type": "application/json"},
                      json=body, timeout=120)
    r.raise_for_status()
    j = r.json()
    # gpt-oss returns content list with reasoningContent + text blocks;
    # Claude/Llama return single text block. Concat all text blocks.
    text = ""
    for blk in j.get("output", {}).get("message", {}).get("content", []):
        if "text" in blk:
            text += blk["text"]
    u = j.get("usage", {})
    return {
        "text": text,
        "tokens": {
            "prompt_tokens": u.get("inputTokens", 0),
            "completion_tokens": u.get("outputTokens", 0),
            "total_tokens": u.get("totalTokens", 0),
        },
    }

def _call_bedrock_responses(model: str, prompt: str, temperature: float,
                            max_tokens: int) -> Dict:
    """Bedrock /openai/v1/responses endpoint (different host + URL shape
    from the Converse API). Used for GPT-5.x reasoning models.

    Response format: output is a list of items with type=reasoning
    (internal CoT, usually empty summary) and type=message
    (the actual user-facing answer). usage has input_tokens, output_tokens
    (which already includes reasoning_tokens), and
    output_tokens_details.reasoning_tokens.
    """
    url = ("https://bedrock-mantle.us-east-2.api.aws/"
           "openai/v1/responses")
    body = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    r = requests.post(url,
                      headers={"Authorization": f"Bearer {BEDROCK_KEY}",
                               "content-type": "application/json"},
                      json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    text = ""
    for item in j.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
    u = j.get("usage", {})
    return {
        "text": text,
        "tokens": {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        },
    }


def _call_gemini(model: str, prompt: str, temperature: float, max_tokens: int) -> Dict:
    url = (f"https://generativelanguage.googleapis.com/v1/models/"
           f"{model}:generateContent?key={GEMINI_KEY}")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    r = requests.post(url, json=body, timeout=180)
    r.raise_for_status()
    d = r.json()
    cands = d.get("candidates", [])
    if not cands:
        raise RuntimeError(f"Gemini returned no candidates: {d}")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    u = d.get("usageMetadata", {})
    return {
        "text": text,
        "tokens": {
            "prompt_tokens": u.get("promptTokenCount", 0),
            "completion_tokens": u.get("candidatesTokenCount", 0),
            "total_tokens": u.get("totalTokenCount", 0),
        },
    }

# ── Public dispatch ──────────────────────────────────────────────────────────
# Fast-fail tracker: per-process counter of consecutive call failures per backbone.
# After CONSECUTIVE_FAIL_THRESHOLD (3) all-retries-failed calls in a row, sys.exit
# the process so it doesn't silent-burn 4-12h while every cell fails with quota
# / model-not-found / auth errors (ops bug 9 root cause).
_CONSECUTIVE_FAILS: Dict[str, int] = {}
CONSECUTIVE_FAIL_THRESHOLD = int(os.environ.get(
    "LLM_CLIENT_FAST_FAIL_THRESHOLD", "3"))


def _record_call_success(backbone: str):
    _CONSECUTIVE_FAILS[backbone] = 0


def _record_call_failure(backbone: str, err: Exception):
    n = _CONSECUTIVE_FAILS.get(backbone, 0) + 1
    _CONSECUTIVE_FAILS[backbone] = n
    if n >= CONSECUTIVE_FAIL_THRESHOLD:
        import sys
        msg = (f"[FAST-FAIL] backbone={backbone}: {n} consecutive _call "
               f"failures (threshold={CONSECUTIVE_FAIL_THRESHOLD}). "
               f"Last error: {type(err).__name__}: {str(err)[:200]}. "
               f"Killing process to avoid silent-burn.")
        print(msg, flush=True)
        sys.stderr.write(msg + "\n"); sys.stderr.flush()
        # Reset counter so a downstream "try/except RuntimeError" caller can
        # surface the message but we still exit the process. We use os._exit
        # to bypass atexit so it's fast.
        os._exit(101)


def _call(backbone: str, prompt: str, temperature: float, max_tokens: int,
          retries: int = 4) -> Dict:
    cfg = BACKBONE_CONFIGS[backbone]
    provider = cfg["provider"]
    _ensure_keys(provider)
    last_err = None
    for attempt in range(retries):
        try:
            if provider == "anthropic_proxy":
                out = _call_anthropic_proxy(cfg["model"], prompt, temperature,
                                            max_tokens, cfg["base_url"])
            elif provider == "groq":
                out = _call_groq(cfg["model"], prompt, temperature, max_tokens)
            elif provider == "openai":
                out = _call_openai(cfg["model"], prompt, temperature, max_tokens)
            elif provider == "gemini":
                out = _call_gemini(cfg["model"], prompt, temperature, max_tokens)
            elif provider == "bedrock":
                out = _call_bedrock(cfg["model"], prompt, temperature,
                                    max_tokens, cfg.get("region", "us-east-1"))
            elif provider == "bedrock_responses":
                # GPT-5.x reasoning models need a much larger output budget
                # because reasoning consumes most of it before the answer.
                eff_max = max(max_tokens, cfg.get("default_max_output_tokens", 16384))
                out = _call_bedrock_responses(cfg["model"], prompt, temperature,
                                              eff_max)
            else:
                raise RuntimeError(f"Unknown provider {provider}")
            if cfg["sleep_after"] > 0:
                time.sleep(cfg["sleep_after"])
            _record_call_success(backbone)
            return out
        except Exception as e:
            last_err = e
            wait = min(60.0, 5.0 * (attempt + 1))
            print(f"  [LLM err {backbone} attempt {attempt+1}/{retries}] "
                  f"{type(e).__name__}: {str(e)[:200]} (wait {wait}s)", flush=True)
            time.sleep(wait)
    _record_call_failure(backbone, last_err)  # may sys.exit
    raise RuntimeError(f"{backbone} failed after {retries} retries: {last_err}")

# Compatibility surface (old API)
def call_llm(prompt: str, model: str = None, temperature: float = 0.8,
             max_tokens: int = 2048, n: int = 1, retries: int = 4) -> list:
    backbone = model or DEFAULT_BACKBONE
    if backbone not in BACKBONE_CONFIGS:
        # Old code passed model="gpt-4o-mini" string; map common ones
        alias = {
            "gpt-4o-mini": "gpt4o_mini",
            "gpt-4o": "gpt4o",
            "gpt-4.1": "gpt41",
            "claude-haiku-4-5": "claude_haiku",
        }
        backbone = alias.get(backbone, DEFAULT_BACKBONE)
    out = []
    for _ in range(n):
        r = _call(backbone, prompt, temperature, max_tokens, retries=retries)
        out.append(r["text"])
    return out

def call_llm_single(prompt: str, model: str = None, temperature: float = 0.7,
                    max_tokens: int = 2048) -> str:
    return call_llm(prompt, model=model, temperature=temperature,
                    max_tokens=max_tokens, n=1)[0]

def call_llm_multi(prompt: str, n: int = 5, model: str = None,
                   temperature: float = 0.9, max_tokens: int = 2048) -> list:
    return call_llm(prompt, model=model, temperature=temperature,
                    max_tokens=max_tokens, n=n)

# Token-counted variant for runners that need usage
def call_llm_with_usage(prompt: str, backbone: str = None,
                        temperature: float = 0.7, max_tokens: int = 2048) -> Dict:
    return _call(backbone or DEFAULT_BACKBONE, prompt, temperature, max_tokens)

if __name__ == "__main__":
    import sys
    backbone = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKBONE
    out = call_llm_with_usage("Reply with exactly: SMOKETEST_OK",
                              backbone=backbone, max_tokens=20)
    print(f"[{backbone}] -> {out['text'][:80]} tokens={out['tokens']}")
