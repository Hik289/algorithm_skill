"""
Provider-agnostic LLM client for AlgoSkill experiments.

The public experiment scripts use generic backend aliases such as
``default``, ``fast``, ``strong``, and ``judge``. Concrete providers, model
identifiers, endpoints, and keys are supplied through environment variables or
an optional local JSON config file. This keeps the repository portable and
avoids exposing which hosted services were used for a particular run.

Compatible public API:
- call_llm(prompt, model=..., temperature=, max_tokens=, n=)
- call_llm_single(prompt, ...) -> str
- call_llm_multi(prompt, n=, ...) -> list[str]
- call_llm_with_usage(prompt, backend=...) -> {"text": ..., "tokens": ...}
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


DEFAULT_BACKEND = os.environ.get("ALGOSKILL_BACKEND", "default")
CONFIG_PATH = os.environ.get("ALGOSKILL_BACKEND_CONFIG", "")

GENERIC_BACKENDS = ("default", "fast", "strong", "judge")


def _env_name(alias: str, suffix: str) -> str:
    clean = alias.upper().replace("-", "_")
    return f"ALGOSKILL_{clean}_{suffix}"


def _maybe_float(value: Optional[str], default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _maybe_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value in (None, ""):
        return default
    return int(value)


def _read_backend_from_env(alias: str) -> Optional[Dict]:
    api_style = os.environ.get(_env_name(alias, "API_STYLE"))
    model = os.environ.get(_env_name(alias, "MODEL"))
    base_url = os.environ.get(_env_name(alias, "BASE_URL"))
    api_key = os.environ.get(_env_name(alias, "API_KEY"))
    if not any([api_style, model, base_url, api_key]):
        return None
    cfg = {
        "api_style": api_style or "chat_completions",
        "model": model or "",
        "base_url": base_url or "",
        "api_key_env": _env_name(alias, "API_KEY"),
        "sleep_after": _maybe_float(os.environ.get(_env_name(alias, "SLEEP_AFTER")), 0.0),
    }
    max_output = _maybe_int(os.environ.get(_env_name(alias, "MAX_OUTPUT_TOKENS")), None)
    if max_output is not None:
        cfg["default_max_output_tokens"] = max_output
    region = os.environ.get(_env_name(alias, "REGION"))
    if region:
        cfg["region"] = region
    return cfg


def _load_json_config() -> Dict[str, Dict]:
    if not CONFIG_PATH:
        return {}
    path = Path(CONFIG_PATH).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ALGOSKILL_BACKEND_CONFIG not found: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("ALGOSKILL_BACKEND_CONFIG must be a JSON object")
    return data


def _normalize_config(alias: str, cfg: Dict) -> Dict:
    out = dict(cfg)
    out.setdefault("api_style", out.pop("provider", "chat_completions"))
    out.setdefault("model", "")
    out.setdefault("base_url", "")
    out.setdefault("api_key_env", _env_name(alias, "API_KEY"))
    out.setdefault("sleep_after", 0.0)
    return out


def _build_backend_configs() -> Dict[str, Dict]:
    configs: Dict[str, Dict] = {}
    configs.update(_load_json_config())
    for alias in GENERIC_BACKENDS:
        env_cfg = _read_backend_from_env(alias)
        if env_cfg is not None:
            configs[alias] = env_cfg

    # Always expose the generic aliases so argparse choices stay stable.
    # If no env/config exists for an alias, it inherits the default at call time.
    if "default" not in configs:
        configs["default"] = {
            "api_style": os.environ.get("ALGOSKILL_API_STYLE", "chat_completions"),
            "model": os.environ.get("ALGOSKILL_MODEL", ""),
            "base_url": os.environ.get("ALGOSKILL_BASE_URL", ""),
            "api_key_env": "ALGOSKILL_API_KEY",
            "sleep_after": _maybe_float(os.environ.get("ALGOSKILL_SLEEP_AFTER"), 0.0),
        }
        max_output = _maybe_int(os.environ.get("ALGOSKILL_MAX_OUTPUT_TOKENS"), None)
        if max_output is not None:
            configs["default"]["default_max_output_tokens"] = max_output
    for alias in GENERIC_BACKENDS:
        configs.setdefault(alias, {"alias_of": "default"})

    return {k: _normalize_config(k, v) for k, v in configs.items()}


BACKBONE_CONFIGS = _build_backend_configs()


def resolve_backend(name: Optional[str]) -> str:
    backend = name or DEFAULT_BACKEND
    if backend not in BACKBONE_CONFIGS:
        aliases = ", ".join(sorted(BACKBONE_CONFIGS))
        raise ValueError(f"Unknown backend '{backend}'. Available aliases: {aliases}")
    seen = set()
    while "alias_of" in BACKBONE_CONFIGS[backend]:
        if backend in seen:
            raise ValueError(f"Backend alias cycle involving '{backend}'")
        seen.add(backend)
        backend = BACKBONE_CONFIGS[backend]["alias_of"]
    return backend


def _api_key(cfg: Dict) -> str:
    key_env = cfg.get("api_key_env", "ALGOSKILL_API_KEY")
    key = os.environ.get(key_env, "")
    if not key:
        raise RuntimeError(f"{key_env} not set")
    return key


def _model(cfg: Dict) -> str:
    model = cfg.get("model") or os.environ.get("ALGOSKILL_MODEL", "")
    if not model:
        raise RuntimeError("No model configured for selected backend")
    return model


def _tokens(prompt=0, completion=0, total=0) -> Dict[str, int]:
    total = total or (prompt + completion)
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(total or 0),
    }


def _endpoint(base_url: str, suffix: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base:
        raise RuntimeError("BASE_URL is required for this backend")
    if base.endswith(suffix):
        return base
    return f"{base}/{suffix}"


def _call_chat_completions(cfg: Dict, prompt: str, temperature: float, max_tokens: int) -> Dict:
    url = _endpoint(cfg.get("base_url", ""), "chat/completions")
    body = {
        "model": _model(cfg),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {_api_key(cfg)}", "content-type": "application/json"},
        json=body,
        timeout=180,
    )
    r.raise_for_status()
    d = r.json()
    choices = d.get("choices", [])
    if not choices:
        raise RuntimeError(f"chat_completions returned no choices: {d}")
    text = choices[0].get("message", {}).get("content", "") or ""
    usage = d.get("usage", {})
    return {
        "text": text,
        "tokens": _tokens(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        ),
    }


def _call_responses(cfg: Dict, prompt: str, temperature: float, max_tokens: int) -> Dict:
    url = _endpoint(cfg.get("base_url", ""), "responses")
    body = {
        "model": _model(cfg),
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {_api_key(cfg)}", "content-type": "application/json"},
        json=body,
        timeout=600,
    )
    r.raise_for_status()
    d = r.json()
    text = d.get("output_text", "") or ""
    if not text:
        for item in d.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")
    usage = d.get("usage", {})
    return {
        "text": text,
        "tokens": _tokens(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("total_tokens", 0),
        ),
    }


def _call_messages(cfg: Dict, prompt: str, temperature: float, max_tokens: int) -> Dict:
    base_url = (cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("messages backend requires BASE_URL")
    headers = {
        "content-type": "application/json",
        "x-api-key": _api_key(cfg),
        "connection": "close",
    }
    headers.update(cfg.get("extra_headers", {}))
    body = {
        "model": _model(cfg),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    with requests.Session() as sess:
        r = sess.post(f"{base_url}/v1/messages", headers=headers, json=body, timeout=(10, 120))
    r.raise_for_status()
    d = r.json()
    text = ""
    for block in d.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    usage = d.get("usage", {})
    return {
        "text": text,
        "tokens": _tokens(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        ),
    }


def _call_generate_content(cfg: Dict, prompt: str, temperature: float, max_tokens: int) -> Dict:
    base_url = (cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("generate_content backend requires BASE_URL")
    url = f"{base_url}/v1/models/{_model(cfg)}:generateContent?key={_api_key(cfg)}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    r = requests.post(url, json=body, timeout=180)
    r.raise_for_status()
    d = r.json()
    cands = d.get("candidates", [])
    if not cands:
        raise RuntimeError(f"generate_content returned no candidates: {d}")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    usage = d.get("usageMetadata", {})
    return {
        "text": text,
        "tokens": _tokens(
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
            usage.get("totalTokenCount", 0),
        ),
    }


def _call_converse(cfg: Dict, prompt: str, temperature: float, max_tokens: int) -> Dict:
    base_url = (cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("converse backend requires BASE_URL")
    url = f"{base_url}/model/{_model(cfg)}/converse"
    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {_api_key(cfg)}", "content-type": "application/json"},
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    text = ""
    for block in d.get("output", {}).get("message", {}).get("content", []):
        if "text" in block:
            text += block["text"]
    usage = d.get("usage", {})
    return {
        "text": text,
        "tokens": _tokens(
            usage.get("inputTokens", 0),
            usage.get("outputTokens", 0),
            usage.get("totalTokens", 0),
        ),
    }


API_STYLES = {
    "chat_completions": _call_chat_completions,
    "responses": _call_responses,
    "messages": _call_messages,
    "generate_content": _call_generate_content,
    "converse": _call_converse,
}


_CONSECUTIVE_FAILS: Dict[str, int] = {}
CONSECUTIVE_FAIL_THRESHOLD = int(os.environ.get("ALGOSKILL_FAST_FAIL_THRESHOLD", "3"))


def _record_success(backend: str):
    _CONSECUTIVE_FAILS[backend] = 0


def _record_failure(backend: str, err: Exception):
    count = _CONSECUTIVE_FAILS.get(backend, 0) + 1
    _CONSECUTIVE_FAILS[backend] = count
    if count >= CONSECUTIVE_FAIL_THRESHOLD:
        import sys

        msg = (
            f"[FAST-FAIL] backend={backend}: {count} consecutive failed calls "
            f"(threshold={CONSECUTIVE_FAIL_THRESHOLD}). Last error: "
            f"{type(err).__name__}: {str(err)[:200]}"
        )
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
        os._exit(101)


def _call(backend: str, prompt: str, temperature: float, max_tokens: int, retries: int = 4) -> Dict:
    resolved = resolve_backend(backend)
    cfg = BACKBONE_CONFIGS[resolved]
    api_style = cfg.get("api_style", "chat_completions")
    if api_style not in API_STYLES:
        raise RuntimeError(f"Unsupported api_style '{api_style}' for backend '{resolved}'")
    effective_max = max(max_tokens, int(cfg.get("default_max_output_tokens", max_tokens)))
    last_err = None
    for attempt in range(retries):
        try:
            out = API_STYLES[api_style](cfg, prompt, temperature, effective_max)
            sleep_after = float(cfg.get("sleep_after", 0.0))
            if sleep_after > 0:
                time.sleep(sleep_after)
            _record_success(resolved)
            return out
        except Exception as err:
            last_err = err
            wait = min(60.0, 5.0 * (attempt + 1))
            print(
                f"  [LLM err backend={resolved} attempt {attempt + 1}/{retries}] "
                f"{type(err).__name__}: {str(err)[:200]} (wait {wait}s)",
                flush=True,
            )
            time.sleep(wait)
    _record_failure(resolved, last_err)
    raise RuntimeError(f"backend '{resolved}' failed after {retries} retries: {last_err}")


def call_llm(
    prompt: str,
    model: str = None,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    n: int = 1,
    retries: int = 4,
) -> List[str]:
    backend = model or DEFAULT_BACKEND
    out = []
    for _ in range(n):
        result = _call(backend, prompt, temperature, max_tokens, retries=retries)
        out.append(result["text"])
    return out


def call_llm_single(
    prompt: str,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    return call_llm(prompt, model=model, temperature=temperature, max_tokens=max_tokens, n=1)[0]


def call_llm_multi(
    prompt: str,
    n: int = 5,
    model: str = None,
    temperature: float = 0.9,
    max_tokens: int = 2048,
) -> List[str]:
    return call_llm(prompt, model=model, temperature=temperature, max_tokens=max_tokens, n=n)


def call_llm_with_usage(
    prompt: str,
    backbone: str = None,
    backend: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Dict:
    selected = backend or backbone or DEFAULT_BACKEND
    return _call(selected, prompt, temperature, max_tokens)


if __name__ == "__main__":
    import sys

    selected = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKEND
    out = call_llm_with_usage(
        "Reply with exactly: SMOKETEST_OK",
        backend=selected,
        max_tokens=20,
    )
    print(f"[{selected}] -> {out['text'][:80]} tokens={out['tokens']}")
