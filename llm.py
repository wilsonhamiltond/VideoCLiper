"""Which AI writes the text (hooks, captions, thumbnails, post text, narration): the local Ollama model or a cloud API.

Settings live in data/ai.json (this machine only): {"provider": "local|gemini|claude|deepseek", "keys": {...}, "models": {...}}.
API keys can also come from GEMINI_API_KEY, ANTHROPIC_API_KEY and DEEPSEEK_API_KEY; a key saved in the app wins.
"""
import json
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException

DATA = Path(os.environ.get("CLIPSTUDIO_DATA") or Path(__file__).parent / "data")
CONFIG = DATA / "ai.json"
_lock = threading.Lock()

PROVIDERS = {
    "local": {"name": "Local (Ollama)", "env": None, "model": "", "vision": True},
    "gemini": {"name": "Google Gemini", "env": "GEMINI_API_KEY", "model": "gemini-3.8-flash", "vision": True},
    "claude": {"name": "Anthropic Claude", "env": "ANTHROPIC_API_KEY", "model": "claude-opus-5-5", "vision": True},
    # DeepSeek's API reads text only: it gets the transcript and titles, never the frames.
    "deepseek": {"name": "DeepSeek", "env": "DEEPSEEK_API_KEY", "model": "deepseek-chat", "vision": False},
}


def load() -> dict:
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    cfg.setdefault("provider", "local")
    cfg.setdefault("keys", {})
    cfg.setdefault("models", {})
    if cfg["provider"] not in PROVIDERS:
        cfg["provider"] = "local"
    return cfg


def save(cfg: dict):
    DATA.mkdir(parents=True, exist_ok=True)
    with _lock:
        CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def api_key(provider: str, cfg: dict | None = None) -> str:
    cfg = cfg or load()
    env = PROVIDERS[provider]["env"]
    return (cfg["keys"].get(provider) or (os.environ.get(env, "") if env else "")).strip()


def model_for(provider: str, cfg: dict | None = None) -> str:
    cfg = cfg or load()
    return (cfg["models"].get(provider) or PROVIDERS[provider]["model"]).strip()


def current(local_model: str | None) -> dict | None:
    """The provider the app should use now: {"provider", "model", "label"}, or None when it can't run.

    `local_model` is the Ollama vision model server.py found (None when Ollama isn't running).
    """
    cfg = load()
    p = cfg["provider"]
    if p == "local":
        return {"provider": "local", "model": local_model, "label": local_model} if local_model else None
    if not api_key(p, cfg):
        return None
    m = model_for(p, cfg)
    return {"provider": p, "model": m, "label": f"{PROVIDERS[p]['name'].split()[-1]} {m}"}


def missing_message(local_hint: str) -> str:
    p = load()["provider"]
    if p == "local":
        return local_hint
    return f"No API key for {PROVIDERS[p]['name']}. Add it under Settings › AI model, or set {PROVIDERS[p]['env']}."


# ---------- calls ----------

def _media_type(b64: str) -> str:
    if b64.startswith("iVBOR"):
        return "image/png"
    if b64.startswith("UklGR"):
        return "image/webp"
    if b64.startswith("R0lG"):
        return "image/gif"
    return "image/jpeg"


def _post(url: str, payload: dict, headers: dict, timeout: float = 180) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _strict_schema(schema: dict) -> dict:
    """A copy of an Ollama-style JSON schema that Claude's structured outputs accept:
    every object closed (additionalProperties: false) and number bounds dropped."""
    s = {k: v for k, v in schema.items() if k not in ("minimum", "maximum")}
    if s.get("type") == "object":
        s["additionalProperties"] = False
        s["properties"] = {k: _strict_schema(v) for k, v in s.get("properties", {}).items()}
    if s.get("type") == "array" and isinstance(s.get("items"), dict):
        s["items"] = _strict_schema(s["items"])
    return s


def _with_schema_hint(prompt: str, schema: dict) -> str:
    return prompt + "\nReply with one JSON object that matches this JSON schema:\n" + json.dumps(schema)


def _gemini(model: str, key: str, prompt: str, schema: dict, images: list, temperature: float) -> dict:
    parts = [{"inline_data": {"mime_type": _media_type(b), "data": b}} for b in images]
    parts.append({"text": _with_schema_hint(prompt, schema)})
    res = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"contents": [{"role": "user", "parts": parts}],
                 "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}},
                {"x-goog-api-key": key})
    cand = (res.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    if not text:
        raise RuntimeError(f"empty reply ({cand.get('finishReason') or res.get('promptFeedback')})")
    return _parse_json(text)


def _deepseek(model: str, key: str, prompt: str, schema: dict, images: list, temperature: float) -> dict:
    if images:
        prompt += ("\n(You can't see the images: they were left out. Work from the text above, and where you"
                   " are asked to choose a frame, choose the one most likely to show the peak moment.)")
    res = _post("https://api.deepseek.com/chat/completions",
                {"model": model, "temperature": temperature, "response_format": {"type": "json_object"},
                 "messages": [{"role": "user", "content": _with_schema_hint(prompt, schema)}]},
                {"Authorization": f"Bearer {key}"})
    return _parse_json(res["choices"][0]["message"]["content"])


def _claude(model: str, key: str, prompt: str, schema: dict, images: list) -> dict:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("the anthropic package is missing: run  .venv\\Scripts\\pip install anthropic") from None
    client = anthropic.Anthropic(api_key=key, timeout=180.0)
    content = [{"type": "image", "source": {"type": "base64", "media_type": _media_type(b), "data": b}} for b in images]
    content.append({"type": "text", "text": prompt})
    # Structured outputs guarantee JSON matching the schema. Sampling settings (temperature) aren't accepted by
    # current Claude models, so the creative/strict balance comes from the prompt. A refused request falls back
    # server-side to another Claude model instead of failing.
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _strict_schema(schema)}},
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    text = "".join(b.text for b in response.content if b.type == "text")
    return _parse_json(text)


def chat_json(spec: dict, prompt: str, schema: dict, images: list | None, temperature: float) -> dict:
    """One JSON reply from a cloud provider (spec from current()). Raises HTTPException(502) on failure."""
    p, model = spec["provider"], spec["model"]
    key = api_key(p)
    images = images or []
    try:
        if p == "gemini":
            return _gemini(model, key, prompt, schema, images, temperature)
        if p == "deepseek":
            return _deepseek(model, key, prompt, schema, images, temperature)
        if p == "claude":
            return _claude(model, key, prompt, schema, images)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"{spec['label']} failed: {e}")
    raise HTTPException(500, f"Unknown AI provider {p}")


# ---------- settings API ----------

router = APIRouter()


def _public(cfg: dict) -> dict:
    return {"provider": cfg["provider"], "providers": [
        {"id": p, "name": v["name"], "model": model_for(p, cfg), "default_model": v["model"], "vision": v["vision"],
         "has_key": bool(api_key(p, cfg)) if v["env"] else None,
         "key_from_env": bool(v["env"] and not cfg["keys"].get(p) and os.environ.get(v["env"]))}
        for p, v in PROVIDERS.items()]}


@router.get("/api/ai-settings")
def get_settings():
    return _public(load())


@router.post("/api/ai-settings")
def set_settings(body: dict):
    """{"provider": "...", "keys": {"gemini": "..."}, "models": {"gemini": "..."}}. An empty key removes the saved one."""
    cfg = load()
    if body.get("provider") is not None:
        if body["provider"] not in PROVIDERS:
            raise HTTPException(400, "Unknown AI provider.")
        cfg["provider"] = body["provider"]
    for p, k in (body.get("keys") or {}).items():
        if p in PROVIDERS and PROVIDERS[p]["env"]:
            k = str(k or "").strip()
            if k:
                cfg["keys"][p] = k
            else:
                cfg["keys"].pop(p, None)
    for p, m in (body.get("models") or {}).items():
        if p in PROVIDERS and p != "local":
            m = str(m or "").strip()
            if m and m != PROVIDERS[p]["model"]:
                cfg["models"][p] = m
            else:
                cfg["models"].pop(p, None)
    save(cfg)
    return _public(cfg)
