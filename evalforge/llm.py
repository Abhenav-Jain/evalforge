"""Minimal LLM client used by the `llm` target and the LLM-judge check.
Providers: gemini (google-genai) | openai (any OpenAI-compatible server, incl. Ollama/vLLM/Groq)."""
import json, os, urllib.request

_clients = {}

def judge_available(cfg) -> bool:
    if not cfg:
        return False
    if cfg.get("provider", "gemini") == "gemini":
        return bool(os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY")))
    return bool(os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))) or "base_url" in cfg

def chat(cfg: dict, prompt: str, system: str = None, timeout: int = 60) -> str:
    provider = cfg.get("provider", "gemini")
    if provider == "gemini":
        from google import genai
        from google.genai import types
        env = cfg.get("api_key_env", "GEMINI_API_KEY")
        if env not in _clients:
            _clients[env] = genai.Client(api_key=os.environ[env])
        r = _clients[env].models.generate_content(
            model=cfg["model"], contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=cfg.get("temperature", 0)))
        return r.text or ""
    if provider == "openai":
        base = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = json.dumps({"model": cfg["model"], "messages": msgs, "temperature": cfg.get("temperature", 0)}).encode()
        req = urllib.request.Request(base + "/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    raise ValueError(f"unknown llm provider: {provider}")
