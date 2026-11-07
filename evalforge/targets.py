"""Targets = anything you want to evaluate. Every target returns an Output.
type: python | http | cli | llm"""
import importlib, json, os, shlex, subprocess, sys, time, urllib.request
from .core import Output, expand_env, get_path, normalize
from . import llm as llm_mod


def _fill(obj, inp):
    if isinstance(obj, str):
        if obj.strip() == "{{input}}":
            return inp
        return obj.replace("{{input}}", inp if isinstance(inp, str) else json.dumps(inp))
    if isinstance(obj, dict):
        return {k: _fill(v, inp) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, inp) for v in obj]
    return obj


class Target:
    def call(self, inp):  # returns raw value / dict / Output
        raise NotImplementedError

    def __call__(self, inp) -> Output:
        t0 = time.time()
        try:
            out = normalize(self.call(inp))
        except Exception as e:  # a crashing target is a test result, not a framework crash
            out = Output(error=f"{type(e).__name__}: {e}")
        out.latency_s = time.time() - t0
        return out


class PythonTarget(Target):
    """callable: 'package.module:function'. Works for sklearn / torch / tf / any python code.
    factory: true -> the callable builds and returns the predict function (load weights once)."""
    def __init__(self, cfg, base_dir):
        for p in (os.getcwd(), base_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
        mod, fn = cfg["callable"].split(":")
        f = getattr(importlib.import_module(mod), fn)
        self.kwargs = cfg.get("kwargs", {})
        if cfg.get("factory"):
            f, self.kwargs = f(**self.kwargs), {}
        self.fn = f

    def call(self, inp):
        return self.fn(inp, **self.kwargs)


class HTTPTarget(Target):
    def __init__(self, cfg):
        self.url = expand_env(cfg["url"])
        self.method = cfg.get("method", "POST").upper()
        self.headers = {k: expand_env(str(v)) for k, v in cfg.get("headers", {}).items()}
        self.body = cfg.get("body", "{{input}}")
        self.resp = cfg.get("response", {})
        self.timeout = cfg.get("timeout", 30)
        self.retries = cfg.get("retries", 1)

    def call(self, inp):
        data = None if self.method == "GET" else json.dumps(_fill(self.body, inp)).encode()
        hdrs = {"Content-Type": "application/json", **self.headers}
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=data, headers=hdrs, method=self.method)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = r.read().decode("utf-8", "replace")
                break
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
        try:
            js = json.loads(payload)
        except ValueError:
            return {"text": payload}
        if not self.resp:
            return js
        return {k: get_path(js, p) for k, p in self.resp.items()}


class CLITarget(Target):
    """command: 'python predict.py --text {input}'  (or stdin: true to pipe the input)."""
    def __init__(self, cfg):
        self.cmd, self.stdin, self.timeout = cfg["command"], cfg.get("stdin", False), cfg.get("timeout", 60)

    def call(self, inp):
        s = inp if isinstance(inp, str) else json.dumps(inp)
        argv = [a.replace("{input}", s) for a in shlex.split(self.cmd)]
        p = subprocess.run(argv, input=s if self.stdin else None, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip()[-300:] or f"exit {p.returncode}")
        txt = p.stdout.strip()
        try:
            return json.loads(txt)
        except ValueError:
            return txt


class LLMTarget(Target):
    """Prompt-level evaluation of a raw LLM: template with {{input}}, optional system prompt."""
    def __init__(self, cfg):
        self.cfg, self.template, self.system = cfg, cfg.get("template", "{{input}}"), cfg.get("system")

    def call(self, inp):
        s = inp if isinstance(inp, str) else json.dumps(inp)
        return llm_mod.chat(self.cfg, self.template.replace("{{input}}", s), self.system)


def build_target(cfg, base_dir=".") -> Target:
    t = cfg["type"]
    if t == "python": return PythonTarget(cfg, base_dir)
    if t == "http": return HTTPTarget(cfg)
    if t == "cli": return CLITarget(cfg)
    if t == "llm": return LLMTarget(cfg)
    raise ValueError(f"unknown target type: {t}")
