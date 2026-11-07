"""Core data model shared by every target, check and metric."""
import json, os, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Case:
    id: str
    input: Any
    expected: Any = None
    category: str = "default"
    meta: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    xfail: Optional[str] = None  # known issue: failing is tracked, not fatal


@dataclass
class Output:
    text: str = ""
    prediction: Any = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    contexts: List[Any] = field(default_factory=list)  # retrieved docs: str or {"id":..,"text":..}
    raw: Any = None
    latency_s: float = 0.0
    error: Optional[str] = None


@dataclass
class CheckResult:
    type: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class CaseResult:
    case: Case
    output: Output
    checks: List[CheckResult]
    variant: str = "base"

    @property
    def failed_checks(self):
        return [c for c in self.checks if not c.skipped and not c.passed]

    @property
    def status(self) -> str:
        active = [c for c in self.checks if not c.skipped]
        if self.output.error and not any(c.type == "expects_error" for c in active):
            bad = True
        elif not active:
            return "unchecked"
        else:
            bad = any(not c.passed for c in active)
        if self.case.xfail:
            return "xfail" if bad else "xpass"
        return "failed" if bad else "passed"


# ---------- helpers ----------
_TOKEN = re.compile(r"[^.\[\]]+|\[\d+\]")

def get_path(obj: Any, path: Optional[str]) -> Any:
    """Tiny JSONPath: 'choices[0].message.content' or '$.a.b'."""
    if not path or path == "$":
        return obj
    path = path[2:] if path.startswith("$.") else path.lstrip("$")
    cur = obj
    try:
        for tok in _TOKEN.findall(path):
            cur = cur[int(tok[1:-1])] if tok.startswith("[") else (cur.get(tok) if isinstance(cur, dict) else None)
            if cur is None:
                return None
    except (IndexError, KeyError, TypeError):
        return None
    return cur

def expand_env(s: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), s)

def to_plain(x: Any) -> Any:
    """numpy / torch / pandas scalars and arrays -> plain python."""
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes, dict, list)):
        return x.tolist()
    return x

def expected_get(case: Case, key: Optional[str]) -> Any:
    if key and isinstance(case.expected, dict):
        return case.expected.get(key)
    return case.expected

_KNOWN = ("text", "answer", "output", "response", "prediction", "label", "tool_calls", "contexts", "context", "retrieved")

def normalize(raw: Any) -> Output:
    """Turn whatever a target returned into an Output."""
    if isinstance(raw, Output):
        return raw
    raw = to_plain(raw)
    if isinstance(raw, dict) and any(k in raw for k in _KNOWN):
        text = next((raw[k] for k in ("text", "answer", "output", "response") if raw.get(k) is not None), None)
        pred = raw.get("prediction", raw.get("label"))
        if pred is None:
            pred = text
        if text is None:
            text = "" if pred is None else (pred if isinstance(pred, str) else json.dumps(pred, default=str))
        ctx = next((raw[k] for k in ("contexts", "context", "retrieved") if raw.get(k) is not None), [])
        calls = [c if isinstance(c, dict) else {"name": str(c), "args": {}} for c in (raw.get("tool_calls") or [])]
        return Output(text=str(text), prediction=pred, tool_calls=calls, contexts=list(ctx), raw=raw)
    if isinstance(raw, str):
        return Output(text=raw, prediction=raw, raw=raw)
    if isinstance(raw, dict):
        return Output(text=json.dumps(raw, default=str), prediction=raw, raw=raw)
    return Output(text="" if raw is None else str(raw), prediction=raw, raw=raw)

def ctx_items(out: Output) -> List[Tuple[Any, str]]:
    items = []
    for i, c in enumerate(out.contexts):
        if isinstance(c, dict):
            items.append((c.get("id", i), str(c.get("text") or c.get("content") or "")))
        else:
            items.append((i, str(c)))
    return items

def as_str(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, default=str, sort_keys=True)
