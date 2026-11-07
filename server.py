"""evalforge dashboard: run suites + inspect results + try single inputs. Start: uvicorn server:app --reload
SECURITY: only allowlisted suites under examples/ can run (python/cli targets execute code). Never add an upload-a-suite endpoint publicly."""
import glob, json, os, threading, time, uuid
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from evalforge.runner import run_suite, load_cases
from evalforge.targets import build_target
from evalforge.checks import run_checks
from evalforge.core import Case

ROOT = os.path.dirname(os.path.abspath(__file__))
SUITES = {os.path.basename(os.path.dirname(p)): p for p in sorted(glob.glob(os.path.join(ROOT, "examples", "*", "suite.yaml")))
          if "liveprep_ws" not in p}          # live-server suite excluded from the public allowlist
HISTORY, LOCK, TARGETS = [], threading.Lock(), {}
NEEDS_EXPECTED = {"equals_expected", "contains_expected", "retrieval_hit"}
app = FastAPI(title="evalforge dashboard")

def _suite(name):
    if name not in SUITES: raise HTTPException(404, "unknown suite")
    return SUITES[name], yaml.safe_load(open(SUITES[name], encoding="utf-8"))

@app.get("/")
def index(): return FileResponse(os.path.join(ROOT, "index.html"))

@app.get("/api/suites")
def suites():
    out = []
    for k, p in SUITES.items():
        y = yaml.safe_load(open(p, encoding="utf-8")); cases = load_cases(y["dataset"], os.path.dirname(p))
        out.append({"id": k, "name": y.get("name", k), "target": y["target"]["type"], "n_cases": len(cases),
                    "categories": sorted({c.category for c in cases}), "gates": y.get("gates", {})})
    return out

@app.post("/api/run/{name}")
def run(name: str):
    path, _ = _suite(name)
    with LOCK:                                   # runs import user code; keep them serialized
        t0 = time.time(); rep = run_suite(path, out_dir=os.path.join(ROOT, "results"))
    prev = next((h for h in reversed(HISTORY) if h["suite"] == name), None)
    entry = {"id": uuid.uuid4().hex[:8], "suite": name, "ts": time.time(), "duration_s": round(time.time() - t0, 2),
             "ok": rep["ok"], "pass_rate": rep["metrics"].get("pass_rate"), "report": rep,
             "prev_metrics": prev["report"]["metrics"] if prev else None}
    HISTORY.append(entry); del HISTORY[:-50]
    return entry

@app.get("/api/history/{name}")
def history(name: str):
    return [{k: h[k] for k in ("id", "ts", "ok", "pass_rate")} for h in HISTORY if h["suite"] == name]

class Probe(BaseModel):
    input: str
    category: str = "default"

@app.post("/api/playground/{name}")
def playground(name: str, p: Probe):
    path, y = _suite(name)
    if len(p.input) > 2000: raise HTTPException(400, "input too long")
    try: inp = json.loads(p.input)
    except ValueError: inp = p.input
    with LOCK:
        if name not in TARGETS: TARGETS[name] = build_target(y["target"], os.path.dirname(path))
        out = TARGETS[name](inp)
    case = Case(id="playground", input=inp, category=p.category)
    checks = [c for c in run_checks(case, out, [s for s in y.get("checks", []) if s["type"] not in NEEDS_EXPECTED], {})]
    return {"text": out.text, "prediction": out.prediction, "tool_calls": out.tool_calls, "contexts": out.contexts,
            "latency_ms": round(out.latency_s * 1000, 2), "error": out.error,
            "checks": [{"type": c.type, "passed": c.passed, "detail": c.detail, "skipped": c.skipped} for c in checks]}
