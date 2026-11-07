import csv, json, operator, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import yaml
from .core import Case, CaseResult, as_str
from .targets import build_target
from .checks import run_checks
from . import metrics as M
from . import perturb

QUALITY = {"pass_rate", "accuracy", "f1_macro", "precision_macro", "recall_macro", "r2"}
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq, "!=": operator.ne}


def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return x

def load_cases(ds, base_dir):
    cfg = {"path": ds} if isinstance(ds, str) else dict(ds)
    path = cfg["path"] if os.path.isabs(cfg["path"]) else os.path.join(base_dir, cfg["path"])
    fmt = cfg.get("format") or os.path.splitext(path)[1].lstrip(".").lower()
    cases = []
    if fmt in ("jsonl", "json"):
        txt = open(path, encoding="utf-8").read()
        rows = json.loads(txt) if fmt == "json" else [json.loads(l) for l in txt.splitlines() if l.strip()]
        for i, r in enumerate(rows):
            cases.append(Case(id=str(r.get("id", f"case-{i + 1}")), input=r["input"], expected=r.get("expected"),
                              category=r.get("category", "default"), meta=r.get("meta", {}),
                              checks=r.get("checks", []), xfail=r.get("xfail")))
    elif fmt == "csv":
        icols, ecol = cfg.get("input_columns"), cfg.get("expected_column", "expected")
        mcols = cfg.get("meta_columns", [])
        for i, r in enumerate(csv.DictReader(open(path, encoding="utf-8", newline=""))):
            if icols:
                vals = {c: _num(r[c]) for c in icols}
                inp = list(vals.values()) if cfg.get("input_as") == "list" else vals
            else:
                inp = r[cfg.get("input_column", "input")]
            cases.append(Case(id=str(r.get(cfg.get("id_column", "id")) or f"case-{i + 1}"), input=inp,
                              expected=r.get(ecol), category=r.get(cfg.get("category_column", "category")) or "default",
                              meta={c: r[c] for c in mcols}))
    else:
        raise ValueError(f"unsupported dataset format: {fmt}")
    return cases

def _run(target, cases, suite, variant="base"):
    conc = suite.get("concurrency", 1)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        outs = list(ex.map(target, [c.input for c in cases]))
    ctx = {"judge": suite.get("judge")}
    return [CaseResult(c, o, run_checks(c, o, suite.get("checks", []), ctx), variant) for c, o in zip(cases, outs)]

def compute(results, mcfgs):
    m, details = M.pass_stats(results), {}
    for cfg in mcfgs:
        args = {k: v for k, v in cfg.items() if k != "type"}
        vals = M.REGISTRY[cfg["type"]](results, args)
        m.update(vals); details[cfg["type"]] = vals
    return m, details

def eval_gate(metric, expr, metrics):
    mt = re.match(r"^\s*(>=|<=|>|<|==|!=)\s*(-?[\d.]+)\s*$", str(expr))
    if not mt: return {"metric": metric, "expr": expr, "value": None, "passed": False, "note": "bad gate expression"}
    v = metrics.get(metric)
    if v is None: return {"metric": metric, "expr": expr, "value": None, "passed": False, "note": "metric not computed"}
    return {"metric": metric, "expr": expr, "value": v, "passed": OPS[mt.group(1)](v, float(mt.group(2)))}

def baseline_compare(cur, base, cfg):
    tol = cfg.get("max_drop", 0.03)
    keys = cfg.get("metrics") or [k for k in base if (k in QUALITY or k in ("mae", "rmse", "mrr") or k.startswith("hit@")) and "@" not in k.split("hit@")[0]]
    out = []
    for k in keys:
        if k not in cur or k not in base: continue
        if k in M.LOWER_BETTER:
            change = (cur[k] - base[k]) / max(abs(base[k]), 1e-9); bad = change > tol
        else:
            change = base[k] - cur[k]; bad = change > tol
        out.append({"metric": f"baseline:{k}", "expr": f"regression <= {tol}", "value": round(change, 4), "passed": not bad,
                    "note": f"baseline={base[k]:.4f} current={cur[k]:.4f}"})
    return out

def run_suite(path, baseline_path=None, update_baseline=False, out_dir="results", set_baseline=False):
    suite = yaml.safe_load(open(path, encoding="utf-8"))
    base_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, base_dir)
    name = suite.get("name") or os.path.splitext(os.path.basename(path))[0]
    target = build_target(suite["target"], base_dir)
    cases = load_cases(suite["dataset"], base_dir)
    mcfgs = suite.get("metrics", [{"type": "latency"}])
    base_res = _run(target, cases, suite)
    metrics, details = compute(base_res, mcfgs)
    variants = {}
    for p in suite.get("perturbations", []):
        vc = [replace(c, input=ni) for c in cases if (ni := perturb.apply(p, c.input)) is not None]
        if not vc: continue
        vres = _run(target, vc, suite, p)
        vm, _ = compute(vres, mcfgs)
        variants[p] = {"n": len(vc), "metrics": vm, "results": vres}
        for k, v in vm.items():
            metrics[f"{k}@{p}"] = v
            if k in QUALITY and k in metrics: metrics[f"{k}_drop@{p}"] = metrics[k] - v
            if k in ("mae", "rmse") and k in metrics: metrics[f"{k}_increase@{p}"] = (v - metrics[k]) / max(abs(metrics[k]), 1e-9)
    gates = [eval_gate(k, e, metrics) for k, e in suite.get("gates", {}).items()]
    bcfg = dict(suite.get("baseline", {}))
    bpath = baseline_path or bcfg.get("path")
    if bpath and not os.path.isabs(bpath): bpath = os.path.join(base_dir, bpath) if not baseline_path else bpath
    if bpath and update_baseline:
        os.makedirs(os.path.dirname(bpath) or ".", exist_ok=True)
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, open(bpath, "w"), indent=2)
    elif bpath and os.path.exists(bpath):
        gates += baseline_compare(metrics, json.load(open(bpath)), bcfg)
    fails = [r for r in base_res if r.status in ("failed", "xpass") or (r.status == "xfail")]
    report = {"name": name, "target": suite["target"].get("type"), "n_cases": len(cases), "metrics": metrics, "details": details,
              "categories": M.by_category(base_res), "gates": gates, "ok": all(g["passed"] for g in gates),
              "cases": [{"id": r.case.id, "category": r.case.category, "status": r.status,
                         "input": as_str(r.case.input)[:160], "output": (r.output.error or as_str(r.output.prediction if r.output.text == "" else r.output.text))[:160],
                         "tool_calls": [c.get("name") for c in r.output.tool_calls], "xfail": r.case.xfail,
                         "failed_checks": [f"{c.type}: {c.detail}" for c in r.failed_checks]} for r in base_res if r.status != "unchecked" or r.output.error],
              "skipped_checks": sorted({c.type for r in base_res for c in r.checks if c.skipped})}
    from .report import write_report
    write_report(report, os.path.join(out_dir, name))
    return report
