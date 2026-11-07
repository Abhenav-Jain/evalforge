"""Dataset-level metrics. Each returns {name: value}. Works on any model that yields predictions."""
import math
from collections import defaultdict
from .core import CaseResult, as_str, ctx_items, expected_get, get_path

LOWER_BETTER = {"mae", "rmse", "mape", "error_rate", "slice_gap"}

def _pct(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None

def _pred(r, a):
    p = r.output.text if a.get("from") == "text" else r.output.prediction
    if a.get("prediction_key") and isinstance(p, dict): p = get_path(p, a["prediction_key"])
    return p

def _norm(x, ci): s = as_str(x).strip(); return s.lower() if ci else s

def classification(rs, a):
    ci, ek = a.get("case_insensitive", False), a.get("expected_key")
    rows = [(_norm(expected_get(r.case, ek), ci), "<error>" if r.output.error else _norm(_pred(r, a), ci))
            for r in rs if expected_get(r.case, ek) is not None]
    if not rows: return {}
    labels = sorted({t for t, _ in rows})
    out, f1s, ps, rcs = {"accuracy": sum(t == p for t, p in rows) / len(rows)}, [], [], []
    for l in labels:
        tp = sum(t == l and p == l for t, p in rows); fp = sum(t != l and p == l for t, p in rows); fn = sum(t == l and p != l for t, p in rows)
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[f"f1_{l}"], out[f"recall_{l}"], out[f"precision_{l}"] = f1, rc, pr
        f1s.append(f1); ps.append(pr); rcs.append(rc)
    out.update(f1_macro=sum(f1s) / len(f1s), precision_macro=sum(ps) / len(ps), recall_macro=sum(rcs) / len(rcs))
    return out

def regression(rs, a):
    ek, pairs, failed = a.get("expected_key"), [], 0
    for r in rs:
        e = expected_get(r.case, ek)
        if e is None: continue
        try: pairs.append((float(e), float(_pred(r, a))))
        except (TypeError, ValueError): failed += 1
    if not pairs: return {}
    n, err = len(pairs), [p - t for t, p in pairs]
    mean_t = sum(t for t, _ in pairs) / n
    ss_tot = sum((t - mean_t) ** 2 for t, _ in pairs)
    out = {"mae": sum(abs(e) for e in err) / n, "rmse": math.sqrt(sum(e * e for e in err) / n), "n_unscored": failed}
    if ss_tot > 0: out["r2"] = 1 - sum(e * e for e in err) / ss_tot
    if all(t != 0 for t, _ in pairs): out["mape"] = sum(abs(e / t) for e, (t, _) in zip(err, pairs)) / n
    return out

def retrieval(rs, a):
    ks, key, hits, rr, n = a.get("ks", [1, 3, 5]), a.get("doc_ids_key", "doc_ids"), defaultdict(int), 0.0, 0
    for r in rs:
        exp = expected_get(r.case, key)
        if not exp: continue
        n += 1; ids = [i for i, _ in ctx_items(r.output)]
        for k in ks: hits[k] += any(e in ids[:k] for e in exp)
        rr += next((1 / (i + 1) for i, x in enumerate(ids) if x in exp), 0.0)
    return {} if not n else {**{f"hit@{k}": hits[k] / n for k in ks}, "mrr": rr / n}

def latency(rs, a):
    l = [r.output.latency_s for r in rs if not r.output.error]
    out = {"error_rate": sum(1 for r in rs if r.output.error) / len(rs) if rs else 0.0}
    if l: out.update(latency_mean_s=sum(l) / len(l), latency_p50_s=_pct(l, .5), latency_p95_s=_pct(l, .95))
    return out

def slices(rs, a):
    """Per-group metric (e.g. accuracy by region) + the gap between best and worst group -> fairness / data-slice checks."""
    by, metric, groups = a["by"], a.get("metric", "accuracy"), defaultdict(list)
    for r in rs: groups[str(r.case.meta.get(by, "<none>"))].append(r)
    fn = regression if metric in ("mae", "rmse", "r2") else classification
    vals = {g: fn(v, a).get(metric) for g, v in groups.items() if len(v) >= a.get("min_size", 3)}
    vals = {g: v for g, v in vals.items() if v is not None}
    if not vals: return {}
    out = {f"{metric}[{by}={g}]": v for g, v in vals.items()}
    out["slice_min"], out["slice_gap"] = min(vals.values()), max(vals.values()) - min(vals.values())
    return out

REGISTRY = {"classification": classification, "regression": regression, "retrieval": retrieval,
            "latency": latency, "slices": slices}

def pass_stats(rs):
    st = defaultdict(int)
    for r in rs: st[r.status] += 1
    scored = st["passed"] + st["xpass"] + st["failed"]
    out = {"n_cases": len(rs), "n_xfail": st["xfail"], "n_xpass": st["xpass"], "n_failed": st["failed"], "n_passed": st["passed"]}
    if scored: out["pass_rate"] = (st["passed"] + st["xpass"]) / scored
    return out

def by_category(rs):
    cats = defaultdict(lambda: defaultdict(int))
    for r in rs: cats[r.case.category][r.status] += 1
    return {k: dict(v) for k, v in cats.items()}
