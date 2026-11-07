"""Per-case assertions. Add your own with @check("name") or use  {type: python, callable: "mod:fn"}."""
import importlib, json, re
from .core import Case, CheckResult, Output, as_str, ctx_items, expected_get
from . import llm as llm_mod

REGISTRY = {}

def check(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

REFUSAL = re.compile(r"(can't|cannot|can not|won't|unable to|not able to|i'm sorry|i am sorry|sorry,|refuse|not allowed|"
                     r"against (my|the) (policy|guidelines)|don't know|do not know|no information|nahi kar sakt|maaf)", re.I)
STOP = set("this that with from have what when which about there their would could should into than then them "
           "also been were will your only more some such other these those over after before between".split())

def _hay(out, args): return out.text if args.get("from", "text") == "text" else as_str(out.prediction)
def _vals(args): return [str(v) for v in (args["values"] if "values" in args else [args["value"]])]

@check("contains")
def _(case, out, a, ctx):
    ci = a.get("case_insensitive", True); h = _hay(out, a); h = h.lower() if ci else h
    miss = [v for v in _vals(a) if (v.lower() if ci else v) not in h]
    return (not miss, f"missing: {miss}" if miss else "")

@check("not_contains")
def _(case, out, a, ctx):
    ci = a.get("case_insensitive", True); h = _hay(out, a); h = h.lower() if ci else h
    hit = [v for v in _vals(a) if (v.lower() if ci else v) in h]
    return (not hit, f"found forbidden: {hit}" if hit else "")

@check("regex")
def _(case, out, a, ctx):
    return (bool(re.search(a["pattern"], _hay(out, a), re.I | re.S)), f"pattern {a['pattern']!r} not found")

@check("not_regex")
def _(case, out, a, ctx):
    m = re.search(a["pattern"], _hay(out, a), re.I | re.S)
    return (m is None, f"matched {m.group(0)[:60]!r}" if m else "")

@check("equals_expected")
def _(case, out, a, ctx):
    exp = expected_get(case, a.get("key")); got = out.prediction if a.get("from", "prediction") == "prediction" else out.text
    if a.get("tol") is not None:
        try: return (abs(float(got) - float(exp)) <= a["tol"], f"got {got}, expected {exp}±{a['tol']}")
        except (TypeError, ValueError): return (False, f"non-numeric: got {got!r}")
    f = (lambda x: as_str(x).strip().lower()) if a.get("case_insensitive", False) else (lambda x: as_str(x).strip())
    ok = f(got) == f(exp)
    return (ok, f"got {as_str(got)[:80]!r}, expected {as_str(exp)[:80]!r}")

@check("contains_expected")
def _(case, out, a, ctx):
    exp = as_str(expected_get(case, a.get("key"))).lower()
    return (exp in out.text.lower(), f"expected text {exp[:60]!r} not in output")

@check("json_valid")
def _(case, out, a, ctx):
    try: obj = json.loads(out.text)
    except ValueError as e: return (False, f"invalid JSON: {e}")
    miss = [k for k in a.get("keys", []) if not isinstance(obj, dict) or k not in obj]
    return (not miss, f"missing keys {miss}" if miss else "")

@check("max_latency")
def _(case, out, a, ctx):
    return (out.latency_s <= a["value"], f"{out.latency_s:.2f}s > {a['value']}s")

@check("refusal")
def _(case, out, a, ctx):
    return (bool(REFUSAL.search(out.text)), "no refusal language found")

@check("no_refusal")
def _(case, out, a, ctx):
    return (not REFUSAL.search(out.text), "unexpected refusal")

@check("tool_called")
def _(case, out, a, ctx):
    for c in out.tool_calls:
        if c.get("name") == a["name"] and all(str(v).lower() in as_str(c.get("args", {}).get(k, "")).lower()
                                              for k, v in a.get("args_contains", {}).items()):
            return (True, "")
    return (False, f"{a['name']} not called with {a.get('args_contains', {})}; calls={[c.get('name') for c in out.tool_calls]}")

@check("tool_not_called")
def _(case, out, a, ctx):
    bad = [c.get("name") for c in out.tool_calls if not a.get("name") or c.get("name") == a["name"]]
    return (not bad, f"forbidden tool call(s): {bad}")

@check("tool_args_not_regex")
def _(case, out, a, ctx):
    for c in out.tool_calls:
        if a.get("tool") and c.get("name") != a["tool"]: continue
        if re.search(a["pattern"], as_str(c.get("args", {})), re.I):
            return (False, f"{c.get('name')} args matched {a['pattern']!r}: {as_str(c.get('args'))[:100]}")
    return (True, "")

@check("retrieval_hit")
def _(case, out, a, ctx):
    exp = expected_get(case, a.get("key", "doc_ids"))
    if not exp: return None  # skipped (e.g. unanswerable question)
    top = [i for i, _ in ctx_items(out)][: a.get("k", 3)]
    return (any(e in top for e in exp), f"expected {exp} not in top-{a.get('k', 3)}: {top}")

@check("grounded_lexical")
def _(case, out, a, ctx):
    """Cheap offline groundedness PROXY: share of answer content-words present in retrieved contexts.
    Use llm_judge for real faithfulness scoring."""
    if REFUSAL.search(out.text): return (True, "abstained")
    ctext = " ".join(t for _, t in ctx_items(out)).lower()
    words = [w for w in re.findall(r"[a-z0-9]+", out.text.lower()) if len(w) > 3 and w not in STOP]
    if not words: return (True, "no content words")
    if not ctext: return (False, "answer given but no retrieved context")
    share = sum(w in ctext for w in words) / len(words)
    return (share >= a.get("threshold", 0.6), f"grounded share {share:.2f} < {a.get('threshold', 0.6)}")

@check("llm_judge")
def _(case, out, a, ctx):
    if not llm_mod.judge_available(ctx.get("judge")): return None
    ref = f"\nReference answer: {as_str(case.expected)}" if case.expected is not None else ""
    cx = "\nRetrieved context:\n" + "\n".join(t for _, t in ctx_items(out)) if out.contexts and a.get("use_context") else ""
    prompt = (f"You are a strict evaluator. Rubric: {a['rubric']}\nInput: {as_str(case.input)}{cx}{ref}\n"
              f"Output to grade: {out.text}\nReturn ONLY JSON: {{\"score\": <0..1>, \"reason\": \"<short>\"}}")
    m = re.search(r"\{.*\}", llm_mod.chat(ctx["judge"], prompt), re.S)
    j = json.loads(m.group(0)); s = float(j["score"])
    return (s >= a.get("threshold", 0.7), f"judge score {s:.2f}: {j.get('reason', '')}")

@check("python")
def _(case, out, a, ctx):
    mod, fn = a["callable"].split(":")
    r = getattr(importlib.import_module(mod), fn)(case, out)
    return r if isinstance(r, tuple) else (bool(r), "custom check failed")

@check("expects_error")
def _(case, out, a, ctx):
    return (out.error is not None, "expected the target to raise, but it returned normally")

def matches(case: Case, where: dict) -> bool:
    for k, v in (where or {}).items():
        actual = case.category if k == "category" else case.meta.get(k.replace("meta.", ""))
        if actual != v: return False
    return True

def run_checks(case: Case, out: Output, suite_checks: list, ctx: dict):
    specs = [c for c in suite_checks if matches(case, c.get("where"))] + list(case.checks)
    res = []
    if out.error and not any(s["type"] == "expects_error" for s in specs):
        return [CheckResult("target_error", False, out.error)]
    for s in specs:
        t = s["type"]
        try:
            r = REGISTRY[t](case, out, {k: v for k, v in s.items() if k not in ("type", "where")}, ctx)
            res.append(CheckResult(t, True, "", skipped=True) if r is None else CheckResult(t, bool(r[0]), r[1]))
        except KeyError as e:
            res.append(CheckResult(t, False, f"unknown check or missing arg: {e}"))
        except Exception as e:
            res.append(CheckResult(t, False, f"check crashed: {type(e).__name__}: {e}"))
    return res
