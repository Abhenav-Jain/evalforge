import json, os

def _f(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

def write_report(rep, out):
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=2, default=str)
    m = rep["metrics"]
    L = [f"# Eval report: {rep['name']}", "",
         f"**Result: {'PASS' if rep['ok'] else 'FAIL'}** | target: `{rep['target']}` | cases: {rep['n_cases']}", ""]
    if rep["gates"]:
        L += ["## Quality gates", "", "| Metric | Gate | Value | |", "|---|---|---|---|"]
        for g in rep["gates"]:
            L.append(f"| {g['metric']} | {g['expr']} | {_f(g['value'])} | {'✅' if g['passed'] else '❌'} {g.get('note', '')} |")
        L.append("")
    L += ["## Metrics", "", "| Metric | Value |", "|---|---|"] + [f"| {k} | {_f(v)} |" for k, v in sorted(m.items()) if isinstance(v, (int, float)) and "@" not in k]
    pv = {k: v for k, v in m.items() if "@" in k and ("_drop@" in k or "_increase@" in k)}
    if pv: L += ["", "## Robustness (vs. unperturbed)", "", "| Metric | Value |", "|---|---|"] + [f"| {k} | {_f(v)} |" for k, v in sorted(pv.items())]
    if len(rep["categories"]) > 1 or "default" not in rep["categories"]:
        L += ["", "## By category", "", "| Category | passed | failed | xfail | xpass |", "|---|---|---|---|---|"]
        for c, s in sorted(rep["categories"].items()):
            L.append(f"| {c} | {s.get('passed', 0)} | {s.get('failed', 0)} | {s.get('xfail', 0)} | {s.get('xpass', 0)} |")
    bad = [c for c in rep["cases"] if c["status"] in ("failed", "xpass")]
    if bad:
        L += ["", f"## Failures ({len(bad)})", ""]
        for c in bad[:25]:
            L.append(f"- **{c['id']}** [{c['category']}] `{c['status']}` input=`{c['input'][:80]}` -> {'; '.join(c['failed_checks']) or 'output: ' + c['output']}")
    kn = [c for c in rep["cases"] if c["status"] == "xfail"]
    if kn:
        L += ["", f"## Known issues tracked ({len(kn)})", ""] + [f"- {c['id']}: {c['xfail']}" for c in kn]
    if rep["skipped_checks"]:
        L += ["", f"_Skipped checks (not configured): {', '.join(rep['skipped_checks'])}_"]
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
