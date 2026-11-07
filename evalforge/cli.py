import argparse, sys
from .runner import run_suite

def main(argv=None):
    ap = argparse.ArgumentParser(prog="evalforge", description="Universal evaluation harness for AI/ML systems")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or more suite files")
    r.add_argument("suites", nargs="+")
    r.add_argument("--out", default="results")
    r.add_argument("--baseline", help="baseline json to compare against (overrides suite config)")
    r.add_argument("--update-baseline", action="store_true", help="write current metrics as the new baseline")
    a = ap.parse_args(argv)
    code = 0
    for s in a.suites:
        try:
            rep = run_suite(s, a.baseline, a.update_baseline, a.out)
        except Exception as e:
            print(f"[{s}] CONFIG/RUNTIME ERROR: {type(e).__name__}: {e}", file=sys.stderr); code = max(code, 2); continue
        print(f"\n=== {rep['name']}: {'PASS' if rep['ok'] else 'FAIL'} ({rep['n_cases']} cases) ===")
        for g in rep["gates"]:
            v = g["value"]; print(f"  {'ok  ' if g['passed'] else 'FAIL'} {g['metric']} {g['expr']}  (value={v if v is None else round(v, 4)}) {g.get('note', '')}")
        if not rep["ok"]: code = max(code, 1)
    return code

if __name__ == "__main__":
    sys.exit(main())
