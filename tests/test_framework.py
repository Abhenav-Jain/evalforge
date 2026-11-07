"""Unit tests for the framework itself: metric math, plumbing, gates, baseline, and end-to-end exit codes."""
import json, math, os, subprocess, sys, textwrap
import pytest
from evalforge.core import Case, CaseResult, CheckResult, Output, get_path, normalize
from evalforge import metrics as M, perturb
from evalforge.runner import eval_gate, baseline_compare, run_suite
from evalforge.checks import run_checks

def cr(exp, pred, **meta):
    c = Case(id="x", input=0, expected=exp, meta=meta)
    return CaseResult(c, normalize(pred), [])

def test_get_path():
    js = {"choices": [{"message": {"content": "hi"}}], "a": {"b": 2}}
    assert get_path(js, "choices[0].message.content") == "hi"
    assert get_path(js, "$.a.b") == 2 and get_path(js, "a.zzz") is None and get_path(js, "choices[5].x") is None

def test_normalize_variants():
    assert normalize("hi").prediction == "hi"
    assert normalize(3).prediction == 3 and normalize(3).text == "3"
    o = normalize({"answer": "A", "contexts": [{"id": "d1", "text": "t"}], "tool_calls": [{"name": "f", "args": {}}]})
    assert o.text == "A" and o.prediction == "A" and o.contexts[0]["id"] == "d1" and o.tool_calls[0]["name"] == "f"
    class Arr:  # numpy/torch-like
        def tolist(self): return [1, 2]
    assert normalize(Arr()).prediction == [1, 2]

def test_classification_metrics():
    rs = [cr("a", "a"), cr("a", "a"), cr("a", "b"), cr("b", "b")]
    m = M.classification(rs, {})
    assert m["accuracy"] == 0.75
    assert math.isclose(m["recall_a"], 2 / 3) and math.isclose(m["precision_a"], 1.0)
    assert math.isclose(m["f1_macro"], (0.8 + 2 / 3) / 2)

def test_errors_count_as_wrong():
    bad = CaseResult(Case("e", 0, expected="a"), Output(error="boom"), [])
    assert M.classification([cr("a", "a"), bad], {})["accuracy"] == 0.5

def test_regression_metrics():
    rs = [cr(1.0, 2.0), cr(2.0, 2.0), cr(3.0, 1.0)]
    m = M.regression(rs, {})
    assert math.isclose(m["mae"], 1.0) and math.isclose(m["rmse"], math.sqrt(5 / 3))
    assert math.isclose(m["r2"], 1 - 5 / 2)

def test_retrieval_metrics():
    c = Case("r", "q", expected={"doc_ids": ["d2"]})
    r = CaseResult(c, normalize({"answer": "x", "contexts": [{"id": "d1", "text": ""}, {"id": "d2", "text": ""}]}), [])
    m = M.retrieval([r], {"ks": [1, 2]})
    assert m["hit@1"] == 0 and m["hit@2"] == 1 and m["mrr"] == 0.5

def test_slices_gap():
    rs = [cr("a", "a", g="x"), cr("a", "a", g="x"), cr("a", "a", g="x"), cr("a", "b", g="y"), cr("a", "b", g="y"), cr("a", "a", g="y")]
    m = M.slices(rs, {"by": "g", "metric": "accuracy"})
    assert math.isclose(m["slice_gap"], 1 - 1 / 3)

def test_gate_expressions():
    assert eval_gate("acc", ">=0.9", {"acc": 0.95})["passed"]
    assert not eval_gate("acc", ">=0.9", {"acc": 0.5})["passed"]
    assert not eval_gate("missing", ">=0.9", {})["passed"]           # unknown metric must fail loudly
    assert not eval_gate("acc", "banana", {"acc": 1})["passed"]

def test_baseline_regression_directions():
    res = baseline_compare({"accuracy": 0.80, "mae": 1.2}, {"accuracy": 0.90, "mae": 1.0}, {"max_drop": 0.03})
    assert all(not g["passed"] for g in res)                           # accuracy fell, mae rose
    ok = baseline_compare({"accuracy": 0.91, "mae": 0.9}, {"accuracy": 0.90, "mae": 1.0}, {"max_drop": 0.03})
    assert all(g["passed"] for g in ok)

def test_perturb_deterministic_and_applicability():
    assert perturb.apply("typos", "hello world today") == perturb.apply("typos", "hello world today")
    assert perturb.apply("noise:0.1", "text") is None and perturb.apply("typos", [1, 2]) is None
    assert perturb.apply("noise:0.1", [1.0, 2.0]) != [1.0, 2.0]

def test_checks_and_xfail_status():
    case = Case("c", "q", xfail="known")
    out = normalize({"text": "hello", "tool_calls": [{"name": "execute_command", "args": {"command": "rm -rf /"}}]})
    res = run_checks(case, out, [{"type": "tool_not_called", "name": "execute_command"}], {})
    assert not res[0].passed
    assert CaseResult(case, out, res).status == "xfail"
    assert CaseResult(Case("d", "q", xfail="k"), out, [CheckResult("x", True)]).status == "xpass"
    assert CaseResult(Case("d", "q"), Output(error="boom"), [CheckResult("target_error", False)]).status == "failed"

def test_skipped_judge_not_counted():
    c = Case("c", "q")
    res = run_checks(c, normalize("hi"), [{"type": "llm_judge", "rubric": "r"}], {"judge": None})
    assert res[0].skipped and CaseResult(c, normalize("hi"), res).status == "unchecked"

def _suite(tmp_path, body, rows):
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "t.py").write_text(textwrap.dedent(body))
    (tmp_path / "s.yaml").write_text(textwrap.dedent("""
        name: tmp
        target: {type: python, callable: "t:f"}
        dataset: data.jsonl
        checks: [{type: equals_expected}]
        gates: {pass_rate: ">=1.0"}
    """))
    return str(tmp_path / "s.yaml")

def test_end_to_end_pass_and_fail(tmp_path):
    rows = [{"id": "1", "input": "a", "expected": "A"}, {"id": "2", "input": "b", "expected": "B"}]
    good = _suite(tmp_path, "def f(x): return x.upper()", rows)
    assert run_suite(good, out_dir=str(tmp_path / "o"))["ok"]
    sys.modules.pop("t", None)
    (tmp_path / "t.py").write_text("def f(x): return x")      # broken model
    assert not run_suite(good, out_dir=str(tmp_path / "o"))["ok"]

def test_crashing_target_is_a_failure_not_a_crash(tmp_path):
    sys.modules.pop("t", None)
    s = _suite(tmp_path, "def f(x): raise ValueError('boom')", [{"id": "1", "input": "a", "expected": "A"}])
    rep = run_suite(s, out_dir=str(tmp_path / "o"))
    assert not rep["ok"] and rep["cases"][0]["status"] == "failed" and "boom" in rep["cases"][0]["output"]

def test_baseline_update_then_compare(tmp_path):
    sys.modules.pop("t", None)
    s = _suite(tmp_path, "def f(x): return x.upper()", [{"id": "1", "input": "a", "expected": "A"}])
    b = str(tmp_path / "base.json")
    run_suite(s, baseline_path=b, update_baseline=True, out_dir=str(tmp_path / "o"))
    assert json.load(open(b))["pass_rate"] == 1.0
    assert run_suite(s, baseline_path=b, out_dir=str(tmp_path / "o"))["ok"]

@pytest.mark.parametrize("suite,code", [("classifier", 0), ("rag", 0), ("llm_safety", 0), ("liveprep_guardrail", 0)])
def test_examples_pass_via_cli(suite, code, tmp_path):
    root = os.path.dirname(os.path.dirname(__file__))
    r = subprocess.run([sys.executable, "-m", "evalforge", "run", f"examples/{suite}/suite.yaml", "--out", str(tmp_path)],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == code, r.stdout + r.stderr

def test_cli_exit_codes(tmp_path):
    root = os.path.dirname(os.path.dirname(__file__))
    r = subprocess.run([sys.executable, "-m", "evalforge", "run", "nope.yaml"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 2
