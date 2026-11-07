# evalforge: a universal evaluation harness for AI/ML systems

One framework, one YAML config, any system under test: **LLM apps, agents, RAG pipelines, classical ML models, deep-learning
models, REST APIs, CLI tools, or plain Python functions.** It runs your dataset through the system, scores it, and **fails the
build** when quality drops below your thresholds or regresses against a saved baseline.

```
 dataset (jsonl / csv / json)        Target  (what you test)             Checks + Metrics (how)             Gates
 ┌───────────────────────┐   ──►   ┌───────────────────────┐   ──►   ┌───────────────────────────┐   ──►  ┌───────────┐
 │ input, expected, meta │         │ python | http | cli | │         │ per-case assertions       │        │ thresholds│
 └───────────────────────┘         │ llm    (adapter layer)│         │ dataset metrics, slices,  │        │ baseline  │
                                   └───────────────────────┘         │ robustness, latency       │        │ exit code │
                                                                     └───────────────────────────┘        └───────────┘
```
The framework knows nothing about your app. A **target adapter** turns your system into a normalized `Output`
(`text`, `prediction`, `tool_calls`, `contexts`, `latency`, `error`); everything else is reusable.

## Quick start
```bash
pip install -r requirements.txt
python -m evalforge run examples/classifier/suite.yaml examples/rag/suite.yaml     # reports in results/<suite>/report.md
python -m evalforge run examples/classifier/suite.yaml --update-baseline            # save a baseline
python -m evalforge run examples/classifier/suite.yaml                              # later runs also gate on regression vs baseline
pytest -q                                                                           # framework's own tests
```
Exit code: `0` all gates pass, `1` a gate failed, `2` config/runtime error. Wire it into CI (see `.github/workflows/eval.yml`).

## Evaluate your own system in 3 steps
**1. Dataset**: `jsonl` rows `{id, input, expected, category, meta, checks, xfail}`, or a `csv` (tabular ML) with `input_columns`.

**2. Target**: pick one:
| type | use for | config |
|---|---|---|
| `python` | sklearn / XGBoost / PyTorch / TF / ONNX, any function or class | `callable: "module:function"` (+ `factory: true` to load weights once) |
| `http` | any deployed API / LLM app | `url`, `headers` (`${ENV}` expansion), `body` with `{{input}}`, `response:` field mapping (`choices[0].message.content`) |
| `cli` | any command line tool | `command: "python predict.py {input}"` or `stdin: true` |
| `llm` | prompt-level testing of a raw model | `provider: gemini | openai (any OpenAI-compatible server)`, `template`, `system` |

```python
# PyTorch / sklearn: this is the whole adapter
def make():                                   # callable: "adapter:make", factory: true
    import torch; m = torch.load("model.pt").eval()
    def predict(x):
        with torch.no_grad(): return int(m(torch.tensor([x])).argmax())
    return predict
```
**3. Suite YAML**: checks, metrics, gates:
```yaml
name: my-model
target: {type: python, callable: "adapter:make", factory: true}
dataset: {path: data.csv, input_columns: [f1, f2, f3], input_as: list, expected_column: label, meta_columns: [region]}
metrics: [{type: classification}, {type: slices, by: region}, {type: latency}]
perturbations: ["noise:0.05"]                       # robustness
gates: {accuracy: ">=0.9", f1_macro: ">=0.85", slice_gap: "<=0.1", "accuracy_drop@noise:0.05": "<=0.03", latency_p95_s: "<=0.2"}
baseline: {path: baseline.json, max_drop: 0.02}     # regression gate vs the last accepted run
```

## What it can measure
| Area | Checks (per case) / Metrics (per dataset) |
|---|---|
| **Any output** | `contains`, `not_contains`, `regex`, `not_regex`, `equals_expected` (+`tol`), `json_valid`, `max_latency`, custom `python` check, `expects_error` |
| **LLM / agents** | `refusal`, `tool_called`, `tool_not_called`, `tool_args_not_regex` (prompt-injection / secret-leak / destructive-command tests), `llm_judge` (rubric scoring, pluggable judge) |
| **RAG** | `retrieval_hit@k`, `grounded_lexical` (offline proxy), `llm_judge` with `use_context`; metrics `hit@k`, `mrr` |
| **Classification** | accuracy, per-class + macro precision / recall / F1 |
| **Regression** | MAE, RMSE, R², MAPE |
| **Fairness / slices** | any metric per `meta` group + `slice_gap` / `slice_min` |
| **Robustness** | perturbations `typos`, `upper`, `whitespace`, `noise:<rel>` → `<metric>_drop@<perturbation>` |
| **Reliability** | error rate (crashes count as failures), latency mean/p50/p95 |
| **Regression tracking** | baseline file, per-metric `max_drop`; `xfail` rows track known weaknesses and flag when they get fixed (`xpass`) |

## Examples (all runnable offline)
| Suite | System under test | Shows |
|---|---|---|
| `examples/classifier` | tabular ML model (CSV) | classification metrics, slice gap by region, noise robustness |
| `examples/rag` | retrieval + answer pipeline | retrieval hit/MRR, groundedness, refusal on unanswerable questions, typo robustness, known issues |
| `examples/llm_safety` | mock tool-calling agent | injection / secret-leak / destructive-command checks on `tool_calls` |
| `examples/liveprep_guardrail` | a plain function (LivePrep's regex guardrail) | 7 tracked bypasses as `xfail`, false-positive checks |
| `examples/liveprep_ws` | **untested template**: live LivePrep WebSocket backend | python adapter over a streaming API (run only in a sandbox: LivePrep executes tools for real) |

Findings the examples surface in their demo systems: the RAG app over-refuses short questions (2 `xfail`) and loses ~56% pass-rate under
typos (gate pinned at the current level; tighten as you improve it); the guardrail misses 7 dangerous command variants.

## Extending
Add a check: `@check("name")` in `checks.py` (or `{type: python, callable: "mod:fn"}` in YAML, no code change in the framework).
Add a metric: register a function in `metrics.py`. Add a target type: subclass `Target` and return anything `normalize()` understands.

## Status / honesty notes
Tested: all core modules via 20 unit/integration tests; the 4 offline examples. **Not exercised in this repo's tests** (need keys/servers):
`llm` target, Gemini/OpenAI judge, `http` target against a live server, `liveprep_ws`. `grounded_lexical` is a cheap proxy, not true faithfulness.
Roadmap: Ragas/DeepEval metric bridges, Appium/Playwright targets, SQL/Athena result store, trend dashboard, JUnit XML output.

## Dashboard (visual UI)
```bash
pip install -r requirements.txt
uvicorn server:app --reload        # open http://localhost:8000
```
Pick a suite, **Run evaluation** (gates, metrics with ▲▼ delta vs the previous run, category bars, expandable per-case results with filters, robustness),
and use the **Playground** to send your own input to the live target and see its output and checks instantly.
Security: only allowlisted suites under `examples/` can run from the UI; do not add a "upload your own suite" endpoint to a public deployment (python/cli targets execute code).
