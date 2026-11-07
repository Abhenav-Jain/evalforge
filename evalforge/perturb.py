"""Robustness variants of each input (works on text, numeric lists and dict features)."""
import random, copy

def _typos(s, rng):
    cs = list(s)
    for i in range(len(cs) - 1):
        if cs[i].isalpha() and cs[i + 1].isalpha() and rng.random() < 0.08:
            cs[i], cs[i + 1] = cs[i + 1], cs[i]
    return "".join(cs)

def _noise(x, rel, rng):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return x + rng.gauss(0, rel * (abs(x) or 1))
    if isinstance(x, list): return [_noise(v, rel, rng) for v in x]
    if isinstance(x, dict): return {k: _noise(v, rel, rng) for k, v in x.items()}
    return x

def apply(name, inp, seed=0):
    rng = random.Random(f"{seed}:{name}:{inp}")
    kind, _, arg = name.partition(":")
    if kind == "typos" and isinstance(inp, str): return _typos(inp, rng)
    if kind == "upper" and isinstance(inp, str): return inp.upper()
    if kind == "whitespace" and isinstance(inp, str): return "  " + inp.replace(" ", "   ") + "  "
    if kind == "noise" and not isinstance(inp, str): return _noise(copy.deepcopy(inp), float(arg or 0.05), rng)
    return None  # not applicable to this input type
