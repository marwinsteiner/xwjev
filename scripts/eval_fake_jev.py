"""Exercise the full Jev code path with a simulated judge (no API key needed).

    python scripts/eval_fake_jev.py --assets assets puzzles/ [--acc 0.9]

The fake judge knows the solution: with probability --acc it puts most mass on the true answer
when it is among the candidates (else on NONE_OF_THESE); otherwise it answers like the
retriever would. Reports accuracy as a function of judge quality, which bounds what a real Jev
can add on top of candidate recall.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from xwjev.jev import NONE, JevJudge
from xwjev.puzzle import load
from xwjev.solver import Solver


def make_fake(truth_by_key: dict[str, str], acc: float, rng: random.Random):
    async def fake_call(state, questions):
        out = {}
        for key, q in questions.items():
            opts = list(q["criteria"])
            truth = truth_by_key.get(key)
            n = len(opts)
            if rng.random() < acc:
                if truth in opts:
                    p = {o: 0.1 / max(n - 1, 1) for o in opts}
                    p[truth] = 0.9
                else:
                    p = {o: 0.2 / max(n - 1, 1) for o in opts}
                    p[NONE] = 0.8
            else:  # uninformative: mild preference for the first option (mimics jev-1.13 ordering bias)
                p = {o: 1.0 for o in opts}
                p[opts[0]] = 2.0
                z = sum(p.values())
                p = {o: v / z for o, v in p.items()}
            out[key] = p
        return out

    return fake_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("dir")
    ap.add_argument("--acc", type=float, default=0.9)
    ap.add_argument("--budget", type=float, default=45.0)
    a = ap.parse_args()
    rng = random.Random(0)

    solver = Solver(Path(a.assets), use_jev=False, budget=a.budget, verbose=False)
    judge = JevJudge.__new__(JevJudge)
    judge.__init__(per_request=12, concurrency=8, cache_dir=None)
    judge.backend = "fake"
    judge.model = f"fake-{a.acc}"
    solver.jev = judge

    rows = []
    for p in sorted(Path(a.dir).iterdir()):
        if p.suffix.lower() not in (".puz", ".json", ".xd", ".ipuz"):
            continue
        pz = load(p)
        truth = {s.name: pz.answer(s) for s in pz.slots}
        judge._call = make_fake(truth, a.acc, rng)
        judge.cache = {}
        res = solver.solve(pz)
        acc = res.accuracy
        rows.append({"puzzle": p.name, **acc, "elapsed": res.elapsed})
        print(f"{p.name:28s} {res.elapsed:5.1f}s letters {acc['letter_acc']:.3f} words {acc['word_acc']:.3f} perfect={acc['perfect']}  jev={judge.stats}")
        sys.stdout.flush()
    n = len(rows)
    print(f"\nfake judge acc={a.acc}: {n} puzzles mean letters {sum(r['letter_acc'] for r in rows) / n:.4f} words {sum(r['word_acc'] for r in rows) / n:.4f} perfect {sum(r['perfect'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
