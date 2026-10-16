"""Live smoke test of the Jev integration. Needs TYPESAFE_API_KEY (or OPENROUTER_API_KEY).

    python scripts/jev_smoke.py

Sends a handful of clues with candidate lists and prints Jev's probabilities and latency.
"""

from __future__ import annotations

import sys
import time

from xwjev.jev import JevJudge, JudgeRequest

CASES = [
    ("Tortilla chip dip", 5, ["SALSA", "QUESO", "RANCH", "HUMUS", "AIOLI"], "SALSA"),
    ("UFO passenger", 5, ["ALIEN", "PILOT", "ROBOT", "ANGEL", "GHOST"], "ALIEN"),
    ("Phisher's scheme", 4, ["SCAM", "RUSE", "PLOY", "HOAX", "PLAN"], "SCAM"),
    ("Many a braces wearer, agewise", 4, ["TEEN", "TYKE", "AGED", "BABY", "GEEK"], "TEEN"),
    ("Pea, e.g", 14, ["CRIBBAGEMARKER", "GREENVEGETABLE", "LEGUMINOUSSEED"], "CRIBBAGEMARKER"),
]


def main() -> int:
    j = JevJudge(per_request=3, concurrency=2)
    if not j.available:
        print("no TYPESAFE_API_KEY / OPENROUTER_API_KEY in the environment")
        return 1
    print(f"backend={j.backend} model={j.model}")
    reqs = [JudgeRequest(key=str(i), clue=c, length=n, candidates=[(a, a.lower()) for a in cands]) for i, (c, n, cands, _) in enumerate(CASES)]
    t0 = time.time()
    res = j.judge(reqs, puzzle_meta={"title": "smoke test"})
    dt = time.time() - t0
    ok = 0
    for i, (clue, n, cands, truth) in enumerate(CASES):
        probs = res.get(str(i), {})
        top = max(probs, key=probs.get) if probs else None
        ok += int(top == truth)
        shown = ", ".join(f"{k}={v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:4])
        print(f"{clue!r:40s} top={top!s:16s} truth={truth:16s} {shown}")
    print(f"{ok}/{len(CASES)} correct, {dt:.2f}s wall, stats={j.stats}")
    return 0 if ok >= 4 else 2


if __name__ == "__main__":
    sys.exit(main())
