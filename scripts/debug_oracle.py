"""Where does the solver fail even with an oracle judge? (diagnostic; run on the box with assets)"""
import random, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from eval_fake_jev import make_fake
from xwjev.jev import JevJudge
from xwjev.puzzle import load
from xwjev.solver import Solver

assets, d = Path(sys.argv[1]), Path(sys.argv[2])
solver = Solver(assets, use_jev=False, verbose=False)
judge = JevJudge.__new__(JevJudge); judge.__init__(cache_dir=None); judge.backend = "fake"; judge.model = "oracle"
solver.jev = judge
orig_candidates = solver._candidates
captured = {}
def cap(pz):
    st = orig_candidates(pz); captured["states"] = st; return st
solver._candidates = cap
for p in sorted(d.glob("*.puz")):
    pz = load(p); truth = {s.name: pz.answer(s) for s in pz.slots}
    judge._call = make_fake(truth, 1.0, random.Random(0)); judge.cache = {}
    res = solver.solve(pz)
    states = captured["states"]
    wrong = [(st, truth[st.slot.name]) for st, so in zip(states, res.slots) if so["answer"] != truth[st.slot.name]]
    print(f"\n{p.name}: {len(wrong)} wrong slots")
    for st, t in wrong:
        got = "".join(res.grid[r][c] for r, c in st.slot.cells)
        rank = sorted(st.prior, key=lambda a: -st.prior[a]).index(t) + 1 if t in st.prior else None
        print(f"  {st.slot.name:4s} truth={t:16s} got={got:16s} in_vocab={solver.retr.in_vocab(t)!s:5s} in_cands={t in st.cands!s:5s} rank={rank} p_null={st.p_null:.2f}  clue={st.slot.clue[:50]!r}")
