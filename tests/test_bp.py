import math

import numpy as np

from xwjev.bp import SlotVar, run_bp
from xwjev.puzzle import Puzzle


def _letters(w):
    return [ord(c) - 65 for c in w]


def _var(idx, cands, probs, p_null, cross):
    arr = np.array([_letters(c) for c in cands])
    logp = np.log(np.array(probs) * (1 - p_null))
    return SlotVar(idx, len(cands[0]), arr, logp, p_null, cross)


def test_bp_resolves_ambiguity_through_crossings():
    # 2x2 grid: 1A = ?? , 1D = ??, 3A, 2D ; truth: AT / TO  -> rows "AT","TO"; cols "AT","TO"
    black = [[False, False], [False, False]]
    pz = Puzzle.from_grid(black, {1: "", 3: ""}, {1: "", 2: ""})
    names = {s.name: s.idx for s in pz.slots}
    cross = lambda si: [pz.crossing(si, i) for i in range(2)]
    # 1A ambiguous between AT and IT; 1D says AT strongly; 3A between TO/GO; 2D says TO strongly
    v = [
        _var(names["1A"], ["IT", "AT"], [0.55, 0.45], 0.05, cross(names["1A"])),
        _var(names["3A"], ["GO", "TO"], [0.6, 0.4], 0.05, cross(names["3A"])),
        _var(names["1D"], ["AT"], [1.0], 0.05, cross(names["1D"])),
        _var(names["2D"], ["TO"], [1.0], 0.05, cross(names["2D"])),
    ]
    v.sort(key=lambda x: x.idx)
    res = run_bp(v, n_iter=10)
    a1 = v[names["1A"]]
    assert a1.cands[int(np.argmax(res.slot_belief[names["1A"]]))].tolist() == _letters("AT")
    a3 = v[names["3A"]]
    assert a3.cands[int(np.argmax(res.slot_belief[names["3A"]]))].tolist() == _letters("TO")
    for (si, pos), m in res.cell_marg.items():
        assert math.isclose(m.sum(), 1.0, rel_tol=1e-4)


def test_bp_null_mass_does_not_force_wrong_candidate():
    black = [[False, False], [False, False]]
    pz = Puzzle.from_grid(black, {1: "", 3: ""}, {1: "", 2: ""})
    names = {s.name: s.idx for s in pz.slots}
    cross = lambda si: [pz.crossing(si, i) for i in range(2)]
    # 1A has only a wrong candidate "XX" with low confidence; all crossings are confident about A? / ?T
    v = [
        _var(names["1A"], ["XX"], [1.0], 0.9, cross(names["1A"])),
        _var(names["3A"], ["TO"], [1.0], 0.02, cross(names["3A"])),
        _var(names["1D"], ["AT"], [1.0], 0.02, cross(names["1D"])),
        _var(names["2D"], ["TO"], [1.0], 0.02, cross(names["2D"])),
    ]
    v.sort(key=lambda x: x.idx)
    res = run_bp(v, n_iter=10)
    si = names["1A"]
    assert res.slot_null[si] > 0.9
    assert int(np.argmax(res.cell_marg[(si, 0)])) == ord("A") - 65
    assert int(np.argmax(res.cell_marg[(si, 1)])) == ord("T") - 65
