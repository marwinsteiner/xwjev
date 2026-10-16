"""Solve orchestration: candidates -> Jev judging -> belief propagation -> pattern expansion -> local search.

Wall-clock budget is enforced: every stage after the first BP pass checks the deadline, so the
solver always returns a complete grid inside `budget` seconds (default 45 s, under the 49 s target).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bp import BPResult, SlotVar, run_bp
from .jev import NONE, JevJudge, JudgeRequest
from .puzzle import Puzzle, Slot
from .retriever import Candidate, Retriever


@dataclass
class SlotState:
    slot: Slot
    cands: dict[str, Candidate] = field(default_factory=dict)
    prior: dict[str, float] = field(default_factory=dict)  # mixed probability per candidate (sums to 1 - p_null)
    p_null: float = 0.3
    judged: set[str] = field(default_factory=set)
    jev_none: float | None = None  # Jev's NONE_OF_THESE mass from the latest judging round
    qvec: object = None


@dataclass
class SolveResult:
    grid: list[list[str]]
    elapsed: float
    timing: dict[str, float]
    accuracy: dict
    slots: list[dict]
    jev_stats: dict
    t_perfect: float | None = None  # seconds until the grid first became fully correct (needs the solution; None if never)

    def to_dict(self) -> dict:
        return {"grid": self.grid, "elapsed": self.elapsed, "t_perfect": self.t_perfect, "timing": self.timing, "accuracy": self.accuracy, "slots": self.slots, "jev": self.jev_stats}


class Solver:
    def __init__(
        self,
        assets: Path,
        use_jev: bool = True,
        budget: float = 45.0,
        verbose: bool = False,
        jev_model: str | None = None,
        k_retrieve: int = 200,
        k_judge: int = 40,
        k_expand: int = 80,
        jev_weight: float = 0.75,
        sim_scale: float = 20.0,
        exact_bonus: float = 2.0,
        base_null: float = 0.15,
        bp_iters: int = 16,
    ):
        self.retr = Retriever(Path(assets))
        self.jev = JevJudge(model=jev_model, cache_dir=Path(assets) / ".jev_cache") if use_jev else None
        if self.jev is not None and not self.jev.available:
            if verbose:
                print("[xwjev] no TYPESAFE_API_KEY / OPENROUTER_API_KEY found: running retriever-only")
            self.jev = None
        self.budget = budget
        self.verbose = verbose
        self.k_retrieve, self.k_judge, self.k_expand = k_retrieve, k_judge, k_expand
        self.jev_weight, self.sim_scale, self.exact_bonus, self.base_null = jev_weight, sim_scale, exact_bonus, base_null
        self.bp_iters = bp_iters
        self.trace: list[dict] | None = None  # set to [] to record grid snapshots for visualisation
        self._t0 = 0.0
        self._pz: Puzzle | None = None

    def log(self, *a):
        if self.verbose:
            print("[xwjev]", *a, flush=True)

    def _mark(self, grid: list[list[str]]) -> None:
        """Benchmark bookkeeping only (never used by the solver's decisions): time at which the grid first became fully correct."""
        if self._t_perfect is None and self._pz is not None and self._pz.solution is not None and self._pz.score(grid).get("perfect"):
            self._t_perfect = round(time.time() - self._t0, 3)

    def _snap(self, stage: str, grid: list[list[str]] | None = None, cell_marg: dict | None = None, states: list[SlotState] | None = None) -> None:
        """Record one visualisation frame: letters + per-cell confidence (max marginal) at this moment."""
        if grid is not None:
            self._mark(grid)
        if self.trace is None or self._pz is None:
            return
        pz = self._pz
        conf = [[0.0 for _ in range(pz.width)] for _ in range(pz.height)]
        if cell_marg is not None and states is not None:
            grid = pz.empty_grid()
            for si, st in enumerate(states):
                for i, (r, c) in enumerate(st.slot.cells):
                    m = cell_marg[(si, i)]
                    j = int(np.argmax(m))
                    if not grid[r][c] or m[j] > conf[r][c]:
                        grid[r][c] = chr(65 + j)
                        conf[r][c] = float(m[j])
        elif grid is not None:
            for r in range(pz.height):
                for c in range(pz.width):
                    if grid[r][c]:
                        conf[r][c] = 1.0
        else:
            grid = pz.empty_grid()
        self.trace.append({"stage": stage, "t": round(time.time() - self._t0, 3), "grid": [row[:] for row in grid], "conf": [[round(v, 3) for v in row] for row in conf], "acc": pz.score(grid) if pz.solution else {}})

    # ------------------------------------------------------------------ stage 1: candidates
    def _candidates(self, pz: Puzzle) -> list[SlotState]:
        clues = [s.clue for s in pz.slots]
        lengths = [s.length for s in pz.slots]
        Q = self.retr.encode_clues(clues, lengths)
        states = []
        for i, s in enumerate(pz.slots):
            st = SlotState(slot=s, qvec=Q[i])
            for ans, n in self.retr.exact(s.clue, s.length).items():
                st.cands[ans] = Candidate(ans, self.retr.text_of(ans), exact=n, src={"exact"})
            for ans, sim in self.retr.search_vec(Q[i], s.length, k=self.k_retrieve):
                c = st.cands.get(ans)
                if c is None:
                    st.cands[ans] = Candidate(ans, self.retr.text_of(ans), retr=sim, src={"retr"})
                else:
                    c.retr = sim
                    c.src.add("retr")
            missing = [a for a, c in st.cands.items() if "retr" not in c.src]
            if missing:
                for a, sim in zip(missing, self.retr.sim_for(Q[i], missing)):
                    st.cands[a].retr = sim
            self._set_prior(st)
            states.append(st)
        return states

    def _retr_scores(self, st: SlotState) -> dict[str, float]:
        return {a: self.sim_scale * c.retr + (self.exact_bonus * math.log1p(c.exact) if c.exact else 0.0) for a, c in st.cands.items()}

    def _set_prior(self, st: SlotState) -> None:
        """Mix retriever softmax with Jev probabilities (where available) into st.prior / st.p_null."""
        st.__dict__.pop("_wcache", None)  # word-score caches depend on p_null and the candidate pool
        st.__dict__.pop("_pool", None)
        if not st.cands:
            st.prior, st.p_null = {}, 1.0
            return
        sc = self._retr_scores(st)
        mx = max(sc.values())
        z = sum(math.exp(v - mx) for v in sc.values())
        retr_p = {a: math.exp(v - mx) / z for a, v in sc.items()}
        has_exact = any(c.exact for c in st.cands.values())
        p_null = 0.03 if has_exact else self.base_null
        judged = {a: c.jev for a, c in st.cands.items() if c.jev is not None}
        if judged:
            jev_none = st.jev_none if st.jev_none is not None else max(0.0, 1.0 - sum(judged.values()))
            # trust Jev in proportion to how decisive it was: a diffuse distribution says little
            conf = max(max(judged.values()), jev_none)
            w = self.jev_weight * min(1.0, conf / 0.5)
            mixed = {}
            for a in st.cands:
                jp = judged.get(a, 0.0)
                mixed[a] = (1 - w) * retr_p[a] + w * jp
            p_null = max(0.02, (1 - w) * p_null + w * jev_none)
            tot = sum(mixed.values())
            st.prior = {a: (1 - p_null) * v / tot for a, v in mixed.items()}
        else:
            st.prior = {a: (1 - p_null) * v for a, v in retr_p.items()}
        st.p_null = p_null

    # ------------------------------------------------------------------ stage 2: Jev
    def _judge(self, pz: Puzzle, states: list[SlotState], per_slot: dict[int, list[str]], patterns: dict[int, str] | None, deadline: float) -> int:
        if self.jev is None:
            return 0
        reqs = []
        for si, answers in per_slot.items():
            st = states[si]
            answers = [a for a in answers if a in st.cands and a not in st.judged]
            if not answers:
                continue
            cand_list = [(a, st.cands[a].text) for a in answers]
            reqs.append(JudgeRequest(key=f"{st.slot.name}", clue=st.slot.clue, length=st.slot.length, candidates=cand_list, pattern=patterns.get(si) if patterns else None))
        if not reqs:
            return 0
        meta = {"title": pz.title, "date": pz.date, "notes": pz.notes}
        res = self.jev.judge(reqs, puzzle_meta=meta, deadline=deadline)
        n = 0
        name_to_idx = {st.slot.name: i for i, st in enumerate(states)}
        for key, probs in res.items():
            st = states[name_to_idx[key]]
            st.jev_none = probs.get(NONE, 0.0)
            for a, p in probs.items():
                if a in st.cands:
                    st.cands[a].jev = p
                    st.judged.add(a)
                    n += 1
        for st in states:
            self._set_prior(st)
        return n

    # ------------------------------------------------------------------ stage 3: BP
    def _bp(self, pz: Puzzle, states: list[SlotState], iters: int | None = None) -> BPResult:
        vars_ = []
        for si, st in enumerate(states):
            answers = list(st.prior.keys())
            K = len(answers)
            cands = np.zeros((K, st.slot.length), dtype=np.int64)
            for k, a in enumerate(answers):
                cands[k] = [ord(ch) - 65 for ch in a]
            logp = np.array([math.log(max(st.prior[a], 1e-12)) for a in answers], dtype=np.float64)
            cross = [pz.crossing(si, i) for i in range(st.slot.length)]
            vars_.append(SlotVar(si, st.slot.length, cands, logp, st.p_null, cross))
        on_iter = None
        if self.trace is not None:
            label = self._bp_label

            def on_iter(k, snap):
                self._snap(f"{label} sweep {k}", cell_marg=snap, states=states)

        return run_bp(vars_, n_iter=iters or self.bp_iters, on_iter=on_iter)

    def _grid_from_bp(self, pz: Puzzle, states: list[SlotState], bp: BPResult) -> list[list[str]]:
        grid = pz.empty_grid()
        for si, st in enumerate(states):
            for i, (r, c) in enumerate(st.slot.cells):
                if not grid[r][c]:
                    grid[r][c] = chr(65 + int(np.argmax(bp.cell_marg[(si, i)])))
        return grid

    @staticmethod
    def _pattern(st: SlotState, bp: BPResult, si: int, thresh: float) -> str:
        out = []
        for i in range(st.slot.length):
            cm = bp.cell_marg[(si, i)]
            j = int(np.argmax(cm))
            out.append(chr(65 + j) if cm[j] >= thresh else "?")
        return "".join(out)

    # ------------------------------------------------------------------ stage 4: expansion
    def _expand(self, pz: Puzzle, states: list[SlotState], bp: BPResult, deadline: float, thresh: float = 0.8) -> tuple[int, dict[int, list[str]], dict[int, str]]:
        new_per_slot: dict[int, list[str]] = {}
        patterns: dict[int, str] = {}
        n_new = 0
        for si, st in enumerate(states):
            answers = list(st.prior.keys())
            best = float(bp.slot_belief[si].max()) if len(answers) else 0.0
            if best >= 0.97 and bp.slot_null[si] < 0.03:
                continue
            pat = self._pattern(st, bp, si, thresh)
            patterns[si] = pat
            found = self.retr.pattern_candidates(st.slot.length, pat, k=self.k_expand, qvec=st.qvec)
            fresh = []
            for a, sim in found:
                if a not in st.cands:
                    st.cands[a] = Candidate(a, self.retr.text_of(a), retr=sim, src={"pattern"})
                    fresh.append(a)
                    n_new += 1
            # re-judge the existing candidates compatible with the pattern as well (pattern context sharpens Jev)
            compatible = [a for a in answers if all(p == "?" or p == ch for p, ch in zip(pat, a))]
            compatible.sort(key=lambda a: -st.prior.get(a, 0.0))
            for a in compatible[:10]:
                st.judged.discard(a)
            new_per_slot[si] = (compatible[:10] + fresh)[: self.k_judge]
            self._set_prior(st)
        return n_new, new_per_slot, patterns

    # ------------------------------------------------------------------ stage 4b: repair
    def _repair(self, pz: Puzzle, states: list[SlotState], grid: list[list[str]], bp: BPResult, k_per_pos: int = 8, k_total: int = 15, confident: float = 0.9) -> tuple[int, dict[int, list[str]], dict[int, str]]:
        per_slot: dict[int, list[str]] = {}
        patterns: dict[int, str] = {}
        n_new = 0
        if self.jev is None:
            return 0, per_slot, patterns  # nothing can adjudicate the neighbours without a judge
        for si, st in enumerate(states):
            w = "".join(grid[r][c] for r, c in st.slot.cells)
            c = st.cands.get(w)
            if c is not None and c.jev is not None and c.jev >= confident:
                continue  # Jev already vouched for this exact fill
            L = st.slot.length
            found: dict[str, float] = {}
            for i in range(L):
                pat = w[:i] + "?" + w[i + 1 :]
                for a, sim in self.retr.pattern_candidates(L, pat, k=k_per_pos, qvec=st.qvec):
                    if a != w:
                        found[a] = max(found.get(a, -9.0), sim)
            top = sorted(found.items(), key=lambda kv: -kv[1])[:k_total]
            if w not in st.cands:
                # the current fill came from the crossings; give it a neutral retriever score if it is not a known word
                sim_w = self.retr.sim_for(st.qvec, [w])[0] if self.retr.in_vocab(w) else (top[0][1] if top else 0.0)
                st.cands[w] = Candidate(w, self.retr.text_of(w), retr=sim_w, src={"grid"})
                n_new += 1
            fresh = []
            for a, sim in top:
                if a not in st.cands:
                    st.cands[a] = Candidate(a, self.retr.text_of(a), retr=sim, src={"repair"})
                    n_new += 1
                fresh.append(a)
            alts = [a for a, _ in sorted(st.prior.items(), key=lambda kv: -kv[1])[:5] if a != w and a not in fresh]
            per_slot[si] = [w] + fresh + alts
            for a in per_slot[si]:
                st.judged.discard(a)
            patterns[si] = self._pattern(st, bp, si, 0.95)
            self._set_prior(st)
        return n_new, per_slot, patterns

    # ------------------------------------------------------------------ stage 5: local search
    def _slot_logp(self, st: SlotState, word: str) -> float:
        """log-score of any word in a slot: mixed prior for judged/retrieved candidates, retriever softmax for other
        vocabulary words (scored against the slot's candidate pool), a frequency floor for non-words."""
        p = st.prior.get(word)
        if p is not None:
            return math.log(max(p, 1e-12))
        cache = st.__dict__.setdefault("_wcache", {})
        if word in cache:
            return cache[word]
        if not self.retr.in_vocab(word):
            v = math.log(max(st.p_null, 1e-6)) + self.retr.word_prior(word)
        else:
            pool = st.__dict__.get("_pool")
            if pool is None:
                sc = self._retr_scores(st)
                mx = max(sc.values()) if sc else 0.0
                z = sum(math.exp(s - mx) for s in sc.values())
                pool = st.__dict__["_pool"] = (mx, z)
            mx, z = pool
            sim = self.retr.sim_for(st.qvec, [word])[0]
            s = self.sim_scale * sim
            e = math.exp(s - mx)
            v = math.log(max((1 - st.p_null) * e / (z + e), 1e-12))
        cache[word] = v
        return v

    def _proposals(self, pz: Puzzle, states: list[SlotState], grid: list[list[str]], si: int, slot_score: dict[int, float]) -> list[str]:
        """Candidate replacements for one slot: top prior candidates, Hamming-1 vocabulary neighbours of the current fill,
        and vocabulary words matching the pattern left when weak crossings are wildcarded."""
        st = states[si]
        L = st.slot.length
        cur = "".join(grid[r][c] for r, c in st.slot.cells)

        def pat_search(pat: str, k: int):
            key = (si, pat, k)
            hit = self._pat_cache.get(key)
            if hit is None:
                hit = self._pat_cache[key] = self.retr.pattern_candidates(L, pat, k=k, qvec=st.qvec)
            return hit

        out: dict[str, float] = {}
        for a, p in sorted(st.prior.items(), key=lambda kv: -kv[1])[:12]:
            out[a] = 10 + p
        for i in range(L):
            for a, sim in pat_search(cur[:i] + "?" + cur[i + 1 :], 6):
                out[a] = max(out.get(a, -9), sim)
        for thresh in (-6.0, -3.5):
            pat = []
            for i in range(L):
                cr = pz.crossing(si, i)
                weak = cr is None or slot_score.get(cr[0], -99) < thresh
                pat.append("?" if weak else cur[i])
            pat = "".join(pat)
            if "?" in pat:
                for a, sim in pat_search(pat, 20):
                    out[a] = max(out.get(a, -9), sim)
        out.pop(cur, None)
        return [a for a, _ in sorted(out.items(), key=lambda kv: -kv[1])[:48]]

    def _local_search(self, pz: Puzzle, states: list[SlotState], grid: list[list[str]], deadline: float, max_kicks: int = 40) -> tuple[list[list[str]], int]:
        """Iterated local search on the global log-score: greedy best-improvement sweeps, then kicks (forced moves on the
        weakest slots) when stuck; always returns the best grid seen. Stops when kicks stop helping or at the deadline."""

        def word(g, si):
            return "".join(g[r][c] for r, c in pz.slots[si].cells)

        def slot_scores(g):
            return {si: self._slot_logp(states[si], word(g, si)) for si in range(len(states))}

        def affected_of(si):
            s = {si}
            for i in range(pz.slots[si].length):
                cr = pz.crossing(si, i)
                if cr:
                    s.add(cr[0])
            return s

        def apply(g, si, w):
            for (r, c), ch in zip(pz.slots[si].cells, w):
                g[r][c] = ch

        n_moves = 0
        scores = slot_scores(grid)
        obj = sum(scores.values())
        best_obj, best_grid = obj, [row[:] for row in grid]
        kicks = 0
        tabu: set[tuple[int, str]] = set()
        while time.time() < deadline:
            improved = False
            order = sorted(range(len(states)), key=lambda si: scores[si])
            for si in order:
                if time.time() > deadline:
                    break
                cur = word(grid, si)
                aff = affected_of(si)
                base = sum(scores[j] for j in aff)
                best_gain, best_w, best_sc = 1e-6, None, None
                for w in self._proposals(pz, states, grid, si, scores):
                    if (si, w) in tabu:
                        continue
                    apply(grid, si, w)
                    new_sc = {j: self._slot_logp(states[j], word(grid, j)) for j in aff}
                    gain = sum(new_sc.values()) - base
                    apply(grid, si, cur)
                    if gain > best_gain:
                        best_gain, best_w, best_sc = gain, w, new_sc
                if best_w is not None:
                    apply(grid, si, best_w)
                    scores.update(best_sc)
                    obj += best_gain
                    n_moves += 1
                    improved = True
                    if obj > best_obj + 1e-9:
                        best_obj, best_grid = obj, [row[:] for row in grid]
                        self._snap(f"local search: {n_moves} moves", grid=grid)
            if improved:
                continue
            # converged: stop early when every fill is a well-supported word (no slot below the confidence floor)
            if min(scores.values()) > math.log(0.05):
                break
            # stuck: kick the weakest slot not kicked before to its best proposal even if it scores worse
            if kicks >= max_kicks:
                break
            kicked = False
            for si in sorted(range(len(states)), key=lambda si: scores[si]):
                cur = word(grid, si)
                props = [w for w in self._proposals(pz, states, grid, si, scores) if (si, w) not in tabu]
                if not props:
                    continue
                aff = affected_of(si)
                best_w, best_tot, best_sc = None, -1e18, None
                for w in props[:12]:
                    apply(grid, si, w)
                    new_sc = {j: self._slot_logp(states[j], word(grid, j)) for j in aff}
                    tot = sum(new_sc.values())
                    apply(grid, si, cur)
                    if tot > best_tot:
                        best_w, best_tot, best_sc = w, tot, new_sc
                if best_w is None:
                    continue
                tabu.add((si, cur))
                apply(grid, si, best_w)
                obj += best_tot - sum(scores[j] for j in aff)
                scores.update(best_sc)
                kicks += 1
                kicked = True
                break
            if not kicked:
                break
        if best_obj > obj:
            grid = best_grid
        return grid, n_moves

    # ------------------------------------------------------------------ driver
    def solve(self, pz: Puzzle) -> SolveResult:
        t0 = time.time()
        deadline = t0 + self.budget
        timing: dict[str, float] = {}
        self._t0, self._pz = t0, pz
        self._t_perfect = None
        self._pat_cache: dict = {}
        self._bp_label = "belief propagation 1"
        self._snap("start")

        t = time.time()
        states = self._candidates(pz)
        timing["candidates"] = time.time() - t
        self._snap("candidates retrieved")
        self.log(f"candidates: {sum(len(s.cands) for s in states)} for {len(states)} slots in {timing['candidates']:.2f}s")

        t = time.time()
        per_slot = {si: [a for a, _ in sorted(st.prior.items(), key=lambda kv: -kv[1])[: self.k_judge]] for si, st in enumerate(states)}
        n = self._judge(pz, states, per_slot, None, deadline - 15)
        timing["jev_round1"] = time.time() - t
        if self.jev:
            self.log(f"jev round 1: {n} candidate probabilities in {timing['jev_round1']:.2f}s  ({self.jev.stats})")

        t = time.time()
        bp = self._bp(pz, states)
        grid = self._grid_from_bp(pz, states, bp)
        timing["bp1"] = time.time() - t
        self.log(f"bp1 {timing['bp1']:.2f}s  " + (str(pz.score(grid)) if pz.solution else ""))
        self._snap("belief propagation 1 done", grid=grid)

        rounds = 0
        thresholds = [0.9, 0.8, 0.95]
        while rounds < len(thresholds) and time.time() < deadline - 12:
            rounds += 1
            t = time.time()
            n_new, new_per_slot, patterns = self._expand(pz, states, bp, deadline, thresh=thresholds[rounds - 1])
            n = self._judge(pz, states, new_per_slot, patterns, deadline - 8)
            timing[f"expand{rounds}"] = time.time() - t
            self._snap(f"pattern expansion {rounds}: +{n_new} candidates", grid=grid)
            t = time.time()
            self._bp_label = f"belief propagation {rounds + 1}"
            bp = self._bp(pz, states)
            grid = self._grid_from_bp(pz, states, bp)
            timing[f"bp{rounds + 1}"] = time.time() - t
            self.log(f"expand{rounds}: +{n_new} cands, {len(new_per_slot)} slots re-judged ({n} probs) in {timing[f'expand{rounds}']:.2f}s; bp {timing[f'bp{rounds + 1}']:.2f}s  " + (str(pz.score(grid)) if pz.solution else ""))
            self._snap(f"belief propagation {rounds + 1} done", grid=grid)

        t = time.time()
        grid, moves = self._local_search(pz, states, grid, deadline - 1)
        timing["local_search"] = time.time() - t
        self.log(f"local search: {moves} moves in {timing['local_search']:.2f}s  " + (str(pz.score(grid)) if pz.solution else ""))
        self._snap(f"local search: {moves} moves", grid=grid)

        # repair rounds: propose vocabulary neighbours (Hamming distance 1) of every unsettled fill,
        # let Jev judge them against the current fill, then BP + local search again
        repairs = 0
        while time.time() < deadline - 6 and repairs < (2 if self.jev is not None else 1):
            repairs += 1
            t = time.time()
            n_new, per_slot, patterns = self._repair(pz, states, grid, bp)
            n = self._judge(pz, states, per_slot, patterns, deadline - 3)
            if n_new or n:
                self._bp_label = f"repair {repairs}: belief propagation"
                bp = self._bp(pz, states)
                grid = self._grid_from_bp(pz, states, bp)
                grid, moves2 = self._local_search(pz, states, grid, deadline - 0.5)
                moves += moves2
                self._snap(f"repair {repairs} done", grid=grid)
            timing[f"repair{repairs}"] = time.time() - t
            self.log(f"repair{repairs}: {len(per_slot)} slots, +{n_new} cands, {n} probs in {timing[f'repair{repairs}']:.2f}s  " + (str(pz.score(grid)) if pz.solution else ""))
            if n_new == 0 and n == 0:
                break

        self._snap("final answer", grid=grid)
        elapsed = time.time() - t0
        t_perfect = self._t_perfect if pz.score(grid).get("perfect") else None  # only meaningful if the returned grid is correct
        slots_out = []
        for si, st in enumerate(states):
            w = "".join(grid[r][c] for r, c in st.slot.cells)
            slots_out.append({"slot": st.slot.name, "clue": st.slot.clue, "answer": w, "truth": pz.answer(st.slot), "p": round(st.prior.get(w, 0.0), 4), "in_cands": w in st.cands})
        return SolveResult(grid=grid, elapsed=elapsed, timing=timing, accuracy=pz.score(grid), slots=slots_out, jev_stats=dict(self.jev.stats) if self.jev else {}, t_perfect=t_perfect)
