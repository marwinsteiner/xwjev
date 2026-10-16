"""Loopy belief propagation over the crossword grid.

Variables : one per slot, domain = its candidate answers + an implicit "unknown word" option
Factors   : crossing cells (letters must agree)
Messages  : slot -> cell letter distributions (26 letters). Each slot's outgoing message at
            position i marginalises its candidates weighted by prior * incoming messages at the
            other positions. The unknown-word option contributes a smooth letter distribution
            with mass p_null, which keeps BP from forcing a wrong candidate when the right answer
            is missing from the list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A = 26
# English letter frequencies in crossword grids (rough, used for the unknown-word option)
_LETTER_PRIOR = np.array([8.6, 1.6, 3.0, 3.8, 11.8, 1.6, 2.2, 3.4, 7.3, 0.3, 1.0, 4.4, 2.8, 6.7, 7.6, 2.4, 0.2, 6.6, 7.1, 7.2, 3.1, 1.1, 1.3, 0.3, 2.3, 0.5])
LETTER_PRIOR = _LETTER_PRIOR / _LETTER_PRIOR.sum()


@dataclass
class SlotVar:
    idx: int
    length: int
    cands: np.ndarray  # (K, L) int8 letters 0..25
    logp: np.ndarray  # (K,) log prior probability of each candidate (sums to <= 1 with p_null)
    p_null: float  # probability that the answer is not among cands
    cross: list[tuple[int, int] | None]  # per position: (other slot idx, other pos) or None


@dataclass
class BPResult:
    slot_belief: list[np.ndarray]  # per slot (K,) posterior over candidates (excl. null)
    slot_null: list[float]  # per slot posterior mass on "unknown"
    cell_marg: dict[tuple[int, int], np.ndarray]  # (slot, pos) -> (26,) letter marginal at that cell
    slot_msgs: list[np.ndarray]  # per slot (L, 26) outgoing messages


def run_bp(slots: list[SlotVar], n_iter: int = 12, damping: float = 0.4, eps: float = 1e-4, seed: int = 0, on_iter=None) -> BPResult:
    """`on_iter(k, cell_marg)` is called after every sweep with the current cell marginals (for tracing)."""
    rng = np.random.default_rng(seed)
    n = len(slots)
    onehots = []
    for s in slots:
        oh = np.zeros((s.cands.shape[0], s.length, A), dtype=np.float32)
        if s.cands.shape[0]:
            k_idx = np.arange(s.cands.shape[0])[:, None]
            l_idx = np.arange(s.length)[None, :]
            oh[k_idx, l_idx, s.cands] = 1.0
        onehots.append(oh)
    msgs = [np.tile(LETTER_PRIOR, (s.length, 1)).astype(np.float32) for s in slots]
    uniform = np.tile(LETTER_PRIOR, (1, 1)).astype(np.float32)

    def incoming(s: SlotVar) -> np.ndarray:
        inc = np.empty((s.length, A), dtype=np.float32)
        for i, cr in enumerate(s.cross):
            inc[i] = msgs[cr[0]][cr[1]] if cr is not None else uniform[0]
        return inc

    for _ in range(n_iter):
        order = rng.permutation(n)
        for si in order:
            s = slots[si]
            K = s.cands.shape[0]
            inc = incoming(s)
            log_inc = np.log(inc + eps)
            new = np.empty((s.length, A), dtype=np.float32)
            if K:
                G = log_inc[np.arange(s.length)[None, :], s.cands]  # (K, L)
                total = s.logp + G.sum(1)  # (K,)
                tG = total[:, None] - G  # (K, L) evidence for candidate k excluding position i
                mx = tG.max(0)
                w = np.exp(tG - mx)
                m = np.einsum("kl,klc->lc", w, onehots[si])
                m = m / np.maximum(m.sum(1, keepdims=True), 1e-30)
                # per-position evidence: candidates vs the unknown-word option (both excluding position i)
                cand_i = np.log(w.sum(0) + 1e-30) + mx  # (L,)
                null_pos = np.log((inc * LETTER_PRIOR).sum(1) + eps)  # (L,)
                null_i = np.log(s.p_null + 1e-12) + null_pos.sum() - null_pos
                pn = 1.0 / (1.0 + np.exp(np.clip(cand_i - null_i, -30, 30)))  # (L,) posterior null share
                new = (1 - pn)[:, None] * m + pn[:, None] * LETTER_PRIOR[None, :]
            else:
                new[:] = LETTER_PRIOR[None, :]
            msgs[si] = damping * msgs[si] + (1 - damping) * new
        if on_iter is not None:
            snap = {}
            for si, s in enumerate(slots):
                inc = incoming(s)
                for i in range(s.length):
                    cm = msgs[si][i] * inc[i]
                    snap[(si, i)] = cm / max(cm.sum(), 1e-30)
            on_iter(_ + 1, snap)

    beliefs, nulls, cell_marg = [], [], {}
    for si, s in enumerate(slots):
        inc = incoming(s)
        log_inc = np.log(inc + eps)
        K = s.cands.shape[0]
        if K:
            G = log_inc[np.arange(s.length)[None, :], s.cands]
            total = s.logp + G.sum(1)
            null_ev = np.log(s.p_null + 1e-12) + np.log((inc * LETTER_PRIOR).sum(1) + eps).sum()
            mx = max(total.max(), null_ev)
            z = np.exp(total - mx).sum() + np.exp(null_ev - mx)
            b = np.exp(total - mx) / z
            nulls.append(float(np.exp(null_ev - mx) / z))
        else:
            b = np.zeros(0)
            nulls.append(1.0)
        beliefs.append(b)
        for i in range(s.length):
            cm = msgs[si][i] * inc[i]
            cell_marg[(si, i)] = cm / max(cm.sum(), 1e-30)
    return BPResult(beliefs, nulls, cell_marg, msgs)
