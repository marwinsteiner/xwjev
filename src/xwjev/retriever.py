"""Candidate generation: exact clue lookup, bi-encoder retrieval, and pattern search over the answer vocabulary.

Two execution modes:
  * numpy + onnxruntime (default on CPU): sub-second startup, ~0.3 s per puzzle for retrieval
  * torch (XWJEV_DEVICE=cuda, or when model.onnx is missing): same maths on the GPU / via sentence-transformers
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .text import clue_key_loose, norm_clue, query_text


@dataclass
class Candidate:
    ans: str
    text: str
    retr: float = 0.0  # retriever cosine similarity (0 if not from retriever)
    exact: int = 0  # exact-clue match count in corpus
    jev: float | None = None  # Jev probability (None if not judged)
    src: set[str] = field(default_factory=set)


class Retriever:
    def __init__(self, assets: Path, model_dir: Path | None = None, device: str | None = None):
        assets = Path(assets)
        t0 = time.time()
        self.assets = assets
        self.answers = pd.read_parquet(assets / "answers.parquet", columns=["ans", "text", "freq", "len"])
        self.ans_arr = self.answers["ans"].to_numpy()
        self.text_arr = self.answers["text"].to_numpy()
        self.freq = self.answers["freq"].to_numpy().astype(np.float64)
        self.len_arr = self.answers["len"].to_numpy()
        self.ans_to_row = {a: i for i, a in enumerate(self.ans_arr)}
        # answers.parquet is sorted by length, so every length is one contiguous row range
        self.by_len: dict[int, np.ndarray] = {}
        self.range_by_len: dict[int, tuple[int, int]] = {}
        for ln in np.unique(self.len_arr):
            idx = np.flatnonzero(self.len_arr == ln)
            self.by_len[int(ln)] = idx
            self.range_by_len[int(ln)] = (int(idx[0]), int(idx[-1]) + 1)
            assert idx[-1] - idx[0] + 1 == len(idx), "answers.parquet must be sorted by length"
        voc = np.load(assets / "vocab.npz")
        self.letters = voc["letters"]  # (N, 25) uint8, 1..26, 0 pad
        self.log_freq = np.log1p(self.freq)

        md = Path(model_dir) if model_dir else assets / "model"
        emb_path = assets / "answer_emb.npy"
        want = device or os.environ.get("XWJEV_DEVICE") or "auto"
        self.mode = None
        self.model = None
        self.emb = None
        if md.exists() and emb_path.exists():
            use_onnx = (md / "model.onnx").exists() and want in ("auto", "cpu")
            if use_onnx:
                try:
                    from .encoder import OnnxEncoder

                    self.model = OnnxEncoder(md)
                    self.mode = "numpy"
                except Exception as e:  # noqa: BLE001 - fall back to torch below
                    print(f"[xwjev] onnx encoder unavailable ({e}); falling back to torch")
            if self.mode is None:
                import torch
                from sentence_transformers import SentenceTransformer

                self.device = want if want in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
                self.model = SentenceTransformer(str(md), device=self.device)
                self.model.max_seq_length = 48
                self.mode = "torch"
            raw = np.load(emb_path)
            if self.mode == "numpy":
                self.emb = raw.astype(np.float32)
            else:
                import torch

                self.emb = torch.from_numpy(raw).to(self.device)
                if self.device == "cpu":
                    self.emb = self.emb.float()
        self.con = sqlite3.connect(f"file:{assets / 'exact.sqlite'}?mode=ro", uri=True, check_same_thread=False)
        self.load_time = time.time() - t0

    # ---------------- exact lookup
    def exact(self, clue: str, length: int) -> dict[str, int]:
        out: dict[str, int] = {}
        key = norm_clue(clue)
        for ans, n in self.con.execute("SELECT ans, n FROM exact WHERE clue_norm=?", (key,)):
            if len(ans) == length:
                out[ans] = out.get(ans, 0) + int(n)
        if not out:
            lk = clue_key_loose(clue)
            for ans, n in self.con.execute("SELECT ans, n FROM loose WHERE key=?", (lk,)):
                if len(ans) == length:
                    out[ans] = out.get(ans, 0) + int(n)
        return out

    # ---------------- dense retrieval
    def encode(self, texts: list[str]):
        if self.mode == "numpy":
            return self.model.encode(texts)
        import torch

        with torch.inference_mode():
            v = self.model.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        return v.to(self.emb.dtype)

    def encode_clues(self, clues: list[str], lengths: list[int]):
        return self.encode([query_text(c, ln) for c, ln in zip(clues, lengths)])

    def _slice(self, length: int):
        a, b = self.range_by_len[length]
        return self.emb[a:b], a

    def search_vec(self, qvec, length: int, k: int = 200, pattern: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Top-k answers of exactly `length` letters by cosine similarity. `pattern`: optional mask over rows of that length."""
        if length not in self.range_by_len:
            return []
        E, base = self._slice(length)
        rows = None
        if pattern is not None:
            rows = np.flatnonzero(pattern)
            if len(rows) == 0:
                return []
        if self.mode == "numpy":
            if rows is not None:
                E = E[rows]
            s = E @ qvec
            k = min(k, s.shape[0])
            if k <= 0:
                return []
            part = np.argpartition(-s, k - 1)[:k]
            order = part[np.argsort(-s[part])]
            vals = s[order]
            pos = order
        else:
            import torch

            if rows is not None:
                E = E[torch.from_numpy(rows).to(E.device)]
            with torch.inference_mode():
                s = E @ qvec
                k = min(k, s.shape[0])
                vals_t, pos_t = torch.topk(s, k)
            pos = pos_t.cpu().numpy()
            vals = vals_t.float().cpu().numpy()
        if rows is not None:
            pos = rows[pos]
        return [(self.ans_arr[base + p], float(v)) for p, v in zip(pos, vals)]

    def sim_for(self, qvec, answers: list[str]) -> list[float]:
        rows = [self.ans_to_row.get(a, -1) for a in answers]
        valid = [r for r in rows if r >= 0]
        sims = np.zeros(0)
        if valid:
            if self.mode == "numpy":
                sims = self.emb[valid] @ qvec
            else:
                import torch

                sims = (self.emb[torch.tensor(valid, device=self.emb.device)] @ qvec).float().cpu().numpy()
        out, j = [], 0
        for r in rows:
            if r >= 0:
                out.append(float(sims[j]))
                j += 1
            else:
                out.append(0.0)
        return out

    # ---------------- pattern search
    def pattern_mask(self, length: int, pattern: str) -> np.ndarray:
        """Boolean mask over rows of `length` matching `pattern` ('?' = wildcard)."""
        idx = self.by_len.get(length)
        if idx is None:
            return np.zeros(0, dtype=bool)
        L = self.letters[idx, :length]
        mask = np.ones(len(idx), dtype=bool)
        for i, ch in enumerate(pattern[:length]):
            if ch != "?":
                mask &= L[:, i] == (ord(ch) - 64)
        return mask

    def pattern_candidates(self, length: int, pattern: str, k: int = 200, qvec=None) -> list[tuple[str, float]]:
        """Vocabulary words matching the pattern, ranked by retriever similarity (if qvec) else frequency."""
        mask = self.pattern_mask(length, pattern)
        if qvec is not None and self.emb is not None:
            return self.search_vec(qvec, length, k=k, pattern=mask)
        idx = self.by_len[length][mask]
        order = np.argsort(-self.freq[idx])[:k]
        return [(self.ans_arr[idx[o]], float(self.log_freq[idx[o]])) for o in order]

    def text_of(self, ans: str) -> str:
        r = self.ans_to_row.get(ans)
        return self.text_arr[r] if r is not None else ans.lower()

    def in_vocab(self, ans: str) -> bool:
        return ans in self.ans_to_row

    def word_prior(self, ans: str) -> float:
        """log-prior for a word independent of the clue (frequency-based, floor for OOV)."""
        r = self.ans_to_row.get(ans)
        if r is None:
            return -12.0
        return float(self.log_freq[r]) - 12.0


_WILD = re.compile(r"[^A-Z?]")


def clean_pattern(p: str) -> str:
    return _WILD.sub("?", p.upper())
