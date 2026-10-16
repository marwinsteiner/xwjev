"""Build the clue-answer corpus and the solve-time assets.

Inputs (freely available):
  * xd clues corpus        https://xd.saul.pw/xd-clues.zip   -> xd/clues.tsv  (pubid, year, answer, clue)
  * CrosswordQA (Berkeley) https://huggingface.co/datasets/albertxu/CrosswordQA -> train.csv, valid.csv

Outputs in --out (the "assets" dir):
  pairs.parquet    one row per unique (clue_norm, answer): clue, clue_norm, ans, text, n, year
  answers.parquet  one row per unique answer: ans, text, freq, nclues, len
  exact.sqlite     exact / loose clue lookup tables
  vocab.npz        padded uint8 letter matrix for fast pattern search
  heldout.parquet  pairs reserved for retriever evaluation (never trained on)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .text import clue_key_loose, norm_answer, norm_clue, segment_answer

MAX_LEN = 25


def _load_xd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=3, keep_default_na=False, on_bad_lines="skip")
    df = df.rename(columns={"answer": "answer_raw"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["text"] = ""
    return df[["pubid", "year", "answer_raw", "clue", "text"]]


def _load_cqa(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        frames.append(d[["clue", "answer"]])
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"answer": "text"})
    df["answer_raw"] = df["text"]
    df["pubid"] = "cqa"
    df["year"] = 0
    return df[["pubid", "year", "answer_raw", "clue", "text"]]


def build(xd_tsv: Path, cqa_csvs: list[Path], out: Path, heldout_frac: float = 0.002, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print("loading xd ...")
    xd = _load_xd(xd_tsv)
    print(f"  xd rows: {len(xd):,}")
    print("loading CrosswordQA ...")
    cqa = _load_cqa(cqa_csvs)
    print(f"  cqa rows: {len(cqa):,}")
    df = pd.concat([xd, cqa], ignore_index=True)
    del xd, cqa

    print("normalising ...")
    df["clue"] = df["clue"].str.strip()
    df["ans"] = [norm_answer(a) for a in tqdm(df["answer_raw"], mininterval=5)]
    df["clue_norm"] = [norm_clue(c) for c in tqdm(df["clue"], mininterval=5)]
    ok = (df["ans"].str.len() >= 1) & (df["ans"].str.len() <= MAX_LEN) & (df["clue_norm"].str.len() >= 1)
    df = df[ok]
    print(f"  kept rows: {len(df):,}")

    # segmented display text: prefer CrosswordQA's segmentation, else wordninja
    print("segmenting answers ...")
    seg = df[df["text"] != ""].copy()
    seg["text"] = seg["text"].str.lower().str.strip()
    seg_map = seg.groupby("ans")["text"].agg(lambda s: s.value_counts().index[0]).to_dict()
    del seg
    all_ans = df["ans"].unique()
    missing = [a for a in all_ans if a not in seg_map]
    print(f"  unique answers: {len(all_ans):,}; segmenting {len(missing):,} with wordninja")
    for a in tqdm(missing, mininterval=5):
        seg_map[a] = segment_answer(a)
    df["text"] = df["ans"].map(seg_map)

    print("deduplicating pairs ...")
    g = df.groupby(["clue_norm", "ans"], sort=False)
    pairs = g.agg(clue=("clue", "first"), text=("text", "first"), n=("clue", "size"), year=("year", "max"), pubid=("pubid", "first")).reset_index()
    print(f"  unique pairs: {len(pairs):,}")

    # held-out pairs for retriever evaluation: recent pairs whose answer also occurs elsewhere
    rng = np.random.default_rng(seed)
    recent = pairs.index[(pairs["year"] >= 2025)].to_numpy()
    k = int(len(pairs) * heldout_frac)
    held_idx = rng.choice(recent, size=min(k, len(recent)), replace=False)
    held = pairs.loc[held_idx]
    pairs = pairs.drop(index=held_idx).reset_index(drop=True)
    held.to_parquet(out / "heldout.parquet", index=False)
    print(f"  held out {len(held):,} pairs for retriever eval")

    pairs.to_parquet(out / "pairs.parquet", index=False)

    print("answers table ...")
    ans = pairs.groupby("ans", sort=False).agg(text=("text", "first"), freq=("n", "sum"), nclues=("n", "size"), year=("year", "max")).reset_index()
    ans["len"] = ans["ans"].str.len()
    ans = ans.sort_values(["len", "freq"], ascending=[True, False]).reset_index(drop=True)
    ans.to_parquet(out / "answers.parquet", index=False)
    print(f"  answers: {len(ans):,}")

    print("vocab.npz ...")
    mat = np.zeros((len(ans), MAX_LEN), dtype=np.uint8)
    for i, a in enumerate(ans["ans"].to_numpy()):
        b = np.frombuffer(a.encode("ascii"), dtype=np.uint8) - 65
        mat[i, : len(b)] = b + 1  # 1..26, 0 = pad
    np.savez(out / "vocab.npz", letters=mat, length=ans["len"].to_numpy().astype(np.int16), freq=ans["freq"].to_numpy().astype(np.int64))

    print("exact.sqlite ...")
    db = out / "exact.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE exact (clue_norm TEXT, ans TEXT, n INTEGER, year INTEGER)")
    con.executemany("INSERT INTO exact VALUES (?,?,?,?)", zip(pairs["clue_norm"], pairs["ans"], pairs["n"].astype(int), pairs["year"].astype(int)))
    con.execute("CREATE INDEX ix_exact ON exact(clue_norm)")
    loose = pairs.assign(key=[clue_key_loose(c) for c in tqdm(pairs["clue_norm"], mininterval=5)])
    loose = loose.groupby(["key", "ans"], sort=False)["n"].sum().reset_index()
    con.execute("CREATE TABLE loose (key TEXT, ans TEXT, n INTEGER)")
    con.executemany("INSERT INTO loose VALUES (?,?,?)", zip(loose["key"], loose["ans"], loose["n"].astype(int)))
    con.execute("CREATE INDEX ix_loose ON loose(key)")
    con.commit()
    con.close()
    print("done:", out)


def extend_vocab(assets: Path, wordlist: Path, min_score: int = 25, model_dir: Path | None = None) -> None:
    """Add entries of a `WORD;score` crossword word list (e.g. the MIT-licensed Collaborative Word List) to the
    answer vocabulary. New entries get a minimal frequency prior and are embedded with the fine-tuned encoder, so
    the solver can score them against a clue like any other vocabulary word. Rewrites answers.parquet, vocab.npz
    and answer_emb.npy (rows re-sorted by length); exact.sqlite is untouched."""
    assets = Path(assets)
    ans = pd.read_parquet(assets / "answers.parquet")
    emb = np.load(assets / "answer_emb.npy")
    assert len(ans) == len(emb)
    known = set(ans["ans"])
    new: dict[str, int] = {}
    with open(wordlist, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ";" not in line:
                continue
            w, s = line.rstrip("\n").rsplit(";", 1)
            try:
                score = int(s)
            except ValueError:
                continue
            if score < min_score or any(ch.isdigit() for ch in w):
                continue
            a = norm_answer(w)
            if 3 <= len(a) <= MAX_LEN and a not in known and a not in new:
                new[a] = score
    print(f"wordlist: {len(new):,} new entries (score >= {min_score}) on top of {len(ans):,}")
    if not new:
        return
    words = list(new)
    texts = [segment_answer(a) for a in tqdm(words, mininterval=5, desc="segmenting")]
    md = Path(model_dir) if model_dir else assets / "model"
    try:
        from .encoder import OnnxEncoder

        enc = OnnxEncoder(md)
        vecs = np.concatenate([enc.encode(texts[i : i + 2048]) for i in tqdm(range(0, len(texts), 2048), desc="embedding")])
    except Exception as e:  # noqa: BLE001
        print(f"onnx encoder unavailable ({e}); using sentence-transformers")
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(str(md))
        m.max_seq_length = 32
        vecs = m.encode(texts, batch_size=1024, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    add = pd.DataFrame({"ans": words, "text": texts, "freq": 1, "nclues": 0, "year": 0, "len": [len(a) for a in words]})
    ans = ans.assign(_emb=range(len(ans)))
    add = add.assign(_emb=range(len(ans), len(ans) + len(add)))
    allv = pd.concat([ans, add], ignore_index=True).sort_values(["len", "freq"], ascending=[True, False]).reset_index(drop=True)
    full = np.concatenate([emb.astype(np.float16), vecs.astype(np.float16)])[allv["_emb"].to_numpy()]
    allv = allv.drop(columns=["_emb"])
    allv.to_parquet(assets / "answers.parquet", index=False)
    np.save(assets / "answer_emb.npy", full)
    mat = np.zeros((len(allv), MAX_LEN), dtype=np.uint8)
    for i, a in enumerate(allv["ans"].to_numpy()):
        b = np.frombuffer(a.encode("ascii"), dtype=np.uint8) - 65
        mat[i, : len(b)] = b + 1
    np.savez(assets / "vocab.npz", letters=mat, length=allv["len"].to_numpy().astype(np.int16), freq=allv["freq"].to_numpy().astype(np.int64))
    print(f"vocabulary now {len(allv):,} entries")
