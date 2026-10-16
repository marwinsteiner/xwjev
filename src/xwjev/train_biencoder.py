"""Fine-tune a sentence-transformers bi-encoder on (clue -> answer) pairs.

Query side : "clue (length)"            (see text.query_text)
Answer side: segmented answer text      (e.g. "buzz lightyear")
Loss       : MultipleNegativesRankingLoss with in-batch negatives, no duplicate positives per batch.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .text import query_text


def make_pairs(pairs_parquet: Path, max_pairs: int | None, seed: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(pairs_parquet, columns=["clue", "ans", "text", "n"])
    df = df[df["clue"].str.len() <= 200]
    # frequency-aware sampling: repeat common pairs at most 3x so the model sees crosswordese often
    rep = np.minimum(df["n"].to_numpy(), 3)
    df = df.loc[np.repeat(df.index.to_numpy(), rep)]
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pd.DataFrame({"anchor": [query_text(c, len(a)) for c, a in zip(df["clue"], df["ans"])], "positive": df["text"].to_numpy()})


def train(
    pairs_parquet: Path,
    out_dir: Path,
    base_model: str = "BAAI/bge-small-en-v1.5",
    max_pairs: int | None = None,
    epochs: float = 1.0,
    batch_size: int = 256,
    lr: float = 5e-5,
    max_seq_length: int = 48,
    warmup_ratio: float = 0.02,
    fp16: bool = True,
) -> None:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    t0 = time.time()
    df = make_pairs(pairs_parquet, max_pairs)
    print(f"training pairs: {len(df):,}  ({time.time() - t0:.0f}s to build)")
    ds = Dataset.from_pandas(df, preserve_index=False)

    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length
    loss = MultipleNegativesRankingLoss(model, scale=20.0)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        fp16=fp16 and torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=200,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to=[],
        lr_scheduler_type="linear",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    model.save(str(out_dir / "model"))
    print(f"saved to {out_dir / 'model'}  total {time.time() - t0:.0f}s")


def embed_answers(model_dir: Path, answers_parquet: Path, out_path: Path, batch_size: int = 1024) -> None:
    """Encode every answer's segmented text; save float16 (N, d) matrix aligned with answers.parquet rows."""
    from sentence_transformers import SentenceTransformer

    ans = pd.read_parquet(answers_parquet, columns=["text"])
    model = SentenceTransformer(str(model_dir))
    model.max_seq_length = 32
    t0 = time.time()
    emb = model.encode(ans["text"].tolist(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    np.save(out_path, emb.astype(np.float16))
    print(f"embedded {len(ans):,} answers -> {out_path}  ({time.time() - t0:.0f}s)")


def evaluate(model_dir: Path, assets: Path, n: int = 5000, ks=(1, 5, 20, 50, 100)) -> dict:
    """Length-filtered retrieval recall@k on held-out pairs."""
    from .retriever import Retriever

    held = pd.read_parquet(assets / "heldout.parquet").sample(n=n, random_state=0)
    r = Retriever(assets, model_dir=model_dir)
    hits = {k: 0 for k in ks}
    clues = [query_text(c, len(a)) for c, a in zip(held["clue"], held["ans"])]
    q = r.encode(clues)
    for i, (a, ln) in enumerate(zip(held["ans"], held["ans"].str.len())):
        cands = r.search_vec(q[i], int(ln), k=max(ks))
        ranks = [j for j, (ca, _) in enumerate(cands) if ca == a]
        if ranks:
            for k in ks:
                if ranks[0] < k:
                    hits[k] += 1
    res = {f"recall@{k}": hits[k] / len(held) for k in ks}
    print(res)
    return res
