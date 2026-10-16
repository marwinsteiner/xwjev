# xwjev: a crossword solver built on Jev

![The solver filling the NYT crossword of Thursday 2026-10-15, sweep by sweep](docs/solve.gif)

The NYT daily crossword of Thursday 2026-10-15, solved with every letter correct 3.0 s
into the solve on a laptop CPU (retriever-only run; model loading adds 3.4 s). Each frame is
one belief-propagation sweep or one accepted local-search move. Pale cells are uncertain.

Solves American-style crosswords (NYT daily format: `.puz`, NYT v6 JSON, xwordinfo JSON,
`.xd`, `.ipuz`) in well under 49 seconds. The pipeline:

1. **Candidate generation** (local, ~0.3 s). For every clue:
   exact and loose lookup of the clue in a 7.2M-pair clue database, plus dense retrieval
   with a bi-encoder fine-tuned on clue -> answer pairs over the 493k-answer vocabulary,
   restricted to the slot length.
2. **Jev judging** (TypeSafe AI System One model, ~1-3 s). One `Choice` question per slot:
   "which candidate answers this clue?" with the top candidates as options plus
   `NONE_OF_THESE`. Jev's calibrated probabilities become the unary potentials; its NONE mass
   becomes the slot's "answer missing" probability. Twelve questions are packed per request
   and requests run concurrently.
3. **Loopy belief propagation** over the grid (~0.05 s): slots exchange letter distributions
   through crossing cells; an explicit unknown-word option stops BP from forcing a wrong
   candidate when the right answer is not in the list.
4. **Pattern expansion** (2 rounds): for unsettled slots the confident crossing letters form a
   pattern (`S?A?E`), matching vocabulary words are ranked by the retriever and re-judged by
   Jev with the pattern in the question, then BP runs again.
5. **Iterated local search** on the global log-score: proposals for each entry are its top
   candidates, vocabulary words at Hamming distance 1 from the current fill, and vocabulary
   words matching the pattern left when weak crossings are wildcarded; best-improvement
   sweeps, then kicks on the weakest entries when stuck, always keeping the best grid.
6. **Repair** (2 rounds): for every fill Jev has not vouched for, the vocabulary words at
   Hamming distance 1 from the current fill (plus the best alternatives) are judged against it;
   BP and local search run again. This fixes the typical single-letter crossing error
   (BEEAST -> BREAST) that pure retrieval never proposes.

Every stage after the first BP pass checks the wall-clock deadline (`--budget`, default 45 s),
so a complete grid is always returned in time.

## Data (all freely available)

* xd clue corpus: https://xd.saul.pw/xd-clues.zip (10M clue/answer usages, 1942-2026)
* CrosswordQA (Berkeley Crossword Solver): https://huggingface.co/datasets/albertxu/CrosswordQA (6.8M pairs)
* Base encoder: `BAAI/bge-small-en-v1.5` (33M parameters), fine-tuned with in-batch negatives

Held-out test puzzles are downloaded with `xword-dl` from dates after the corpus snapshot.

## Setup

```
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
export TYPESAFE_API_KEY=sk-...      # from https://console.typesafe.ai/settings/keys
#   or OPENROUTER_API_KEY=...       # routes to typesafe/jev-1.14 via OpenRouter's Decisions API
```

Without a key the solver runs retriever-only (`--no-jev` forces that mode).

## Build assets and train (GPU box)

```
xwjev build-data --xd-tsv data/xd/clues.tsv --cqa-csv data/CrosswordQA/train.csv data/CrosswordQA/valid.csv --out assets
xwjev train --assets assets --out runs/bge_small_full --epochs 1 --batch-size 256
xwjev embed --assets assets --model-dir runs/bge_small_full/model      # writes assets/answer_emb.npy
cp -r runs/bge_small_full/model assets/model
xwjev export-onnx --model-dir assets/model                              # model.onnx for the fast CPU path
xwjev eval-retriever --assets assets --model-dir assets/model            # recall@k on held-out pairs
```

`scripts/deploy_lynx.sh` syncs the tree to the training box over ssh.

## Solve

```
xwjev solve --assets assets puzzle.puz                 # prints the grid, accuracy (if the file has a solution) and timing
xwjev eval  --assets assets puzzles/                   # batch evaluation
xword-dl nyt -u $NYT_USER -p $NYT_PASS -o today.puz    # fetch today's NYT with a subscription
```

Assets needed at solve time: `answers.parquet`, `vocab.npz`, `exact.sqlite`, `answer_emb.npy`, `model/`.
With `model/model.onnx` present the solver runs on onnxruntime + numpy and never imports torch
(process start about 2 s instead of 15-20 s). `XWJEV_DEVICE=cuda` selects the torch/GPU path.

## Results (2026-10-15)

The benchmark is the time until every letter is correct. The solver never sees the answers;
they are only used to timestamp the moment the grid became fully correct (`t_perfect` in the
result). Because the solver cannot know it is done, it keeps searching until its score
converges, so the stop time is later than the all-correct time.

NYT daily crossword, Thursday 2026-10-15 (McKay Fritz / Will Shortz, FIRE BLANKS theme),
retriever only, laptop CPU, ONNX path:

| event                              | time    |
|------------------------------------|---------|
| model and index loaded             | 3.4 s before t = 0 |
| candidates retrieved               | 1.6 s   |
| belief propagation converged       | 2.8 s (179 / 187 letters) |
| all 187 letters correct            | 3.0 s   |
| solver stopped (score converged)   | 5.7 s   |

Held-out set: 8 puzzles published after the clue-corpus snapshot (5 USA Today, 3 New Yorker,
2026-10-11 to 2026-10-15), so none of their clue/answer pairs could be in the training data.
Bi-encoder: bge-small fine-tuned one epoch on 12.2M (clue, answer) samples (89 min on an RTX 4060).
Vocabulary: 493k corpus answers plus 171k entries from the Collaborative Word List (MIT).

| configuration (retriever only, no Jev) | letters | perfect | all-correct at        | stop time |
|----------------------------------------|---------|---------|-----------------------|-----------|
| iterated local search, extended vocab  | 99.74%  | 5 / 8   | 1.9 to 2.3 s          | 4.0 to 6.1 s |
| earlier greedy search, corpus vocab    | 99.87%  | 6 / 8   | about 0.5 s           | 0.6 s     |
| simulated judge (50%, 80%, oracle)     | 100%    | 8 / 8   | about 1 s             | ~1 s      |

The two held-out puzzles that stay at 1 to 2 wrong letters are scored higher wrong than right
by the retriever alone (the correct grid has the lower objective), so more search cannot fix
them; that is the judge's job.

Retriever alone (length-filtered, no exact lookup) on 3,000 held-out clue/answer pairs from
2025-2026: recall@1 24.5%, recall@20 60.5%, recall@100 75.3%. The exact-clue database covers
41.8% of NYT 2026 clue/answer pairs verbatim.

Timing on an 8-core laptop (ONNX path): 9.7 s from process start to a printed grid for a
15x15 puzzle, of which about 2 s is the solve itself. Jev adds 5-6 rounds of a few concurrent
requests at 0.1-0.5 s each. Budget enforcement defaults to 45 s. The very first run after
copying the 1.5 GB of assets is slower (cold file cache, antivirus scan).

The live Jev path has not been exercised yet (no API key was available while building this);
`scripts/jev_smoke.py` validates it in one call, and `scripts/eval_fake_jev.py` runs the
same code path with a simulated judge. Not handled: rebus squares, and answers absent from
the 493k-entry vocabulary (0.7% of held-out answers, usually long themed phrases) can only
be recovered through their crossings.

## What is not in this repository

Puzzles (NYT, USA Today, New Yorker) are copyrighted by their publishers and are not
included. The clue corpora, the word list, the trained model and the built assets (about
1.8 GB) are not included either; the commands above rebuild them from the public sources.
The Collaborative Word List is at
https://github.com/Crossword-Nexus/collaborative-word-list (`xwjev extend-vocab`).

## License

Copyright (c) 2026 Marwin Steiner. All rights reserved. The source is published for viewing
only; see `LICENSE`. No permission is granted to use, copy, modify or redistribute it.

## Layout

```
src/xwjev/puzzle.py          puzzle model, parsers, scoring
src/xwjev/data.py            corpus build (pairs, answers, exact.sqlite, vocab.npz)
src/xwjev/train_biencoder.py bi-encoder fine-tuning, answer embedding, retriever eval
src/xwjev/retriever.py       exact lookup, dense retrieval, pattern search
src/xwjev/jev.py             Jev client (TypeSafe direct or OpenRouter), batching, caching
src/xwjev/bp.py              loopy belief propagation
src/xwjev/solver.py          orchestration with the time budget
src/xwjev/cli.py             command line
```
