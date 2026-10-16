"""Command line interface: xwjev <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _p(s: str) -> Path:
    return Path(s).expanduser()


def cmd_build_data(a):
    from .data import build

    build(_p(a.xd_tsv), [_p(x) for x in a.cqa_csv], _p(a.out))


def cmd_extend_vocab(a):
    from .data import extend_vocab

    extend_vocab(_p(a.assets), _p(a.wordlist), min_score=a.min_score)


def cmd_train(a):
    from .train_biencoder import train

    train(_p(a.assets) / "pairs.parquet", _p(a.out), base_model=a.base_model, max_pairs=a.max_pairs, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr)


def cmd_embed(a):
    from .train_biencoder import embed_answers

    assets = _p(a.assets)
    embed_answers(_p(a.model_dir), assets / "answers.parquet", assets / "answer_emb.npy")


def cmd_export_onnx(a):
    from .encoder import export_onnx

    export_onnx(_p(a.model_dir))


def cmd_eval_retriever(a):
    from .train_biencoder import evaluate

    evaluate(_p(a.model_dir), _p(a.assets), n=a.n)


def cmd_solve(a):
    from .puzzle import load
    from .solver import Solver

    solver = Solver(_p(a.assets), use_jev=not a.no_jev, budget=a.budget, verbose=True, jev_model=a.jev_model)
    for path in a.puzzle:
        pz = load(path)
        res = solver.solve(pz)
        print(f"\n== {path}  [{pz.title}]  {res.elapsed:.1f}s")
        print(pz.render(res.grid))
        if res.accuracy:
            print(json.dumps(res.accuracy))
        print("timing:", json.dumps({k: round(v, 2) for k, v in res.timing.items()}))
        if a.out:
            Path(a.out).write_text(json.dumps(res.to_dict(), indent=1))


def cmd_eval(a):
    from .puzzle import load
    from .solver import Solver

    solver = Solver(_p(a.assets), use_jev=not a.no_jev, budget=a.budget, verbose=a.verbose, jev_model=a.jev_model)
    paths = sorted(p for p in _p(a.dir).iterdir() if p.suffix.lower() in (".puz", ".json", ".xd", ".ipuz"))
    rows = []
    for p in paths:
        pz = load(p)
        res = solver.solve(pz)
        acc = res.accuracy
        rows.append({"puzzle": p.name, "elapsed": round(res.elapsed, 1), "t_perfect": res.t_perfect, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in acc.items()}, "timing": {k: round(v, 2) for k, v in res.timing.items()}})
        tp = f"{res.t_perfect:5.1f}s" if res.t_perfect is not None else "  never"
        print(f"{p.name:32s} stop {res.elapsed:5.1f}s  all-correct at {tp}  letters {acc.get('letter_acc', 0):.3f}  words {acc.get('word_acc', 0):.3f}")
        sys.stdout.flush()
    if rows:
        n = len(rows)
        tps = [r["t_perfect"] for r in rows if r["t_perfect"] is not None]
        print(f"\n{n} puzzles  mean letter acc {sum(r['letter_acc'] for r in rows) / n:.4f}  mean word acc {sum(r['word_acc'] for r in rows) / n:.4f}  perfect {sum(r['perfect'] for r in rows)}/{n}  max stop time {max(r['elapsed'] for r in rows):.1f}s")
        if tps:
            print(f"time to all-correct grid: mean {sum(tps) / len(tps):.1f}s  max {max(tps):.1f}s  over {len(tps)} solved puzzles")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xwjev")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-data", help="build corpus + assets from xd clues.tsv and CrosswordQA csvs")
    s.add_argument("--xd-tsv", required=True)
    s.add_argument("--cqa-csv", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_data)

    s = sub.add_parser("extend-vocab", help="add a WORD;score crossword word list to the answer vocabulary")
    s.add_argument("--assets", required=True)
    s.add_argument("--wordlist", required=True)
    s.add_argument("--min-score", type=int, default=25)
    s.set_defaults(fn=cmd_extend_vocab)

    s = sub.add_parser("train", help="fine-tune the clue->answer bi-encoder")
    s.add_argument("--assets", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    s.add_argument("--max-pairs", type=int, default=None)
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--batch-size", type=int, default=256)
    s.add_argument("--lr", type=float, default=5e-5)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("embed", help="encode the answer vocabulary with a trained model")
    s.add_argument("--assets", required=True)
    s.add_argument("--model-dir", required=True)
    s.set_defaults(fn=cmd_embed)

    s = sub.add_parser("export-onnx", help="export model_dir/model.onnx for the fast CPU solve path")
    s.add_argument("--model-dir", required=True)
    s.set_defaults(fn=cmd_export_onnx)

    s = sub.add_parser("eval-retriever")
    s.add_argument("--assets", required=True)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--n", type=int, default=5000)
    s.set_defaults(fn=cmd_eval_retriever)

    for name, fn in (("solve", cmd_solve), ("eval", cmd_eval)):
        s = sub.add_parser(name)
        s.add_argument("--assets", required=True)
        s.add_argument("--no-jev", action="store_true", help="offline mode: retriever only")
        s.add_argument("--jev-model", default=None)
        s.add_argument("--budget", type=float, default=45.0, help="wall-clock budget in seconds per puzzle")
        s.add_argument("--out", default=None)
        if name == "solve":
            s.add_argument("puzzle", nargs="+")
        else:
            s.add_argument("dir")
            s.add_argument("--verbose", action="store_true")
        s.set_defaults(fn=fn)

    a = ap.parse_args(argv)
    t0 = time.time()
    a.fn(a)
    print(f"[total {time.time() - t0:.1f}s]", file=sys.stderr)


if __name__ == "__main__":
    main()
