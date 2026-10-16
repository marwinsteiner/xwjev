"""Solve a puzzle with tracing on and render the solve as an animated GIF + a frames JSON.

    python scripts/render_solve.py --assets assets puzzle.json --out out_dir [--no-jev]

Writes out_dir/solve.gif, out_dir/frames.json (letters + confidence per frame, for the HTML replay)
and out_dir/result.json (final grid, timing, accuracy, per-slot answers).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from xwjev.puzzle import Puzzle, load
from xwjev.solver import Solver

FONT_CANDIDATES = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def _font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_frame(pz: Puzzle, frame: dict, cell: int, final: bool) -> Image.Image:
    W, H = pz.width * cell, pz.height * cell
    top, bottom = int(cell * 1.9), int(cell * 1.1)
    img = Image.new("RGB", (W + 2 * cell // 2, H + top + bottom), (250, 250, 248))
    d = ImageDraw.Draw(img)
    f_letter, f_small, f_head = _font(int(cell * 0.62)), _font(max(10, int(cell * 0.28))), _font(int(cell * 0.5))
    ox, oy = cell // 2, top
    numbers = {}
    for s in pz.slots:
        numbers.setdefault(s.cells[0], s.number)
    for r in range(pz.height):
        for c in range(pz.width):
            x, y = ox + c * cell, oy + r * cell
            if pz.black[r][c]:
                d.rectangle([x, y, x + cell, y + cell], fill=(30, 30, 30))
                continue
            conf = frame["conf"][r][c]
            letter = frame["grid"][r][c]
            correct = pz.solution is not None and letter == pz.solution[r][c]
            if final and pz.solution is not None and letter and not correct:
                bg = (250, 205, 205)
            elif letter:
                # confidence tint: pale yellow (uncertain) -> white (certain)
                k = max(0.0, min(1.0, (conf - 0.3) / 0.7))
                bg = (255, 255, int(200 + 55 * k))
            else:
                bg = (255, 255, 255)
            d.rectangle([x, y, x + cell, y + cell], fill=bg, outline=(120, 120, 120))
            if (r, c) in numbers:
                d.text((x + 2, y + 1), str(numbers[(r, c)]), fill=(90, 90, 90), font=f_small)
            if letter:
                g = int(170 * (1 - max(0.0, min(1.0, (conf - 0.2) / 0.8))))
                col = (g, g, g) if not (final and not correct) else (180, 20, 20)
                bbox = d.textbbox((0, 0), letter, font=f_letter)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                d.text((x + (cell - tw) / 2 - bbox[0], y + (cell - th) / 2 - bbox[1] + 1), letter, fill=col, font=f_letter)
    title = (pz.title or "Crossword").strip()
    d.text((ox, int(cell * 0.25)), title[:60], fill=(20, 20, 20), font=f_head)
    acc = frame.get("acc") or {}
    status = f"{frame['stage']}   t = {frame['t']:.2f} s"
    if acc:
        status += f"   letters {acc['letters_ok']}/{acc['letters']}"
    d.text((ox, int(cell * 1.1)), status, fill=(60, 60, 60), font=f_small)
    legend = "letter shade = confidence from crossings; red = wrong in final grid" if final else "letter shade = confidence from crossings"
    d.text((ox, oy + H + int(cell * 0.3)), legend, fill=(110, 110, 110), font=f_small)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("puzzle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--cell", type=int, default=40)
    ap.add_argument("--ms", type=int, default=140, help="ms per sweep frame")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pz = load(a.puzzle)
    t_load = time.time()
    solver = Solver(Path(a.assets), use_jev=not a.no_jev, verbose=True)
    load_s = time.time() - t_load
    solver.trace = []
    res = solver.solve(pz)
    frames = solver.trace
    print(f"{len(frames)} frames, solve {res.elapsed:.2f}s, model load {load_s:.1f}s, acc {res.accuracy}")

    imgs, durations = [], []
    for i, fr in enumerate(frames):
        final = i == len(frames) - 1
        imgs.append(render_frame(pz, fr, a.cell, final))
        is_sweep = "sweep" in fr["stage"]
        durations.append(a.ms if is_sweep else (3500 if final else 900))
    # hold the first and last frames
    durations[0] = 1200
    imgs = [im.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for im in imgs]
    imgs[0].save(out / "solve.gif", save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True)
    imgs[-1].convert("RGB").save(out / "final.png")
    (out / "frames.json").write_text(json.dumps({"width": pz.width, "height": pz.height, "black": pz.black, "title": pz.title, "date": pz.date,
                                                  "solution": pz.solution, "numbers": {f"{s.cells[0][0]},{s.cells[0][1]}": s.number for s in pz.slots},
                                                  "frames": frames, "durations": durations}))
    (out / "result.json").write_text(json.dumps({**res.to_dict(), "model_load_s": load_s, "frames": len(frames)}, indent=1))
    print("wrote", out / "solve.gif", f"({(out / 'solve.gif').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
