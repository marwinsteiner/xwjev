"""Fill scripts/replay_template.html with a traced solve (frames.json + result.json) -> a standalone HTML page.

    python scripts/build_replay_page.py puzzles/nyt_2026-10-15_solve out.html
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("replay_template.html")


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    frames = json.loads((src / "frames.json").read_text())
    res = json.loads((src / "result.json").read_text())
    acc = res["accuracy"]
    jev_used = bool(res.get("jev")) and res["jev"].get("requests", 0) > 0

    lede = (
        f"Puzzle: the New York Times daily crossword for {frames['date']} (Thursday), constructed by McKay Fritz, edited by Will Shortz, "
        f"fetched through the user's own subscription. Solver: exact clue lookup over 7.2M historical clue/answer pairs plus a bi-encoder fine-tuned on them, "
        f"loopy belief propagation over the crossings, three pattern-expansion rounds and local search. "
        + ("Jev judged the candidates." if jev_used else "Jev was not called in this run (no TYPESAFE_API_KEY in the environment), so the candidate probabilities come from the retriever alone and the Jev-only repair rounds were skipped.")
    )
    tp = res.get("t_perfect")
    stats = [
        (f"{tp:.1f} s" if tp is not None else "not reached", "time to all letters correct"),
        (f"{res['elapsed']:.1f} s", "time until the solver stopped"),
        (f"{acc['letters_ok']} / {acc['letters']}", "letters correct at the end"),
        (f"{acc['words_ok']} / {acc['words']}", "entries correct at the end"),
        (f"{res['model_load_s']:.1f} s", "model + index load (before t = 0)"),
        (f"{len(frames['frames'])}", "frames recorded"),
    ]
    stats_html = "".join(f'<div class="stat"><div class="v">{html.escape(v)}</div><div class="k">{html.escape(k)}</div></div>' for v, k in stats)

    rows = []
    for fr in frames["frames"]:
        if "sweep" in fr["stage"]:
            continue
        a = fr.get("acc") or {}
        rows.append(f"<tr><td>{html.escape(fr['stage'])}</td><td class=\"num\">{fr['t']:.2f}</td><td class=\"num\">{a.get('letters_ok', '')}</td></tr>")
    stages_html = "".join(rows)

    misses = [s for s in res["slots"] if s["answer"] != s["truth"]]
    miss_html = "".join(
        f"<tr><td>{html.escape(s['slot'])}</td><td>{html.escape(s['clue'])}</td><td class=\"w bad\">{html.escape(s['answer'])}</td><td class=\"w ok\">{html.escape(s['truth'])}</td></tr>"
        for s in misses
    )
    theme = (
        "This Thursday has a trick: the revealer at 55-Across, FIRE BLANKS, says that the word FIRE fills every blank, so \"___ ball\" is COMET (fireball), "
        "\"Cease ___\" is TRUCE and \"___ place\" is HEARTH. The retriever has never seen those clue/answer pairs and reads the blanks literally, "
        "which is where most of the misses come from; the rest are crossing errors that ride on them. Those are exactly the slots the Jev-judged repair round exists for."
    )
    how = (
        "The benchmark number is the time at which every letter is correct, measured against the published answers (the solver never sees them; they are used only to timestamp the frames). "
        "The solver cannot know when it is done, so it keeps searching until its score converges, which is the later stop time. "
        "Each early frame is one sweep of belief propagation: every entry re-reads the letter distributions its crossings send, re-weights its candidate answers and sends updated distributions back. "
        "Pale cells have a top letter with marginal probability under 0.97; letters get darker as the crossings agree. Then an iterated local search swaps whole entries for vocabulary words "
        "that fit the confident crossing letters and raise the global score; each accepted improvement is a frame. The last frame marks cells that differ from the published answers."
    )

    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("__LEDE__", html.escape(lede)).replace("__STATS__", stats_html).replace("__STAGES__", stages_html)
    page = page.replace("__MISSES__", miss_html).replace("__THEME__", html.escape(theme)).replace("__HOW__", html.escape(how))
    page = page.replace("__FRAMES__", json.dumps(frames).replace("</", "<\\/"))
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {len(misses)} misses")


if __name__ == "__main__":
    main()
