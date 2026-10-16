"""Puzzle model and parsers (.puz, .xd, xwordinfo JSON, NYT v6 JSON, ipuz)."""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .text import norm_answer

Cell = tuple[int, int]


@dataclass
class Slot:
    idx: int
    direction: str  # "A" or "D"
    number: int
    cells: list[Cell]
    clue: str

    @property
    def length(self) -> int:
        return len(self.cells)

    @property
    def name(self) -> str:
        return f"{self.number}{self.direction}"


@dataclass
class Puzzle:
    width: int
    height: int
    black: list[list[bool]]
    slots: list[Slot]
    solution: list[list[str]] | None = None  # None when unknown (locked grid)
    title: str = ""
    author: str = ""
    date: str = ""
    notes: str = ""
    cell_slots: dict[Cell, list[tuple[int, int]]] = field(default_factory=dict)  # cell -> [(slot_idx, pos)]

    def __post_init__(self) -> None:
        self.cell_slots = {}
        for s in self.slots:
            for pos, c in enumerate(s.cells):
                self.cell_slots.setdefault(c, []).append((s.idx, pos))

    # ----- construction helpers -----
    @staticmethod
    def number_grid(black: list[list[bool]]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """Return ([(num,r,c)] across starts, [(num,r,c)] down starts) using standard numbering."""
        h, w = len(black), len(black[0])
        across, down = [], []
        n = 0
        for r in range(h):
            for c in range(w):
                if black[r][c]:
                    continue
                sa = (c == 0 or black[r][c - 1]) and (c + 1 < w and not black[r][c + 1])
                sd = (r == 0 or black[r - 1][c]) and (r + 1 < h and not black[r + 1][c])
                if sa or sd:
                    n += 1
                    if sa:
                        across.append((n, r, c))
                    if sd:
                        down.append((n, r, c))
        return across, down

    @classmethod
    def from_grid(
        cls,
        black: list[list[bool]],
        across_clues: dict[int, str],
        down_clues: dict[int, str],
        solution: list[list[str]] | None = None,
        **meta,
    ) -> "Puzzle":
        h, w = len(black), len(black[0])
        across, down = cls.number_grid(black)
        slots: list[Slot] = []
        for num, r, c in across:
            cells = []
            cc = c
            while cc < w and not black[r][cc]:
                cells.append((r, cc))
                cc += 1
            slots.append(Slot(len(slots), "A", num, cells, across_clues.get(num, "")))
        for num, r, c in down:
            cells = []
            rr = r
            while rr < h and not black[rr][c]:
                cells.append((rr, c))
                rr += 1
            slots.append(Slot(len(slots), "D", num, cells, down_clues.get(num, "")))
        return cls(width=w, height=h, black=black, slots=slots, solution=solution, **meta)

    # ----- convenience -----
    def answer(self, slot: Slot) -> str | None:
        if self.solution is None:
            return None
        return "".join(self.solution[r][c] for r, c in slot.cells)

    def crossing(self, slot_idx: int, pos: int) -> tuple[int, int] | None:
        cell = self.slots[slot_idx].cells[pos]
        for si, p in self.cell_slots[cell]:
            if si != slot_idx:
                return si, p
        return None

    def empty_grid(self) -> list[list[str]]:
        return [["" for _ in range(self.width)] for _ in range(self.height)]

    def render(self, grid: list[list[str]]) -> str:
        rows = []
        for r in range(self.height):
            rows.append(" ".join("#" if self.black[r][c] else (grid[r][c] or ".") for c in range(self.width)))
        return "\n".join(rows)

    def score(self, grid: list[list[str]]) -> dict:
        """Letter / word accuracy of `grid` against the solution (if known)."""
        if self.solution is None:
            return {}
        n = ok = 0
        for r in range(self.height):
            for c in range(self.width):
                if self.black[r][c]:
                    continue
                n += 1
                ok += int(grid[r][c] == self.solution[r][c])
        wn = wok = 0
        for s in self.slots:
            wn += 1
            wok += int("".join(grid[r][c] for r, c in s.cells) == self.answer(s))
        return {
            "letters": n,
            "letters_ok": ok,
            "letter_acc": ok / max(n, 1),
            "words": wn,
            "words_ok": wok,
            "word_acc": wok / max(wn, 1),
            "perfect": ok == n,
        }


# ------------------------------------------------------------------ parsers


def load(path: str | Path) -> Puzzle:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".puz":
        return parse_puz(p.read_bytes())
    if suf == ".xd":
        return parse_xd(p.read_text(encoding="utf-8"))
    if suf in (".json", ".ipuz"):
        data = json.loads(p.read_text(encoding="utf-8"))
        return parse_json(data)
    raise ValueError(f"unknown puzzle format: {p}")


def parse_json(data: dict) -> Puzzle:
    if "body" in data and isinstance(data["body"], list):
        return parse_nyt_v6(data)
    if "puzzle" in data and "dimensions" in data:
        return parse_ipuz(data)
    if "grid" in data and "clues" in data:
        return parse_xwordinfo(data)
    raise ValueError("unrecognised JSON puzzle schema")


_CLUE_NUM = re.compile(r"^\s*(\d+)[.:)]?\s*(.*)$", re.S)


def _split_numbered(items: list[str]) -> dict[int, str]:
    out = {}
    for it in items:
        m = _CLUE_NUM.match(it)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def parse_xwordinfo(d: dict) -> Puzzle:
    rows, cols = d["size"]["rows"], d["size"]["cols"]
    g = d["grid"]
    black = [[g[r * cols + c] == "." for c in range(cols)] for r in range(rows)]
    sol = [[norm_answer(g[r * cols + c])[:1] if not black[r][c] else "" for c in range(cols)] for r in range(rows)]
    return Puzzle.from_grid(
        black,
        _split_numbered(d["clues"]["across"]),
        _split_numbered(d["clues"]["down"]),
        solution=sol,
        title=d.get("title", "") or "",
        author=d.get("author", "") or "",
        date=d.get("date", "") or "",
        notes=d.get("notepad", "") or "",
    )


def parse_nyt_v6(d: dict) -> Puzzle:
    body = d["body"][0]
    w, h = body["dimensions"]["width"], body["dimensions"]["height"]
    cells = body["cells"]
    black = [[not cells[r * w + c].get("answer") and "clues" not in cells[r * w + c] for c in range(w)] for r in range(h)]
    sol = None
    if any(cells[i].get("answer") for i in range(w * h)):
        sol = [[norm_answer(cells[r * w + c].get("answer", ""))[:1] if not black[r][c] else "" for c in range(w)] for r in range(h)]
    across, down = {}, {}
    for cl in body["clues"]:
        text = " ".join(t.get("plain") or t.get("formatted", "") for t in cl.get("text", []))
        num = int(re.sub(r"\D", "", cl["label"]) or 0)
        (across if cl["direction"].lower().startswith("a") else down)[num] = text
    notes = " ".join(n.get("text", "") for n in d.get("notes", []) if isinstance(n, dict))
    return Puzzle.from_grid(
        black,
        across,
        down,
        solution=sol,
        title=d.get("title", "") or "",
        author=", ".join(d.get("constructors", []) or []),
        date=d.get("publicationDate", "") or "",
        notes=notes,
    )


def parse_ipuz(d: dict) -> Puzzle:
    w, h = d["dimensions"]["width"], d["dimensions"]["height"]
    pz = d["puzzle"]

    def is_black(v) -> bool:
        if isinstance(v, dict):
            v = v.get("cell", v.get("value", 0))
        return v in ("#", None) or (isinstance(v, str) and v.strip() == "#")

    black = [[is_black(pz[r][c]) for c in range(w)] for r in range(h)]
    sol = None
    if "solution" in d:

        def letter(v):
            if isinstance(v, dict):
                v = v.get("value", "")
            return norm_answer(v or "")[:1]

        sol = [[letter(d["solution"][r][c]) if not black[r][c] else "" for c in range(w)] for r in range(h)]

    def clues(key_opts):
        for k in key_opts:
            if k in d.get("clues", {}):
                out = {}
                for item in d["clues"][k]:
                    if isinstance(item, dict):
                        out[int(item["number"])] = item.get("clue", "")
                    else:
                        out[int(item[0])] = item[1]
                return out
        return {}

    return Puzzle.from_grid(
        black,
        clues(["Across", "across"]),
        clues(["Down", "down"]),
        solution=sol,
        title=d.get("title", ""),
        author=d.get("author", ""),
        date=d.get("date", ""),
        notes=d.get("notes", ""),
    )


def parse_xd(text: str) -> Puzzle:
    blocks = re.split(r"\n\s*\n", text.strip())
    meta = {}
    for line in blocks[0].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    grid_lines = [ln.strip() for ln in blocks[1].splitlines() if ln.strip()]
    h, w = len(grid_lines), len(grid_lines[0])
    black = [[grid_lines[r][c] == "#" for c in range(w)] for r in range(h)]
    sol = [[grid_lines[r][c].upper() if not black[r][c] else "" for c in range(w)] for r in range(h)]
    across, down = {}, {}
    cre = re.compile(r"^([AD])(\d+)\.\s*(.*?)\s*~\s*(.*?)\s*$")
    for blk in blocks[2:]:
        for ln in blk.splitlines():
            m = cre.match(ln.strip())
            if m:
                (across if m.group(1) == "A" else down)[int(m.group(2))] = m.group(3)
    return Puzzle.from_grid(
        black,
        across,
        down,
        solution=sol,
        title=meta.get("title", ""),
        author=meta.get("author", ""),
        date=meta.get("date", ""),
        notes=meta.get("notes", ""),
    )


def parse_puz(raw: bytes) -> Puzzle:
    """AcrossLite .puz (v1.x / v2.x). Scrambled (locked) solutions are reported as unknown."""
    magic = raw[0x02:0x0E]
    if magic != b"ACROSS&DOWN\0":
        raise ValueError("not a .puz file")
    version = raw[0x18:0x1C].rstrip(b"\0").decode("ascii", "ignore")
    enc = "utf-8" if version.startswith("2") else "latin-1"
    w, h = raw[0x2C], raw[0x2D]
    nclues = struct.unpack_from("<H", raw, 0x2E)[0]
    scrambled = struct.unpack_from("<H", raw, 0x32)[0]
    off = 0x34
    sol_raw = raw[off : off + w * h].decode("latin-1")
    off += 2 * w * h  # skip player state
    strings = []
    while len(strings) < nclues + 4 and off < len(raw):
        end = raw.index(b"\0", off)
        strings.append(raw[off:end].decode(enc, "replace"))
        off = end + 1
    title, author = strings[0], strings[1]
    clues = strings[3 : 3 + nclues]
    notes = strings[3 + nclues] if len(strings) > 3 + nclues else ""
    black = [[sol_raw[r * w + c] == "." for c in range(w)] for r in range(h)]
    sol = None
    if not scrambled:
        sol = [[norm_answer(sol_raw[r * w + c])[:1] if not black[r][c] else "" for c in range(w)] for r in range(h)]
    across_starts, down_starts = Puzzle.number_grid(black)
    a_nums = {n for n, _, _ in across_starts}
    d_nums = {n for n, _, _ in down_starts}
    across, down = {}, {}
    it = iter(clues)
    for n in sorted(a_nums | d_nums):
        if n in a_nums:
            across[n] = next(it, "")
        if n in d_nums:
            down[n] = next(it, "")
    m = re.search(r"(\d{4}-\d{2}-\d{2}|\w+ \d{1,2}, \d{4})", title)
    return Puzzle.from_grid(
        black,
        across,
        down,
        solution=sol,
        title=title,
        author=author,
        date=m.group(1) if m else "",
        notes=notes,
    )
