import struct

from xwjev.puzzle import Puzzle, parse_puz, parse_xd


def _mini_black():
    # 3x3 with one black corner
    return [[False, False, False], [False, False, False], [False, False, True]]


def test_numbering_and_slots():
    pz = Puzzle.from_grid(_mini_black(), {1: "a1", 4: "a4", 5: "a5"}, {1: "d1", 2: "d2", 3: "d3"})
    names = {s.name: s.cells for s in pz.slots}
    assert names["1A"] == [(0, 0), (0, 1), (0, 2)]
    assert names["4A"] == [(1, 0), (1, 1), (1, 2)]
    assert names["5A"] == [(2, 0), (2, 1)]
    assert names["1D"] == [(0, 0), (1, 0), (2, 0)]
    assert names["3D"] == [(0, 2), (1, 2)]
    assert pz.crossing(0, 1) == (pz.slots[4].idx, 0)  # 1A pos1 crosses 2D pos0


def test_parse_xd_roundtrip():
    xd = "Title: T\nAuthor: A\nDate: 2026-01-01\n\nCAT\nARE\nT#D\n\nA1. Feline ~ CAT\nA4. Exist ~ ARE\n\nD1. Feline ~ CAT\n"
    pz = parse_xd(xd)
    assert pz.width == 3 and pz.height == 3
    assert pz.answer(pz.slots[0]) == "CAT"
    assert pz.slots[0].clue == "Feline"
    assert pz.score(pz.solution)["perfect"]


def _make_puz(w, h, sol, clues, title="T", author="A"):
    hdr = bytearray(0x34)
    hdr[0x02:0x0E] = b"ACROSS&DOWN\0"
    hdr[0x18:0x1C] = b"1.3\0"
    hdr[0x2C] = w
    hdr[0x2D] = h
    struct.pack_into("<H", hdr, 0x2E, len(clues))
    body = sol.encode("latin-1") + ("-" * (w * h)).encode()
    strings = [title, author, "(c)"] + clues + [""]
    return bytes(hdr) + body + b"\0".join(s.encode("latin-1") for s in strings) + b"\0"


def test_parse_puz():
    sol = "CATARETD."
    raw = _make_puz(3, 3, sol, ["Feline", "Feline", "Pest", "Pet", "Exist", "Road, briefly"])
    pz = parse_puz(raw)
    assert pz.width == 3 and pz.solution[2][2] == "" and pz.black[2][2]
    by = {s.name: s for s in pz.slots}
    assert by["1A"].clue == "Feline" and pz.answer(by["1A"]) == "CAT"
    assert by["1D"].clue == "Feline" and by["2D"].clue == "Pest" and by["3D"].clue == "Pet"
    assert by["4A"].clue == "Exist" and by["5A"].clue == "Road, briefly"
