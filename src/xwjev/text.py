"""Text normalisation shared by data building, retrieval and solving."""

from __future__ import annotations

import re
import unicodedata

_NON_ALPHA = re.compile(r"[^A-Z]")
_WS = re.compile(r"\s+")
_BLANK = re.compile(r"_{2,}|…|\.\.\.")

_wordninja = None


def norm_answer(s: str) -> str:
    """Uppercase ASCII letters only (accents stripped). Used as the grid form of an answer."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return _NON_ALPHA.sub("", s.upper())


def norm_clue(s: str) -> str:
    """Canonical key for exact-match lookup: lowercase, blanks unified, whitespace collapsed."""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = _BLANK.sub("___", s)
    s = _WS.sub(" ", s)
    return s


def clue_key_loose(s: str) -> str:
    """Looser key: also drops punctuation, so `Clean-up target` == `Cleanup target.`"""
    s = norm_clue(s)
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    return _WS.sub(" ", s).strip()


def segment_answer(ans_norm: str) -> str:
    """Split an all-caps answer into words for the text encoder / Jev display."""
    global _wordninja
    if not ans_norm:
        return ""
    if len(ans_norm) <= 3:
        return ans_norm.lower()
    if _wordninja is None:
        import wordninja  # lazy: ~1s to load its wordlist

        _wordninja = wordninja
    return " ".join(_wordninja.split(ans_norm.lower()))


def query_text(clue: str, length: int) -> str:
    """The exact string fed to the clue encoder (train and inference must agree)."""
    return f"{clue.strip()} ({length})"
