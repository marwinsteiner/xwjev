"""Jev (TypeSafe AI System One model) as the clue judge.

For every slot we ask one Choice question: "which candidate answers this clue?" with the
candidate answers as the options plus a NONE option. Jev returns a calibrated probability
for every option; those become the unary potentials for belief propagation and the NONE
mass raises the slot's "answer not in list" probability.

Questions are evaluated in parallel and in isolation against a shared `state`, so we pack
several clues per request (the clue itself travels in the question `instructions`) and fire
requests concurrently.

Backends:
  * TypeSafe direct   (TYPESAFE_API_KEY)   via typesafe-sdk, POST https://api.typesafe.ai/v1/systemone
  * OpenRouter        (OPENROUTER_API_KEY) via POST https://openrouter.ai/api/alpha/decisions, model typesafe/jev-*
If no key is present `JevJudge.available` is False and the solver runs retriever-only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

NONE = "NONE_OF_THESE"

CONTEXT = (
    "You are an expert solver of American-style crosswords such as the New York Times daily crossword. "
    "Each question gives one clue and the answer length. Choose the candidate that correctly answers the clue. "
    "Conventions: an abbreviation or 'Abbr.' in the clue means an abbreviated answer; a question mark marks wordplay or a pun; "
    "'___' is a fill-in-the-blank; the answer matches the clue's part of speech, tense and number; "
    "'e.g.' or 'for one' means the answer is a category containing the clue; clues in a foreign language take answers in that language; "
    "a 'pattern' field, when present, gives letters already fixed by crossing answers (? = unknown) and the correct answer must fit it. "
    "If no candidate fits, choose NONE_OF_THESE."
)


@dataclass
class JudgeRequest:
    key: str  # unique id within the batch (slot name)
    clue: str
    length: int
    candidates: list[tuple[str, str]]  # (ANSWER, display text) ordered best-first
    pattern: str | None = None


class JevJudge:
    def __init__(self, model: str | None = None, per_request: int = 12, concurrency: int = 8, cache_dir: Path | None = None, timeout: float = 20.0):
        self.ts_key = os.environ.get("TYPESAFE_API_KEY")
        self.or_key = os.environ.get("OPENROUTER_API_KEY")
        self.backend = "typesafe" if self.ts_key else ("openrouter" if self.or_key else None)
        self.model = model or ("jev-latest" if self.backend == "typesafe" else "typesafe/jev-1.14")
        self.per_request = per_request
        self.concurrency = concurrency
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache: dict[str, dict[str, float]] = {}
        self.stats = {"requests": 0, "questions": 0, "input_tokens": 0, "errors": 0, "seconds": 0.0}
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def available(self) -> bool:
        return self.backend is not None

    # ---------------- cache
    @staticmethod
    def _ckey(r: JudgeRequest, model: str) -> str:
        h = hashlib.sha1(json.dumps([model, r.clue, r.length, r.pattern, [a for a, _ in r.candidates]]).encode()).hexdigest()
        return h

    def _cache_get(self, k: str):
        if k in self.cache:
            return self.cache[k]
        if self.cache_dir and (self.cache_dir / f"{k}.json").exists():
            v = json.loads((self.cache_dir / f"{k}.json").read_text())
            self.cache[k] = v
            return v
        return None

    def _cache_put(self, k: str, v: dict[str, float]):
        self.cache[k] = v
        if self.cache_dir:
            (self.cache_dir / f"{k}.json").write_text(json.dumps(v))

    # ---------------- question building
    def _question(self, r: JudgeRequest) -> dict:
        instr = {"clue": r.clue, "answer_length": r.length}
        if r.pattern and "?" in r.pattern and r.pattern != "?" * r.length:
            instr["pattern"] = r.pattern
        elif r.pattern and "?" not in r.pattern:
            instr["pattern"] = r.pattern
        criteria = {}
        for ans, text in r.candidates:
            if ans == NONE or ans in criteria:
                continue
            criteria[ans] = f"{text} ({len(ans)} letters)" if text and text.replace(" ", "").upper() != ans else f"{ans.lower()} ({len(ans)} letters)"
            if len(criteria) >= 254:
                break
        criteria[NONE] = "none of the candidates correctly answers the clue"
        return {"type": "choice", "instructions": instr, "criteria": criteria}

    @staticmethod
    def _state(puzzle_meta: dict | None) -> dict:
        st = {"task": CONTEXT}
        if puzzle_meta:
            st["puzzle"] = {k: v for k, v in puzzle_meta.items() if v}
        return st

    # ---------------- backends
    def _new_client(self):
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        return AsyncTypeSafeClient(api_key=self.ts_key, model=self.model, timeout=self.timeout, retry=RetryPolicy(max_retries=2, timeout=self.timeout))

    async def _call_typesafe(self, state: dict, questions: dict[str, dict]) -> dict[str, dict[str, float]]:
        from typesafe_sdk import Choice

        qs = {k: Choice(instructions=q["instructions"], criteria=q["criteria"]) for k, q in questions.items()}
        resp = await self._client.system_one(state=state, questions=qs)
        self.stats["input_tokens"] += int(getattr(resp.usage, "input_tokens", 0) or 0)
        return {k: dict(a.probabilities) for k, a in resp.choices.items()}

    async def _call_openrouter(self, state: dict, questions: dict[str, dict]) -> dict[str, dict[str, float]]:
        import httpx

        body = {"model": self.model, "state": state, "questions": questions}
        async with httpx.AsyncClient(timeout=self.timeout) as cl:
            for attempt in range(3):
                r = await cl.post("https://openrouter.ai/api/alpha/decisions", headers={"Authorization": f"Bearer {self.or_key}", "Content-Type": "application/json"}, json=body)
                if r.status_code in (429, 529, 502, 503) and attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
        d = r.json()
        self.stats["input_tokens"] += int(d.get("usage", {}).get("input_tokens", 0) or 0)
        return {k: dict(a.get("probabilities", {})) for k, a in d["answers"].items() if a.get("type") == "choice"}

    async def _call(self, state: dict, questions: dict[str, dict]) -> dict[str, dict[str, float]]:
        self.stats["requests"] += 1
        self.stats["questions"] += len(questions)
        if self.backend == "typesafe":
            return await self._call_typesafe(state, questions)
        return await self._call_openrouter(state, questions)

    # ---------------- public
    async def judge_async(self, reqs: list[JudgeRequest], puzzle_meta: dict | None = None, deadline: float | None = None) -> dict[str, dict[str, float]]:
        """Return {req.key: {ANSWER: prob, ..., NONE_OF_THESE: prob}} for every request (missing on failure)."""
        t0 = time.time()
        out: dict[str, dict[str, float]] = {}
        todo = []
        for r in reqs:
            if not r.candidates:
                continue
            ck = self._ckey(r, self.model)
            v = self._cache_get(ck)
            if v is not None:
                out[r.key] = v
            else:
                todo.append((ck, r))
        if not todo or not self.available:
            return out
        state = self._state(puzzle_meta)
        batches = [todo[i : i + self.per_request] for i in range(0, len(todo), self.per_request)]
        sem = asyncio.Semaphore(self.concurrency)
        # one async client per event loop (judge() runs each round in its own asyncio.run)
        self._client = self._new_client() if self.backend == "typesafe" else None

        async def run(batch):
            async with sem:
                if deadline and time.time() > deadline:
                    return
                qs = {r.key: self._question(r) for _, r in batch}
                try:
                    res = await asyncio.wait_for(self._call(state, qs), timeout=self.timeout + 5)
                except Exception as e:  # noqa: BLE001
                    self.stats["errors"] += 1
                    self.stats.setdefault("last_error", str(e)[:300])
                    return
                for ck, r in batch:
                    if r.key in res:
                        probs = {k: float(v) for k, v in res[r.key].items()}
                        out[r.key] = probs
                        self._cache_put(ck, probs)

        try:
            await asyncio.gather(*(run(b) for b in batches))
        finally:
            if self._client is not None:
                try:
                    await self._client.aclose()
                except Exception:  # noqa: BLE001
                    pass
                self._client = None
        self.stats["seconds"] += time.time() - t0
        return out

    def judge(self, reqs: list[JudgeRequest], puzzle_meta: dict | None = None, deadline: float | None = None) -> dict[str, dict[str, float]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                return ex.submit(lambda: asyncio.run(self.judge_async(reqs, puzzle_meta, deadline))).result()
        return asyncio.run(self.judge_async(reqs, puzzle_meta, deadline))
