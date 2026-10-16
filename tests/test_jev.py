import asyncio

from xwjev.jev import NONE, JevJudge, JudgeRequest


def test_question_building_and_batching(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    j = JevJudge(per_request=2, concurrency=2, cache_dir=tmp_path)
    assert j.backend == "typesafe"
    calls = []

    async def fake_call(state, questions):
        calls.append((state, questions))
        out = {}
        for k, q in questions.items():
            opts = list(q["criteria"])
            n = len(opts)
            out[k] = {o: (0.5 if i == 0 else 0.5 / (n - 1)) for i, o in enumerate(opts)}
        return out

    monkeypatch.setattr(j, "_call", fake_call)
    reqs = [JudgeRequest(key=f"{i}A", clue=f"clue {i}", length=3, candidates=[("CAT", "cat"), ("DOG", "dog")], pattern="C??" if i == 0 else None) for i in range(5)]
    res = asyncio.run(j.judge_async(reqs, {"title": "T"}))
    assert len(res) == 5
    assert len(calls) == 3  # 5 questions, 2 per request
    state, qs = calls[0]
    assert state["puzzle"]["title"] == "T"
    q = qs["0A"]
    assert q["type"] == "choice"
    assert q["instructions"]["pattern"] == "C??"
    assert list(q["criteria"])[-1] == NONE
    assert res["0A"]["CAT"] == 0.5
    # second call is served from cache: no new API calls
    res2 = asyncio.run(j.judge_async(reqs))
    assert res2 == res and len(calls) == 3
