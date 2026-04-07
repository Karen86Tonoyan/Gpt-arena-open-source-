"""
tests/test_ensemble.py – Unit tests for the ensemble pipeline.

Uses pytest-asyncio + unittest.mock to stub Ollama calls so tests can run
without a live Ollama server.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from config import ARENA_PARAMS, GenerationRequest
from ensemble import (
    ArenaAnswer,
    _build_synthesis_prompt,
    run_arena,
    run_ensemble,
    run_synthesis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(prompt: str = "What is the capital of France?") -> GenerationRequest:
    """Return a minimal GenerationRequest with 10 arena slots."""
    return GenerationRequest(
        prompt=prompt,
        arena_models=["model-7b"] * 10,
        arena_params=ARENA_PARAMS,
        synthesis_model="model-20b",
        arena_max_tokens=64,
        synthesis_max_tokens=128,
        max_concurrent=5,
    )


def _fake_ollama_response(text: str) -> MagicMock:
    """Build a mock httpx.Response that looks like an Ollama reply."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json = MagicMock(return_value={"response": text})
    return mock


# ---------------------------------------------------------------------------
# Tests – _build_synthesis_prompt
# ---------------------------------------------------------------------------

class TestBuildSynthesisPrompt:
    def test_contains_question(self):
        answers = [
            ArenaAnswer(slot=0, model="m", temperature=0.5, text="Paris"),
            ArenaAnswer(slot=1, model="m", temperature=0.7, text="It is Paris"),
        ]
        prompt = _build_synthesis_prompt("What is the capital of France?", answers)
        assert "What is the capital of France?" in prompt

    def test_contains_all_answers(self):
        answers = [
            ArenaAnswer(slot=i, model="m", temperature=0.5, text=f"Answer {i}")
            for i in range(10)
        ]
        prompt = _build_synthesis_prompt("Q?", answers)
        for i in range(10):
            assert f"Answer {i}" in prompt

    def test_skips_errored_slots(self):
        answers = [
            ArenaAnswer(slot=0, model="m", temperature=0.5, text="Good answer"),
            ArenaAnswer(slot=1, model="m", temperature=0.7, text="", error="timeout"),
        ]
        prompt = _build_synthesis_prompt("Q?", answers)
        assert "Good answer" in prompt
        assert "timeout" not in prompt

    def test_raises_when_all_failed(self):
        answers = [
            ArenaAnswer(slot=i, model="m", temperature=0.5, text="", error="fail")
            for i in range(10)
        ]
        with pytest.raises(ValueError, match="All arena slots failed"):
            _build_synthesis_prompt("Q?", answers)

    def test_slot_numbering_is_one_based(self):
        answers = [ArenaAnswer(slot=0, model="m", temperature=0.3, text="A")]
        prompt = _build_synthesis_prompt("Q?", answers)
        assert "[Answer 1 |" in prompt


# ---------------------------------------------------------------------------
# Tests – run_arena
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_arena_returns_ten_answers():
    """run_arena must produce exactly one ArenaAnswer per slot (10 total)."""
    request = _make_request()

    with patch("ensemble._ollama_generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Paris"
        import httpx
        async with httpx.AsyncClient() as client:
            answers = await run_arena(client, request)

    assert len(answers) == 10
    assert all(isinstance(a, ArenaAnswer) for a in answers)
    assert all(a.text == "Paris" for a in answers)


@pytest.mark.asyncio
async def test_run_arena_slot_indices_match():
    """Each slot's .slot attribute must equal its position in the list."""
    request = _make_request()

    with patch("ensemble._ollama_generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "OK"
        import httpx
        async with httpx.AsyncClient() as client:
            answers = await run_arena(client, request)

    for i, a in enumerate(answers):
        assert a.slot == i


@pytest.mark.asyncio
async def test_run_arena_handles_partial_failures():
    """If some slots raise, they must come back with error set, not crash."""
    request = _make_request()
    call_count = 0

    async def flaky(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            raise RuntimeError("network error")
        return "Some answer"

    with patch("ensemble._ollama_generate", side_effect=flaky):
        import httpx
        async with httpx.AsyncClient() as client:
            answers = await run_arena(client, request)

    assert len(answers) == 10
    errors = [a for a in answers if a.error]
    successes = [a for a in answers if not a.error]
    assert len(errors) > 0
    assert len(successes) > 0


@pytest.mark.asyncio
async def test_run_arena_respects_diverse_temperatures():
    """
    Each arena slot must use the temperature specified in its ARENA_PARAMS entry.
    We capture the kwargs passed to _ollama_generate and check diversity.
    """
    request = _make_request()
    seen_temps: list[float] = []

    async def capture_temp(*args, temperature=0.7, **kwargs):
        seen_temps.append(temperature)
        return "answer"

    with patch("ensemble._ollama_generate", side_effect=capture_temp):
        import httpx
        async with httpx.AsyncClient() as client:
            await run_arena(client, request)

    # We expect 10 distinct temperatures matching ARENA_PARAMS
    expected = [p["temperature"] for p in ARENA_PARAMS]
    assert seen_temps == expected, (
        f"Temperatures sent to arena slots {seen_temps!r} "
        f"don't match ARENA_PARAMS {expected!r}"
    )


# ---------------------------------------------------------------------------
# Tests – run_synthesis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_synthesis_returns_string():
    request = _make_request()
    answers = [ArenaAnswer(slot=i, model="m", temperature=0.5, text=f"A{i}") for i in range(10)]

    with patch("ensemble._ollama_generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Final consolidated answer."
        import httpx
        async with httpx.AsyncClient() as client:
            result = await run_synthesis(client, request, answers)

    assert result == "Final consolidated answer."


@pytest.mark.asyncio
async def test_run_synthesis_uses_synthesis_model():
    """The synthesis call must target the synthesis model, not arena models."""
    request = _make_request()
    answers = [ArenaAnswer(slot=0, model="model-7b", temperature=0.5, text="Paris")]
    called_with_model: list[str] = []

    async def spy(client, model, prompt, **kwargs):
        called_with_model.append(model)
        return "consolidated"

    with patch("ensemble._ollama_generate", side_effect=spy):
        import httpx
        async with httpx.AsyncClient() as client:
            await run_synthesis(client, request, answers)

    assert called_with_model == ["model-20b"]


# ---------------------------------------------------------------------------
# Tests – run_ensemble (integration-level, full mock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_ensemble_full_pipeline():
    """End-to-end: arena + synthesis both mocked, check EnsembleResult shape."""
    request = _make_request("Who wrote Hamlet?")
    call_num = 0

    async def fake_generate(client, model, prompt, **kwargs):
        nonlocal call_num
        call_num += 1
        if model == "model-7b":
            return f"Shakespeare (slot {call_num})"
        return "William Shakespeare wrote Hamlet."

    with patch("ensemble._ollama_generate", side_effect=fake_generate):
        result = await run_ensemble(request)

    assert result.question == "Who wrote Hamlet?"
    assert len(result.arena_answers) == 10
    assert "Shakespeare" in result.synthesis


@pytest.mark.asyncio
async def test_run_ensemble_raises_when_all_arena_fail():
    """If every arena slot fails, run_ensemble must propagate a ValueError."""
    request = _make_request()

    async def always_fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    with patch("ensemble._ollama_generate", side_effect=always_fail):
        with pytest.raises(ValueError, match="All arena slots failed"):
            await run_ensemble(request)
