"""
ensemble.py – Core logic for the GPT-Arena ensemble pipeline.

Pipeline
--------
1. Run ``n_arena`` 7 B-class models **concurrently** (asyncio + httpx).
   Each slot uses different sampling parameters to maximise answer diversity.
2. Feed all candidate answers to a larger 20 B-class synthesis model that
   produces one final, coherent "wall" answer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

import httpx

from config import (
    OLLAMA_BASE_URL,
    SYNTHESIS_PROMPT_TEMPLATE,
    GenerationRequest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArenaAnswer:
    """A single answer produced by one arena slot."""
    slot: int
    model: str
    temperature: float
    text: str
    error: Optional[str] = None


@dataclass
class EnsembleResult:
    """Full result of one ensemble run."""
    question: str
    arena_answers: List[ArenaAnswer]
    synthesis: str


# ---------------------------------------------------------------------------
# Low-level Ollama helpers
# ---------------------------------------------------------------------------

async def _ollama_generate(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.90,
    num_predict: int = 512,
) -> str:
    """Call the Ollama /api/generate endpoint and return the full response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=300.0,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


# ---------------------------------------------------------------------------
# Arena stage
# ---------------------------------------------------------------------------

async def _run_arena_slot(
    client: httpx.AsyncClient,
    slot: int,
    model: str,
    prompt: str,
    params: dict,
    max_tokens: int,
) -> ArenaAnswer:
    """Run a single arena slot, catching errors gracefully."""
    temperature = params.get("temperature", 0.7)
    top_p = params.get("top_p", 0.90)
    logger.info("Arena slot %d | model=%s | temp=%.2f", slot, model, temperature)
    try:
        text = await _ollama_generate(
            client,
            model,
            prompt,
            temperature=temperature,
            top_p=top_p,
            num_predict=max_tokens,
        )
        return ArenaAnswer(slot=slot, model=model, temperature=temperature, text=text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Arena slot %d failed: %s", slot, exc)
        return ArenaAnswer(
            slot=slot,
            model=model,
            temperature=temperature,
            text="",
            error=str(exc),
        )


async def run_arena(
    client: httpx.AsyncClient,
    request: GenerationRequest,
) -> List[ArenaAnswer]:
    """
    Run all arena slots with bounded concurrency.

    Returns a list of ArenaAnswer objects (one per slot, in slot order).
    """
    semaphore = asyncio.Semaphore(request.max_concurrent)

    async def _guarded(slot: int, model: str, params: dict) -> ArenaAnswer:
        async with semaphore:
            return await _run_arena_slot(
                client,
                slot,
                model,
                request.prompt,
                params,
                request.arena_max_tokens,
            )

    tasks = [
        _guarded(i, model, params)
        for i, (model, params) in enumerate(
            zip(request.arena_models, request.arena_params)
        )
    ]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Synthesis stage
# ---------------------------------------------------------------------------

def _build_synthesis_prompt(question: str, answers: List[ArenaAnswer]) -> str:
    """Format the prompt that the synthesis model will receive."""
    valid = [a for a in answers if a.text and not a.error]
    if not valid:
        raise ValueError("All arena slots failed – no answers to synthesise.")

    formatted_answers = "\n\n".join(
        f"[Answer {a.slot + 1} | temp={a.temperature:.2f}]\n{a.text.strip()}"
        for a in valid
    )
    return SYNTHESIS_PROMPT_TEMPLATE.format(
        n=len(valid),
        question=question,
        answers=formatted_answers,
    )


async def run_synthesis(
    client: httpx.AsyncClient,
    request: GenerationRequest,
    arena_answers: List[ArenaAnswer],
) -> str:
    """Feed all arena answers to the synthesis model and return its output."""
    synthesis_prompt = _build_synthesis_prompt(request.prompt, arena_answers)
    logger.info(
        "Synthesis | model=%s | arena_answers=%d",
        request.synthesis_model,
        len([a for a in arena_answers if not a.error]),
    )
    return await _ollama_generate(
        client,
        request.synthesis_model,
        synthesis_prompt,
        temperature=0.3,   # low temperature for the consolidation step
        top_p=0.85,
        num_predict=request.synthesis_max_tokens,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_ensemble(request: GenerationRequest) -> EnsembleResult:
    """
    Execute the full ensemble pipeline:
      1. Run 10 × 7 B arena models concurrently.
      2. Synthesise their outputs with the 20 B model.

    Returns an EnsembleResult with all intermediate answers and the final text.
    """
    async with httpx.AsyncClient() as client:
        # Stage 1 – arena
        arena_answers = await run_arena(client, request)

        # Stage 2 – synthesis
        synthesis_text = await run_synthesis(client, request, arena_answers)

    return EnsembleResult(
        question=request.prompt,
        arena_answers=arena_answers,
        synthesis=synthesis_text,
    )
