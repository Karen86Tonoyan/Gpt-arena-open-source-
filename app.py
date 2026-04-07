"""
app.py – FastAPI application for GPT Arena.

Endpoints
---------
GET  /           – serve the web UI (static/index.html)
POST /generate   – run the ensemble pipeline and return results
GET  /health     – liveness check
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    ARENA_MAX_TOKENS,
    ARENA_MODELS,
    ARENA_PARAMS,
    MAX_CONCURRENT_ARENA,
    SYNTHESIS_MAX_TOKENS,
    SYNTHESIS_MODEL,
    GenerationRequest,
)
from ensemble import run_ensemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GPT Arena – LLM Ensemble",
    description=(
        "Runs 10 × 7 B arena models in parallel, then synthesizes their "
        "diverse answers into one final response using a 20 B model."
    ),
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The question / prompt to answer")
    arena_models: Optional[List[str]] = Field(
        default=None,
        description="Override the list of arena model tags (must have exactly 10 entries)",
    )
    synthesis_model: Optional[str] = Field(
        default=None,
        description="Override the synthesis model tag",
    )
    arena_max_tokens: int = Field(default=ARENA_MAX_TOKENS, ge=64, le=4096)
    synthesis_max_tokens: int = Field(default=SYNTHESIS_MAX_TOKENS, ge=128, le=8192)
    max_concurrent: int = Field(default=MAX_CONCURRENT_ARENA, ge=1, le=10)


class ArenaAnswerOut(BaseModel):
    slot: int
    model: str
    temperature: float
    text: str
    error: Optional[str]


class GenerateResponse(BaseModel):
    question: str
    arena_answers: List[ArenaAnswerOut]
    synthesis: str
    arena_success_count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index))


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """
    Run the ensemble pipeline:
      1. 10 × 7 B arena models answer the prompt (concurrently, varied parameters).
      2. 20 B synthesis model consolidates all valid answers into one final response.
    """
    models = body.arena_models or ARENA_MODELS
    if len(models) != 10:
        raise HTTPException(
            status_code=422,
            detail=f"arena_models must have exactly 10 entries, got {len(models)}",
        )

    request = GenerationRequest(
        prompt=body.prompt,
        arena_models=models,
        arena_params=ARENA_PARAMS,
        synthesis_model=body.synthesis_model or SYNTHESIS_MODEL,
        arena_max_tokens=body.arena_max_tokens,
        synthesis_max_tokens=body.synthesis_max_tokens,
        max_concurrent=body.max_concurrent,
    )

    try:
        result = await run_ensemble(request)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ensemble pipeline failed")
        raise HTTPException(status_code=500, detail=f"Ensemble error: {exc}") from exc

    arena_out = [
        ArenaAnswerOut(
            slot=a.slot,
            model=a.model,
            temperature=a.temperature,
            text=a.text,
            error=a.error,
        )
        for a in result.arena_answers
    ]

    return GenerateResponse(
        question=result.question,
        arena_answers=arena_out,
        synthesis=result.synthesis,
        arena_success_count=sum(1 for a in result.arena_answers if not a.error),
    )
