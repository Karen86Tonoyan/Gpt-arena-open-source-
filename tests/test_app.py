"""
tests/test_app.py – Tests for the FastAPI application layer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app import app
from ensemble import ArenaAnswer, EnsembleResult


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _make_result(synthesis: str = "Final answer") -> EnsembleResult:
    return EnsembleResult(
        question="Test question",
        arena_answers=[
            ArenaAnswer(slot=i, model="model-7b", temperature=0.5, text=f"Answer {i}")
            for i in range(10)
        ],
        synthesis=synthesis,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_generate_success(client: TestClient):
    with patch("app.run_ensemble", new_callable=AsyncMock) as mock_ens:
        mock_ens.return_value = _make_result("Paris is the capital of France.")
        resp = client.post("/generate", json={"prompt": "What is the capital of France?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["synthesis"] == "Paris is the capital of France."
    assert data["arena_success_count"] == 10
    assert len(data["arena_answers"]) == 10


def test_generate_rejects_empty_prompt(client: TestClient):
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422  # Pydantic validation error


def test_generate_rejects_wrong_arena_model_count(client: TestClient):
    with patch("app.run_ensemble", new_callable=AsyncMock):
        resp = client.post(
            "/generate",
            json={"prompt": "Hello", "arena_models": ["model-7b"] * 5},
        )
    assert resp.status_code == 422


def test_generate_503_when_all_arena_fail(client: TestClient):
    with patch("app.run_ensemble", new_callable=AsyncMock) as mock_ens:
        mock_ens.side_effect = ValueError("All arena slots failed")
        resp = client.post("/generate", json={"prompt": "Test"})
    assert resp.status_code == 502
    assert "All arena slots failed" in resp.json()["detail"]


def test_generate_counts_errors(client: TestClient):
    """arena_success_count must reflect only non-error arena slots."""
    result = EnsembleResult(
        question="Q",
        arena_answers=[
            ArenaAnswer(slot=i, model="m", temperature=0.5,
                        text="ok" if i < 7 else "", error="fail" if i >= 7 else None)
            for i in range(10)
        ],
        synthesis="answer",
    )
    with patch("app.run_ensemble", new_callable=AsyncMock) as mock_ens:
        mock_ens.return_value = result
        resp = client.post("/generate", json={"prompt": "Q"})

    assert resp.status_code == 200
    assert resp.json()["arena_success_count"] == 7
