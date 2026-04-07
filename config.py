"""
Configuration for GPT Arena – LLM Ensemble System.

Architecture:
  • 10 × 7 B "arena" models run in parallel with varied sampling parameters
    to produce a diverse set of candidate answers.
  • 1 × 20 B "synthesis" model reads all candidate answers and produces a
    single consolidated, high-quality response (the "wall").
"""

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Ollama base URL (override with OLLAMA_BASE_URL env-var if needed)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Arena (7 B) model pool
# Each entry is an Ollama model tag.  The same model can appear multiple
# times – each slot gets different sampling parameters so answers diverge.
# ---------------------------------------------------------------------------
ARENA_MODELS: List[str] = [
    "llama3.2:latest",  # slots 0-9 – adjust to whatever 7 B models you have
    "llama3.2:latest",
    "llama3.2:latest",
    "llama3.2:latest",
    "llama3.2:latest",
    "llama3.2:latest",
    "llama3.2:latest",
    "llama3.2:latest",
    "llama3.2:latest",
    "llama3.2:latest",
]

# Synthesis (20 B) model tag
SYNTHESIS_MODEL: str = "llama3.1:latest"   # swap for any ≥20 B model you have

# ---------------------------------------------------------------------------
# Sampling-parameter grid applied across the 10 arena slots.
# Having different temperatures / top-p values forces answer diversity.
# ---------------------------------------------------------------------------
ARENA_PARAMS: List[dict] = [
    {"temperature": 0.3, "top_p": 0.85},
    {"temperature": 0.5, "top_p": 0.90},
    {"temperature": 0.7, "top_p": 0.90},
    {"temperature": 0.9, "top_p": 0.92},
    {"temperature": 1.0, "top_p": 0.95},
    {"temperature": 1.1, "top_p": 0.95},
    {"temperature": 0.4, "top_p": 0.80},
    {"temperature": 0.6, "top_p": 0.88},
    {"temperature": 0.8, "top_p": 0.93},
    {"temperature": 1.2, "top_p": 0.97},
]

# Generation limits
ARENA_MAX_TOKENS: int = 512
SYNTHESIS_MAX_TOKENS: int = 2048

# How many arena calls to run in parallel (keep ≤ your GPU VRAM headroom)
MAX_CONCURRENT_ARENA: int = 5

# ---------------------------------------------------------------------------
# Synthesis prompt template
# ---------------------------------------------------------------------------
SYNTHESIS_PROMPT_TEMPLATE: str = """\
You are a senior analytical assistant. Below you will find {n} distinct answers \
to the same question, produced by different language models.

Your task:
1. Identify points of agreement and disagreement across the answers.
2. Combine the best insights from every answer.
3. Produce ONE comprehensive, well-structured response that is strictly accurate \
and internally consistent.  Do NOT mention that you are summarising other models' \
outputs – write as if answering the question yourself.

--- ORIGINAL QUESTION ---
{question}

--- CANDIDATE ANSWERS ---
{answers}

--- YOUR CONSOLIDATED ANSWER ---"""

@dataclass
class GenerationRequest:
    prompt: str
    arena_models: List[str] = field(default_factory=lambda: ARENA_MODELS)
    arena_params: List[dict] = field(default_factory=lambda: ARENA_PARAMS)
    synthesis_model: str = SYNTHESIS_MODEL
    arena_max_tokens: int = ARENA_MAX_TOKENS
    synthesis_max_tokens: int = SYNTHESIS_MAX_TOKENS
    max_concurrent: int = MAX_CONCURRENT_ARENA
