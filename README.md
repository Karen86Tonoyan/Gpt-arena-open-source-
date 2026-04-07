# GPT Arena – LLM Ensemble

> **"Ładujemy 10×7B by dostawać różne odpowiedzi a na koniec 20B robi z nich ścianę zgodnych odpowiedzi."**
>
> *We run 10 × 7 B-parameter models to collect diverse answers, then a 20 B model synthesises them into one comprehensive final response.*

---

## Architecture

```
User prompt
    │
    ▼
┌───────────────────────────────────────────────────────┐
│                   Arena Stage (parallel)              │
│  slot 0  slot 1  slot 2  …  slot 9  ← 10 × 7 B model │
│  temp=0.3 temp=0.5 temp=0.7 …  temp=1.2              │
└───────────────────────┬───────────────────────────────┘
                        │ 10 candidate answers
                        ▼
              ┌─────────────────────┐
              │  Synthesis Stage    │
              │   1 × 20 B model   │
              └─────────┬───────────┘
                        │ one consolidated "wall" answer
                        ▼
                   Final response
```

1. **Arena** – 10 slots run the same (or different) 7 B-class model with varied
   temperature / top-p settings so each answer takes a slightly different angle.
2. **Synthesis** – a single 20 B-class model receives all valid arena answers plus
   the original question and produces one well-structured, internally consistent response.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| [Ollama](https://ollama.com) | latest |

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. Pull models with Ollama

```bash
# 7 B arena model (repeat for each model tag you list in config.py)
ollama pull llama3.2:latest

# 20 B synthesis model
ollama pull llama3.1:latest
```

### 2. Configure models (optional)

Edit `config.py` to change which Ollama model tags are used:

```python
ARENA_MODELS = ["mistral:7b"] * 10        # 10 arena slots (same or different models)
SYNTHESIS_MODEL = "llama3.1:latest"        # must be ≥ 20 B for best results
```

### 3. Start the server

```bash
uvicorn app:app --reload
```

Open **http://localhost:8000** in your browser.

---

## API

### `POST /generate`

```json
{
  "prompt": "Explain quantum entanglement in simple terms.",
  "arena_max_tokens": 512,
  "synthesis_max_tokens": 2048
}
```

**Response:**

```json
{
  "question": "Explain quantum entanglement…",
  "arena_answers": [
    { "slot": 0, "model": "llama3.2:latest", "temperature": 0.3, "text": "…", "error": null },
    "…"
  ],
  "synthesis": "…consolidated final answer…",
  "arena_success_count": 10
}
```

### `GET /health`

Returns `{"status": "ok"}`.

---

## Running tests

```bash
pytest
```

---

## Project layout

```
├── app.py          # FastAPI application
├── ensemble.py     # Core pipeline (arena + synthesis)
├── config.py       # Model tags, sampling parameters, prompt template
├── requirements.txt
├── pytest.ini
├── static/
│   └── index.html  # Web UI
└── tests/
    ├── test_ensemble.py
    └── test_app.py
```
