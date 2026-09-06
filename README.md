# GPT Arena — Local LLM Ensemble

> **FastAPI prototype that aggregates ten Ollama candidate responses and a synthesis response**

GPT Arena serves a simple static interface and a JSON API. A `/generate`
request runs the configured arena model slots concurrently, collects their
results and asks a configured synthesis model for a consolidated answer. The
default configuration targets a local Ollama instance.

## Architecture

```text
app.py          FastAPI routes and request/response models
config.py       model tags, sampling grid and generation limits
ensemble.py     concurrent model invocation and synthesis flow
static/         browser interface
tests/          API and ensemble tests
```

## Requirements

- Python 3;
- a running Ollama service, by default `http://localhost:11434`;
- locally installed model tags matching `config.py`, or a deliberately updated
  local configuration.

## Installation and run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

The interface is served at `/`; `/health` returns a liveness response and
`POST /generate` accepts a prompt plus optional model and token-limit
overrides. The API enforces exactly ten arena model entries.

## Configuration

Set `OLLAMA_BASE_URL` in the environment to override the default endpoint.
Model slots, synthesis model, concurrency and token limits are defined in
`config.py`. The default concurrency is five; choose a value appropriate to
the available local resources.

## Limitations

An ensemble can still repeat errors, hallucinate, time out or exhaust local
resources. A synthesized response is not independently verified merely because
several candidates were generated. Review outputs, especially for factual,
safety-sensitive or consequential use.

## Licence

No root licence file is tracked.
