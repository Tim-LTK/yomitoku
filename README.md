# yomitoku-api

FastAPI backend for **Yomitoku** (Phase 1 — Scan & Explain). AI calls are server-side only; the mobile app never holds `ANTHROPIC_API_KEY`.

## Setup

```bash
cd yomitoku-api
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# set ANTHROPIC_API_KEY in .env
```

## Run

```bash
uvicorn yomitoku_api.main:app --reload --app-dir src
```

## Layout

- `src/yomitoku_api/` — application package (`main.py`, `routers/`, `services/`)
- `src/prompts/` — versioned `.txt` prompts (never inline prompt bodies in code)

## Environment

See `.env.example`. Only the API URL is public to the app; the Anthropic key stays on Railway / your server.
