# Week 6 notebooks — LiteLLM and provider-agnostic LLM calls

## Purpose

Week 6 extends the multi-agent shopping assistant with **LiteLLM**: one Python API surface for multiple LLM vendors (OpenAI, Groq, etc.), combined with **Instructor** for structured (Pydantic) coordinator outputs.

## Files in this folder

- **`01-litellm-router.ipynb`** — Imports, OpenAI baseline coordinator, LiteLLM + Instructor intro (`SimpleResponse`), coordinator with **multi-model try/fallback** via `instructor.from_litellm(completion)`.
- **`utils/`** — Local copy of Week 5-style helpers (`format_ai_message`, `get_tool_descriptions`, retrieval/cart/warehouse **tools**) so notebooks run when `cwd` is the repo root. See `utils/README.md`.

## How it fits the curriculum

- **Week 5:** Coordinator + `instructor.from_openai(OpenAI())`.
- **Week 6:** Swap the completion backend to **`litellm.completion`** via **`instructor.from_litellm`**, use **`provider/model`** strings, and optionally loop models for resilience.

## Environment

- Dev dependencies (including `litellm`) install with `uv sync --all-groups`.
- API keys: either **plaintext** vars in `.env` or **`op://vault/item/field`** references read by the 1Password CLI. LiteLLM and the OpenAI SDK expect standard names such as **`OPENAI_API_KEY`** (see `env.example`).
- **Docker:** From the repo root, `make run-docker-compose` injects secrets via `op` when the CLI is installed, so Compose can substitute `${OPENAI_API_KEY}` into containers (see root `docker-compose.yml`).

## Editor / type checking

Root **`pyrightconfig.json`** and **`[tool.basedpyright]`** in `pyproject.toml` point Pyright at **`.venv`**, which reduces false **“import could not be resolved”** diagnostics in notebooks.

**Instructor + Pyright:** `create_with_completion` may be typed as returning an optional structured object. The notebook coordinators use **`if response is None: raise RuntimeError(...)`** after the call (or after the multi-model loop) so `reportOptionalMemberAccess` is satisfied and you fail fast at runtime if no model succeeded.

## Verification from the host

After `make run-docker-compose` (or `op run --env-file=".env" -- make health`):

- **`make health`** — hybrid Qdrant collection, FastAPI OpenAPI, MCP ports (see `scripts/README.md`).
- **`make smoke-test`** — exercises **`POST /agent/`** end-to-end (not the older `/rag/`-only path).

## Commit hygiene

Run from repo root:

```bash
make clean-notebook-outputs
```

Then follow the bootcamp **pre-commit** checklist (educational comments, READMEs, signed conventional commits).
