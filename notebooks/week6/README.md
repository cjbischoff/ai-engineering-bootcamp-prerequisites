# Week 6 notebooks — LiteLLM and provider-agnostic LLM calls

## Purpose

Week 6 extends the multi-agent shopping assistant with **LiteLLM**: one Python API surface for multiple LLM vendors (OpenAI, Groq, etc.), combined with **Instructor** for structured (Pydantic) coordinator outputs.

## Files in this folder

| Path | Role |
|------|------|
| `01-litellm-router.ipynb` | Imports, OpenAI baseline coordinator, LiteLLM + Instructor intro (`SimpleResponse`), coordinator with **multi-model try/fallback** via `instructor.from_litellm(completion)`. |
| `utils/` | Local copy of Week 5-style helpers (`format_ai_message`, `get_tool_descriptions`, retrieval/cart/warehouse **tools**) so notebooks run when `cwd` is the repo root. See `utils/README.md`. |

## How it fits the curriculum

- **Week 5:** Coordinator + `instructor.from_openai(OpenAI())`.
- **Week 6:** Swap the completion backend to **`litellm.completion`** via **`instructor.from_litellm`**, use **`provider/model`** strings, and optionally loop models for resilience.

## Environment

- Dev dependencies (including `litellm`) install with `uv sync --all-groups`.
- API keys live in `.env` (never commit). LiteLLM reads standard provider env vars.

## Editor / type checking

Root **`pyrightconfig.json`** and **`[tool.basedpyright]`** in `pyproject.toml` point Pyright at **`.venv`**, which reduces false **“import could not be resolved”** diagnostics in notebooks.

## Commit hygiene

Run from repo root:

```bash
make clean-notebook-outputs
```

Then follow the bootcamp **pre-commit** checklist (educational comments, READMEs, signed conventional commits).
