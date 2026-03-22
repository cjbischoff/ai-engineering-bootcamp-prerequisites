# Week 6 `utils/` — notebook-local helpers

## Why this folder exists

`01-litellm-router.ipynb` is developed alongside **Week 5–equivalent** utilities so learners can run:

```python
from utils.utils import format_ai_message, get_tool_descriptions
```

whether Jupyter’s current working directory is the **repository root** or **`notebooks/week6`**. The first notebook cell prepends `notebooks/week6` to `sys.path` when `./utils` is not already visible from `cwd`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker + module docstring; no re-exports (explicit imports keep dependencies obvious in notebooks). |
| `utils.py` | **`format_ai_message`** — builds `AIMessage` with **unique `tool_call` ids** per turn (required for multi-turn OpenAI-compatible APIs). **`get_tool_descriptions` / `parse_function_definition`** — introspect Python tool functions into JSON-schema-like metadata for prompts. |
| `tools.py` | Same retrieval, cart, and warehouse tools as Week 5 notebooks: hybrid Qdrant search, reviews, `tools_database` cart CRUD, warehouse availability/reservation. Used when you wire agents that need real retrieval side effects. |

## Relationship to Week 5

The content intentionally **mirrors** `notebooks/week5/utils` so Sprint 5 can focus on **LiteLLM routing** without re-teaching retrieval SQL. If you fix a bug in one tree, consider duplicating the fix in the other for consistency.

## Curriculum links

- **Hybrid search / RRF:** Week 2 Video 5 (`Prefetch`, `FusionQuery`, BM25 + dense).
- **Structured outputs / tools:** Week 2–3 notebooks, Week 5 coordinator.
- **LiteLLM provider strings:** Week 6 notebook + [LiteLLM providers](https://docs.litellm.ai/docs/providers).

## Operational notes

- Qdrant and Postgres URLs in `tools.py` use **`localhost`** — match your Docker port mappings when running tools from the host (same as earlier sprint notebooks).
- LangSmith `@traceable` decorators annotate embedding/retrieval spans for observability.
