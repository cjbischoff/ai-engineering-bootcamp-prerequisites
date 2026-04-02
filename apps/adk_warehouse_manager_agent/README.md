# ADK warehouse manager app

Standalone **Google Agent Development Kit (ADK)** app that mirrors the warehouse logic from `notebooks/week6/02-Warehouse-Agent-ADK.ipynb`, runnable with **`adk web`** and the Dev UI.

## Layout

```text
apps/adk_warehouse_manager_agent/          ← run `adk web` from HERE (agents root)
  warehouse_manager_agent/                 ← Python package (app name)
    __init__.py
    agent.py                               ← defines root_agent
    tools.py                               ← Postgres warehouse tools
    .env                                   ← local only (gitignored pattern)
```

ADK expects `<agents_dir>/warehouse_manager_agent/agent.py` with **`root_agent`**. Do **not** pass only `warehouse_manager_agent/` as the agents root (one level too deep).

## Run the web UI

From the repo:

```bash
cd apps/adk_warehouse_manager_agent
uv run adk web --port 8010
```

If `.env` under `warehouse_manager_agent/` uses **1Password `op://` references**, inject before starting:

```bash
cd apps/adk_warehouse_manager_agent
op run --env-file="warehouse_manager_agent/.env" -- uv run adk web --port 8010
```

## Dependencies

- **Docker**: Postgres with `tools_database` and `warehouses` schema (same as bootcamp Week 5 warehouse notebooks), port **5433** on the host.
- **Python**: `google-adk` (see root `pyproject.toml` dev group), `psycopg2-binary`, `litellm`.

## Session storage

ADK may create **`warehouse_manager_agent/.adk/`** (session DB, etc.). That path is **gitignored**; it is local runtime state, not curriculum source.

## Related

- Week 6 notebook: `notebooks/week6/02-Warehouse-Agent-ADK.ipynb`
- Shared tool ideas: `notebooks/week6/utils/tools.py`
