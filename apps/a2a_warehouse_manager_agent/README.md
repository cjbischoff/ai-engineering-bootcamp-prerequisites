# A2A warehouse manager agent

## Purpose (Week 6 / Sprint 5)

This app turns the **same warehouse ADK agent** you built in `02-Warehouse-Agent-ADK.ipynb` into a **remote [Agent2Agent (A2A)](https://github.com/google-a2a/A2A)** service. Other processes (or notebooks) can call it over HTTP using the **`a2a-sdk`** client instead of importing Python tools directly.

**Why:** A2A standardizes discovery (`AgentCard`), capabilities, and streaming message exchange so agents can interoperate across languages and deployments—similar in spirit to “OpenAPI for agents,” focused on agent-to-agent workflows.

## Layout

| File / directory | Role |
|------------------|------|
| **`appy.py`** | Process entry: builds `AgentCard`, wires `Runner` + `WarehouseManagerAgentExecutor`, serves JSON-RPC + SSE via `A2AStarletteApplication` and Uvicorn. |
| **`warehouse_manager_agent/agent.py`** | ADK `Agent` + `LiteLlm` + warehouse **tools** (same behavioral contract as the ADK notebook). |
| **`warehouse_manager_agent/tools.py`** | Postgres-backed `check_warehouse_availability` / `reserve_warehouse_items` against `tools_database` (Week 5 Docker Postgres on `localhost:5433`). |
| **`warehouse_manager_agent/agent_executor.py`** | Implements A2A’s `AgentExecutor`: maps A2A `Part` ↔ GenAI `types.Content`, drives `Runner.run_async`, pushes artifacts/status through `TaskUpdater`. |

## How the pieces connect

1. A client (e.g. `notebooks/week6/03-Warehouse-Agent-A2A.ipynb`) uses **`ClientFactory.connect`** with the server `BASE_URL`.
2. The server’s **`DefaultRequestHandler`** receives `message/stream`, delegates to **`WarehouseManagerAgentExecutor.execute`**.
3. The executor converts the user message, **`Runner`** runs the ADK agent (LLM + tools), and streaming updates flow back as A2A events until a final artifact is emitted.

## Run (secrets)

If `.env` uses **`op://…`** references for `OPENAI_API_KEY`, start the server **through 1Password injection** so LiteLLM sees a real key (literal `op://` strings cause OpenAI `401`):

```bash
# From repo root (recommended)
make run-a2a-warehouse-agent
```

Equivalent manual form:

```bash
cd apps/a2a_warehouse_manager_agent && op run --env-file="../../.env" -- uv run appy.py
```

(Adjust `--env-file` if your `.env` lives elsewhere; **`$(CURDIR)/.env`** is what the root `Makefile` uses.)

**Jupyter:** Use **`make jupyter-lab`** and attach Cursor to that server so notebook cells resolve the same secrets.

## Defaults

- **`HOST` / `PORT`:** from environment, default `localhost` / **`10000`** (`appy.py`).
- **Well-known agent card URL** must match what `A2ACardResolver` expects for your server version (if discovery fails, compare paths with server access logs).

## Curriculum tie-in

- **Notebook `02-Warehouse-Agent-ADK`:** in-process ADK + tools.
- **Notebook `03-Warehouse-Agent-A2A`:** remote client to **this** server.
- **`apps/adk_warehouse_manager_agent/`** (if present): alternate layout for `adk web`; this **`a2a_*`** app adds the **networked A2A** surface on top of the same domain logic.
