# Scripts

Utility scripts for testing and infrastructure verification. All are intended to be run from the **project root** via `uv run` or the `Makefile`.

## What to run when

| When | Command | What a good result tells you |
|------|---------|------------------------------|
| Stack just started | `make health` | Containers up, **hybrid Qdrant collection** has data, API is real FastAPI (OpenAPI), ports open. |
| CI / must verify Postgres | `make health-strict` | Same as health, but **fails** if `psycopg` is missing (Postgres was not actually checked). |
| Agent + retrieval + LLM | `make smoke-test` | **`POST /agent/`** completes with `final_answer`, non-empty answer, product context, sane latency. |
| Debug agent without UI | `make test-agent` or `uv run scripts/test_agent.py --query "..."` | Same endpoint as smoke test; prints validation + **wall time** to `final_answer`. |
| LangGraph DB only | `make smoke-test-postgres` | `langgraph_db` accepts connections; **`checkpoints` row count** (0 before first conversation is OK). |
| Week 5 cart schema | `make smoke-test-shopping-cart` | `tools_database` + table + columns; **row count** is informational. |
| MCP HTTP servers | `make smoke-test-mcp` | Ports **8001/8002**, HTTP up, **`/mcp` reachable** (does not call tools). |

## Files

| Script | Purpose |
|--------|---------|
| `health_check.py` | Docker `compose ps`, ports, **Amazon-items-collection-01-hybrid-search** (required), reviews collection (warning), legacy `collection-00` (info), Postgres, OpenAPI, MCP. **`--strict`** / **`make health-strict`**. |
| `smoke_test.py` | SSE **`/agent/`** end-to-end; measures **full stream** latency; **`--verbose`** for full JSON. |
| `smoke_test_mcp.py` | MCP ports + root + **`/mcp`** path. |
| `smoke_test_postgres.py` | `langgraph_db` connectivity + checkpoint table + optional row count. |
| `smoke_test_shopping_cart.py` | `tools_database` schema + **shopping_cart_items** + row count. |
| `test_agent.py` | Interactive-style agent test with **elapsed time** and optional **`--strict`**. |

## Usage

```bash
make health
make health-silent
make health-strict
make smoke-test
make smoke-test-verbose
make smoke-test-postgres
make setup-shopping-cart   # first-time Week 5 DB
make smoke-test-shopping-cart
make smoke-test-mcp
make test-agent QUERY="your question"
```

**1Password:** If `.env` uses `op://` references, start Docker with `make run-docker-compose` (Make uses `op` when the CLI is on `PATH`). Host-only scripts (`health`, `smoke-test`) do not need API keys unless you change them to call providers directly.

## Dependencies

Scripts use `uv run` and dependencies from the workspace root `pyproject.toml`. Run `uv sync` first (the Makefile targets do this for you).
