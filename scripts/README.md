# Scripts

Utility scripts for testing and infrastructure verification.

## Files

| Script | Purpose |
|--------|---------|
| `health_check.py` | Verifies Docker containers, ports, Qdrant collection, Postgres, MCP servers. Run: `make health` |
| `smoke_test.py` | End-to-end RAG pipeline test. Consumes SSE stream from `/rag/`, validates final_answer. Run: `make smoke-test` |
| `smoke_test_mcp.py` | Smoke test for MCP-based agent (04-MCP notebook). |
| `smoke_test_postgres.py` | Smoke test for Postgres/LangGraph checkpointer. |
| `smoke_test_shopping_cart.py` | Smoke test for shopping cart schema, table, columns (Week 5). Run: `make smoke-test-shopping-cart` |

## Usage

```bash
make health          # Full health check output
make health-silent  # Only show failures
make smoke-test     # RAG pipeline smoke test
make smoke-test-verbose  # Smoke test with full JSON
make setup-shopping-cart # Create tools_database + schema (Week 5, first-time)
make smoke-test-shopping-cart  # Shopping cart DB (schema, table, columns)
```

**Note:** Shopping cart uses `tools_database` (bootcamp spec). Run `make setup-shopping-cart` before the Week 5 notebook.

## Dependencies

Scripts use `uv run` and project dependencies from root `pyproject.toml`. Ensure `uv sync` has been run.
