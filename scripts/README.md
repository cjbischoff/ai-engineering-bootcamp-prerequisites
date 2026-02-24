# Scripts

Utility scripts for testing and infrastructure verification.

## Files

| Script | Purpose |
|--------|---------|
| `health_check.py` | Verifies Docker containers, ports, Qdrant collection, Postgres, MCP servers. Run: `make health` |
| `smoke_test.py` | End-to-end RAG pipeline test. Consumes SSE stream from `/rag/`, validates final_answer. Run: `make smoke-test` |
| `smoke_test_mcp.py` | Smoke test for MCP-based agent (04-MCP notebook). |
| `smoke_test_postgres.py` | Smoke test for Postgres/LangGraph checkpointer. |

## Usage

```bash
make health          # Full health check output
make health-silent  # Only show failures
make smoke-test     # RAG pipeline smoke test
make smoke-test-verbose  # Smoke test with full JSON
```

## Dependencies

Scripts use `uv run` and project dependencies from root `pyproject.toml`. Ensure `uv sync` has been run.
