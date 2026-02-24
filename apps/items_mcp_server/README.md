# Items MCP Server

Model Context Protocol (MCP) server exposing product retrieval as tools for AI agents.

## Purpose

MCP standardizes how agents discover and invoke tools across services. This server runs independently (port 8001 on host) so agents can call `get_formatted_items_context` via HTTP. Used by `notebooks/week4/04-MCP.ipynb`.

## Architecture

```
items_mcp_server/
├── src/items_mcp_server/
│   ├── main.py      # FastMCP app, @mcp.tool() get_formatted_items_context
│   ├── utils.py     # retrieve_items_data, process_items_context (Qdrant hybrid search)
│   └── core/config.py
├── Dockerfile
└── pyproject.toml
```

## Tool

- **get_formatted_items_context(query, top_k=5)**: Returns formatted product descriptions from `Amazon-items-collection-01-hybrid-search`. Same logic as `api.agents.tools.get_formatted_items_context`.

## Running

```bash
# Via Docker Compose (with api, streamlit, qdrant)
docker compose up items_mcp_server

# Standalone
uv run python -m items_mcp_server.main
```

Port: 8001 (host) → 8000 (container)
