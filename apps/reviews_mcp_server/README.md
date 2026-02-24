# Reviews MCP Server

Model Context Protocol (MCP) server exposing review retrieval as tools for AI agents.

## Purpose

Separate MCP server for reviews (vs items). Agents can call tools from multiple servers—items for products, reviews for customer feedback. Used by `notebooks/week4/04-MCP.ipynb`.

## Architecture

```
reviews_mcp_server/
├── src/reviews_mcp_server/
│   ├── main.py      # FastMCP app, @mcp.tool() get_formatted_reviews_context
│   ├── utils.py     # retrieve_reviews_data, process_reviews_context (Qdrant)
│   └── core/config.py
├── Dockerfile
└── pyproject.toml
```

## Tool

- **get_formatted_reviews_context(query, item_list, top_k=15)**: Returns formatted reviews from `Amazon-items-collection-01-reviews`, prefiltered by item IDs. Same logic as `api.agents.tools.get_formatted_reviews_context`.

## Running

```bash
# Via Docker Compose
docker compose up reviews_mcp_server

# Standalone
uv run python -m reviews_mcp_server.main
```

Port: 8002 (host) → 8000 (container)
