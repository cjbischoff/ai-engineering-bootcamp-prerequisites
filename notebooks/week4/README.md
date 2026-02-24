# Week 4: Multi-Turn Agents, Multiple Tools, MCP, Streaming

## Overview

Week 4 builds on Week 3's LangGraph ReAct agent with:
- **Multi-turn conversations** (LangGraph checkpointing, thread_id)
- **Multiple tools** (product retrieval + reviews retrieval)
- **Human feedback** (thumbs up/down, LangSmith)
- **MCP** (Model Context Protocol servers)
- **Streaming state** (SSE for real-time UI updates)

## Notebooks

| Notebook | Topics |
|----------|--------|
| 01-Multi-Turn-Agent | Checkpointing, thread_id, conversation state across turns |
| 02-Multiple-Tools | Second tool (reviews), Qdrant collection_exists check |
| 03-Human-Feedback | Thumbs feedback, submit_feedback API |
| 04-MCP | MCP servers (items, reviews), tool discovery |
| 05-Streaming-State | SSE streaming, status updates, final_answer event |

## Key Concepts

### Multi-Turn (01)
- **thread_id**: Stable ID per conversation; same ID = same checkpoint state
- **PostgresSaver**: Persists graph state to Postgres for resumable conversations
- **format_ai_message tool_call_id_prefix**: Unique IDs per turn to avoid OpenAI errors

### Multiple Tools (02)
- **get_formatted_items_context**: Product descriptions (hybrid search)
- **get_formatted_reviews_context**: Reviews filtered by item IDs
- **collection_exists**: Check before create_collection to avoid 409 Conflict

### Streaming (05)
- **SSE (Server-Sent Events)**: text/event-stream; yields "data: ...\n\n"
- **Status updates**: Plain text ("Analysing...", "Planning...") for UX
- **final_answer**: JSON event with answer, used_context, trace_id

## Utils

`notebooks/week4/utils/utils.py` mirrors week3 utils with `format_ai_message` and `tool_call_id_prefix` for multi-turn tool call IDs.

## Related

- [../week3/README.md](../week3/README.md) - LangGraph intro, single-turn agent
- [../../apps/api/src/api/agents/README.md](../../apps/api/src/api/agents/README.md) - Production graph, tools
- [../../apps/items_mcp_server/README.md](../../apps/items_mcp_server/README.md) - Items MCP server
- [../../apps/reviews_mcp_server/README.md](../../apps/reviews_mcp_server/README.md) - Reviews MCP server
