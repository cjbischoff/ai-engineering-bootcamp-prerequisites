# Week 3: LangGraph & ReAct Agents

Sprint 2 notebooks: LangGraph fundamentals, query expansion, routing, and tool-using ReAct agents.

## Overview

Week 3 introduces **LangGraph** for building agentic workflows. The notebooks progress from basic graphs to a full ReAct agent that hides retrieval behind a tool.

## Notebooks (Follow in Order)

| Notebook | Video | Focus |
|----------|-------|-------|
| `01-LangGraph-Intro.ipynb` | 1–2 | StateGraph basics, single node, tool-using agents (ReAct) |
| `02-Query-Rewriting.ipynb` | 3 | Query expansion, parallel retrieval, Send/Command |
| `03-Router.ipynb` | 4 | Intent router, conditional edges, irrelevant-query filtering |
| `04-Agent-Single-Turn.ipynb` | 5 | ReAct agent with retrieval tool, full graph |

## Prerequisites

- Qdrant running (`docker compose up -d`)
- API keys in `.env` (OpenAI)
- Product data indexed in Qdrant (Week 1 preprocessing)
- Run from project root; notebooks use `sys.path` to import `utils`

## Key Concepts

### LangGraph (Video 1–2)

- **StateGraph**: Nodes + edges; state flows between nodes.
- **ToolNode**: Executes tool calls from the agent.
- **ReAct**: Agent reasons and acts (tool calls) in a loop.

### Query Expansion (Video 3)

- Rewrite/expand query for better retrieval.
- Parallel retrieval with `Send`/`Command`.
- Not moved to backend; ReAct agent is preferred.

### Intent Router (Video 4)

- Classify query as relevant or not.
- Conditional edges: relevant → agent, irrelevant → end.
- Reduces unnecessary retrieval for off-topic queries.

### ReAct Agent (Video 5)

- **Retrieval as tool**: `get_formatted_context` instead of always-retrieve.
- Agent decides when and what to retrieve.
- `agent_node` + `tool_node` + `tool_router` loop.

## Utilities

`utils/utils.py` provides:

- `format_ai_message(response)`: AgentResponse → AIMessage (with tool_calls).
- `get_tool_descriptions(function_list)`: Parse docstrings into tool schemas.

## Backend Migration

Video 6 moves the ReAct agent to the backend:

- `apps/api/src/api/agents/agents.py` – agent_node, intent_router_node
- `apps/api/src/api/agents/graph.py` – StateGraph, run_agent, rag_agent_wrapper
- `apps/api/src/api/agents/tools.py` – get_formatted_context

See `apps/api/src/api/agents/README.md` for details.
