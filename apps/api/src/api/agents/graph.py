"""
LangGraph ReAct agent workflow (Sprint 2 / Video 5-6).

Defines StateGraph: START -> intent_router -> agent_node <-> tool_node -> END.
- intent_router: Filters irrelevant queries (product/shopping = relevant; off-topic = not).
- agent_node: LLM that decides tool_calls or final_answer.
- tool_node: Executes get_formatted_context (retrieval tool).
- tool_router: Conditional edge agent_node -> tools or end.
run_agent() invokes the graph; rag_agent_wrapper() enriches result with images/prices.

rag_agent_wrapper behavior:
- used_context: Include every referenced product (image_url/price may be None) so API
  and smoke test get at least one product when the agent retrieves; matches rag_pipeline_wrapper.
- answer: Prefer state.answer; fallback to last assistant message content; if still empty
  but we have references, build a short summary so the response is never empty when products exist.
"""
from qdrant_client import QdrantClient

# Reuse client across requests (avoids per-request connection overhead)
_QDRANT_CLIENT = QdrantClient(url="http://qdrant:6333")
from pydantic import BaseModel
import numpy as np
import json
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import Annotated, Any

from operator import add
from api.agents.agents import ToolCall, RAGUsedContext, agent_node, intent_router_node
from langgraph.graph import StateGraph
from api.agents.tools import get_formatted_items_context, get_formatted_reviews_context
from api.agents.utils.utils import get_tool_descriptions
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver  # Persist conversation state per thread (Week 4 multi-turn)


# --- State: shared across all nodes; add reducer appends messages and references ---
class State(BaseModel):
    messages: Annotated[list[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    final_answer: bool = False
    references: Annotated[list[RAGUsedContext], add] = []
    trace_id: str = ""

def tool_router(state: State) -> str:
    """Conditional edge from agent_node: "tools" -> tool_node, "end" -> END."""
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def intent_router_conditional_edges(state: State) -> str:
    """Routes from intent_router: agent_node if relevant, END otherwise."""
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"

# --- Build graph: nodes + edges (Video 5 pattern) ---
workflow = StateGraph(State)

# Week 4 multiple tools: agent can call product retrieval and/or reviews retrieval.
# get_formatted_context = product descriptions; get_formatted_reviews_context = reviews filtered by item IDs.
tools = [get_formatted_items_context, get_formatted_reviews_context]
tool_node = ToolNode(tools)
tool_descriptions = get_tool_descriptions(tools)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END
    }
)

workflow.add_edge("tool_node", "agent_node")


def rag_agent_stream_wrapper(question: str, thread_id: str):
    """
    Stream LangGraph execution as SSE (Server-Sent Events) for real-time UI updates.

    Yields two event types:
    - Plain text: Human-readable status ("Analysing the question...", "Planning...",
      "Looking for items: X.") for frontend status placeholder.
    - JSON final_answer: {type, data: {answer, used_context, trace_id}} when graph completes.

    Why SSE? Enables progressive feedback (user sees "Planning..." before answer) instead
    of waiting for full response. Frontend consumes with fetch/EventSource or iter_lines().
    """
    def _string_for_sse(message: str):
        """Format message as SSE line: 'data: {message}\\n\\n' (required by SSE spec)."""
        return f"data: {message}\n\n"

    def _process_graph_event(chunk):
        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _is_node_end(chunk):
            return chunk[0] == "updates"

        def _tool_to_text(tool_call):
            if tool_call.name == "get_formatted_items_context":
                return f"Looking for items: {tool_call.arguments.get('query', '')}."
            elif tool_call.name == "get_formatted_reviews_context":
                return "Fetching user reviews..."
            else:
                return f"Unknown tool: {tool_call.name}"

        # Map graph node starts to user-facing status text (Week 4 streaming UX)
        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            elif chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            elif chunk[1].get("payload", {}).get("name") == "tool_node":
                payload = chunk[1].get("payload", {})
                input_data = payload.get("input", {})
                tool_calls = getattr(input_data, "tool_calls", []) if hasattr(input_data, "tool_calls") else input_data.get("tool_calls", [])
                message = "".join([_tool_to_text(tc) for tc in tool_calls])
                return message
            else:
                return False

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions,
    }

    config = {"configurable": {"thread_id": thread_id}}
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = None
        for chunk in graph.stream(
            initial_state,
            config=config,
            stream_mode=["debug", "values"],
        ):
            process_chunk = _process_graph_event(chunk)
            if process_chunk:
                yield _string_for_sse(process_chunk)

            if chunk[0] == "values":
                result = chunk[1]

    # Graph may not produce values if it exits early (e.g. off-topic); surface error to frontend
    if result is None:
        yield _string_for_sse(json.dumps({"type": "error", "data": {"message": "No result from graph"}}))
        return

    # Enrich references with image_url and price from Qdrant (same as non-streaming wrapper)
    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = _QDRANT_CLIENT.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=dummy_vector,
            limit=1,
            using="text-embedding-3-small",
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        ).points[0].payload

        image_url = payload.get("image")
        price = payload.get("price")

        if image_url:
            used_context.append({
                "image_url": image_url,
                "price": price,
                "description": item.description
            })

    yield _string_for_sse(json.dumps({
        "type": "final_answer",
        "data": {
            "answer": result.get("answer", ""),
            "used_context": used_context,
            "trace_id": result.get("trace_id", ""),
        },
    }))
