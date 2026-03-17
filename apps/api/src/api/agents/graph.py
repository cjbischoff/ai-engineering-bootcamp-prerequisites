"""
LangGraph coordinator-based multi-agent workflow (Sprint 2 / Video 5-6; Week 5).

Defines StateGraph: START -> coordinator_agent -> (product_qa_agent | shopping_cart_agent)
<-> tool_node -> back to coordinator. Flow:
- coordinator_agent: Entry point; plans tasks and delegates to product_qa_agent or
  shopping_cart_agent based on user intent.
- product_qa_agent: Answers product questions; uses get_formatted_items_context,
  get_formatted_reviews_context. Loops with tool_node until final_answer.
- shopping_cart_agent: Handles cart add/remove/get. Loops with tool_node until final_answer.
- Both agents return to coordinator when done; coordinator may delegate again or end.

rag_agent_stream_wrapper: Streams graph execution as SSE. Enriches references with
image_url/price from Qdrant. Yields status updates ("Planning...", "Looking for items...")
and final_answer JSON with answer, used_context, trace_id, shopping_cart.
"""
from qdrant_client import QdrantClient
import qdrant_client

# Reuse client across requests (avoids per-request connection overhead)
_QDRANT_CLIENT = QdrantClient(url="http://qdrant:6333")
from pydantic import BaseModel, Field
import numpy as np
import json
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import Annotated, Any, Dict, List

from operator import add
from langsmith import traceable, get_current_run_tree
from api.agents.agents import Delegation, ToolCall, RAGUsedContext, product_qa_agent, shopping_cart_agent, coordinator_agent
from langgraph.graph import StateGraph
from api.agents.tools import get_formatted_items_context, get_formatted_reviews_context, add_to_shopping_cart, remove_from_cart, get_shopping_cart
from api.agents.utils.utils import _sanitize_tool_name, get_tool_descriptions
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver  # Persist conversation state per thread (Week 4 multi-turn)


class AgentProperties(BaseModel):
    """Per-agent state: iteration count, final_answer flag, tool_calls from LLM."""
    iteration: int = 0
    final_answer: bool = False
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []

class CoordinatorAgentProperties(BaseModel):
    """Coordinator state: plan (delegations), next_agent to route to."""
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""

class State(BaseModel):
    """Graph state. Annotated[List, add] reducers append (messages, references) across nodes."""
    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(default_factory=CoordinatorAgentProperties)
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""


def product_qa_agent_tool_edge(state) -> str:
    """Route: product_qa_agent -> tools (if tool_calls) or end (if final_answer or max iterations)."""
    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"

def shopping_cart_agent_tool_edge(state) -> str:
    """Route: shopping_cart_agent -> tools (if tool_calls) or end (if final_answer or max iterations)."""
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def coordinator_agent_edge(state):
    """Route: coordinator_agent -> product_qa_agent, shopping_cart_agent, or end."""
    if state.coordinator_agent.iteration > 3:
        return "end"
    elif state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    else:
        return "end"


# --- Workflow: build StateGraph with coordinator-based routing ---
workflow = StateGraph(State)

# Product QA agent: retrieval tools for answering product questions
product_qa_agent_tools = [get_formatted_items_context, get_formatted_reviews_context]
product_qa_agent_tool_node = ToolNode(product_qa_agent_tools)
product_qa_agent_tool_descriptions = get_tool_descriptions(product_qa_agent_tools)

# Shopping cart agent: cart CRUD tools
shopping_cart_agent_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)
shopping_cart_agent_tool_descriptions = get_tool_descriptions(shopping_cart_agent_tools)

# Add nodes
workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)

workflow.add_node("coordinator_agent", coordinator_agent)

workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)


# Edges: START -> coordinator; coordinator conditionally routes to agents or END
workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,
    {
        "product_qa_agent": "product_qa_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "product_qa_agent",
    product_qa_agent_tool_edge,
    {
        "tools": "product_qa_agent_tool_node",
        "end": "coordinator_agent"
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_edge,
    {
        "tools": "shopping_cart_agent_tool_node",
        "end": "coordinator_agent"
    }
)

# Tool nodes loop back to their agent for next LLM call
workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")











@traceable(name="LangGraph", run_type="chain")
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
            raw = tool_call.get("name", "?") if isinstance(tool_call, dict) else getattr(tool_call, "name", "?")
            name = _sanitize_tool_name(raw) if isinstance(raw, str) else raw
            args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else getattr(tool_call, "arguments", {}) or {}
            if name == "get_formatted_items_context":
                return f"Looking for items: {args.get('query', '')}."
            elif name == "get_formatted_reviews_context":
                return "Fetching user reviews..."
            elif name == "add_to_shopping_cart":
                return "Adding items to cart..."
            elif name == "remove_from_cart":
                return "Removing item from cart..."
            elif name == "get_shopping_cart":
                return "Fetching cart..."
            else:
                return f"Unknown tool: {name}"

        # Map graph node starts to user-facing status text (Week 4 streaming UX)
        if _is_node_start(chunk):
            name = chunk[1].get("payload", {}).get("name")
            if name == "coordinator_agent":
                return "Planning..."
            elif name == "product_qa_agent":
                return "Analyzing the question..."
            elif name == "shopping_cart_agent":
                return "Managing cart..."
            elif name in ("product_qa_agent_tool_node", "shopping_cart_agent_tool_node"):
                payload = chunk[1].get("payload", {})
                input_data = payload.get("input", {})
                # input_data can be State (Pydantic) or dict; State has no .get()
                if isinstance(input_data, dict):
                    agent_key = "product_qa_agent" if name == "product_qa_agent_tool_node" else "shopping_cart_agent"
                    tool_calls = input_data.get(agent_key, {}).get("tool_calls", [])
                else:
                    agent = getattr(input_data, "product_qa_agent" if name == "product_qa_agent_tool_node" else "shopping_cart_agent", None)
                    tool_calls = getattr(agent, "tool_calls", []) if agent else []
                message = "".join([_tool_to_text(tc) for tc in tool_calls])
                return message
            else:
                return False

    qdrant_client = QdrantClient(url="http://qdrant:6333")


    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "user_id": thread_id,
        "cart_id": thread_id,
        "product_qa_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qa_agent_tool_descriptions,
            "tool_calls": []
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_agent_tool_descriptions,
            "tool_calls": []
        },
        "coordinator_agent": {
            "iteration": 0,
            "final_answer": False,
            "plan": [],
            "next_agent": ""
        }
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

    # Enrich each reference: agent returns (id, description); we fetch image/price from Qdrant.
    # dummy_vector: query_points needs a vector; we use filter by parent_asin so vector is unused.
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

    answer = result.get("answer", "")
    if not answer and used_context:
        answer = "Here are the recommended products based on your query."

    # LangSmith: root run ID for feedback. State has no trace_id; get from active trace.
    trace_id = ""
    current_run = get_current_run_tree()
    if current_run:
        root = current_run
        while root and root.parent_run:
            root = root.parent_run
        trace_id = str(root.id) if root else ""

    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [
        {
            "price": float(item.get("price")) if item.get("price") else None,
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": float(item.get("total_price")) if item.get("total_price") else None,
        }
        for item in shopping_cart
    ]

    yield _string_for_sse(json.dumps({
        "type": "final_answer",
        "data": {
            "answer": answer,
            "used_context": used_context,
            "trace_id": trace_id,
            "shopping_cart": shopping_cart_items

        },
    }))
