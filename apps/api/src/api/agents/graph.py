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
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import Annotated, Any

from operator import add
from api.agents.agents import ToolCall, RAGUsedContext, agent_node, intent_router_node
from langgraph.graph import StateGraph
from api.agents.tools import get_formatted_context
from api.agents.utils.utils import get_tool_descriptions
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode


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

tools = [get_formatted_context]
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

graph = workflow.compile()


# --- Execution: run_agent invokes graph; rag_agent_wrapper enriches for frontend ---
def run_agent(question: str) -> dict:
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions,
    }
    result = graph.invoke(initial_state)
    return result





def rag_agent_wrapper(question):
    """
    Entry point for /rag/ endpoint. Runs agent, then enriches references with
    image_url and price from Qdrant (same pattern as rag_pipeline_wrapper).

    used_context: We append every referenced product even when image_url/price are None,
    so the API and smoke test see products whenever the agent used retrieval (avoids
    empty used_context when Qdrant payload lacks images).

    answer: We use state.answer, then fallback to last assistant message content
    (state.answer can be empty when the agent returns references), then a short
    summary from used_context so the response is never empty when products exist.
    """
    result = run_agent(question)

    used_context = []
    # Dummy vector for filter-only query (we only need payload by parent_asin)
    dummy_vector = np.zeros(1536).tolist()

    # Enrich each reference with image/price from Qdrant; include all references (image/price may be None)
    for item in result.get("references", []):
        points_result = _QDRANT_CLIENT.query_points(
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
        )
        if not points_result.points:
            continue
        payload = points_result.points[0].payload

        image_url = payload.get("image")
        price = payload.get("price")
        # Include every referenced product so used_context is non-empty when agent retrieved (smoke test / frontend)
        used_context.append({
            "image_url": image_url,
            "price": price,
            "description": item.description,
        })

    # Answer resolution: 1) state.answer 2) last assistant/AI message content 3) summary from used_context
    answer = result.get("answer") or ""
    if not answer and result.get("messages"):
        for m in reversed(result["messages"]):
            role_ok = (m.get("role") == "assistant" if isinstance(m, dict) else getattr(m, "type", "").lower().endswith("ai") or getattr(m, "type", "") == "ai")
            if not role_ok:
                continue
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", None)
            if content is None:
                continue
            if isinstance(content, str) and content.strip():
                answer = content
                break
            if isinstance(content, list):
                parts = [c.get("text", c) if isinstance(c, dict) else str(c) for c in content]
                text = " ".join(p for p in parts if p).strip()
                if text:
                    answer = text
                    break
    if not answer and used_context:
        # Last resort: agent returned references but state.answer empty (e.g. Instructor left answer blank)
        answer = "Based on the products in our inventory: " + " ".join(
            (c.get("description") or "")[:100] for c in used_context[:3]
        ).strip() or "I found some relevant products; see the list below."
    return {
        "answer": answer,
        "used_context": used_context,
    }
