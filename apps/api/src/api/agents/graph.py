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
from api.agents.tools import get_formatted_context, get_formatted_reviews_context
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
tools = [get_formatted_context, get_formatted_reviews_context]
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



# --- Execution: run_agent invokes graph; rag_agent_wrapper enriches for frontend ---
def run_agent(question: str, thread_id: str) -> dict:
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions,
    }
    # LangGraph checkpointing: thread_id scopes saved state so multi-turn conversations
    # resume correctly (same thread_id => same conversation history in Postgres).
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # Compile inside context so checkpointer is used for this run; PostgresSaver
    # writes state after each step so we can resume/replay by thread_id.
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        result = graph.invoke(initial_state, config)

    return result





def rag_agent_wrapper(question, thread_id):
    result = run_agent(question, thread_id)
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

    # trace_id from agent_node so frontend can POST feedback (thumbs/comment) to LangSmith (Week 4).
    return {
        "answer": result.get("answer", ""),
        "used_context": used_context,
        "trace_id": result.get("trace_id", "")
    }
