"""ReAct Agent nodes for the LangGraph multi-agent workflow (Sprint 2 / Video 5-6; Week 5).

Implements LLM **nodes** invoked by the coordinator-based graph in ``graph.py``:

- **product_qa_agent**: Product Q&A using retrieval tools (``get_formatted_items_context``,
  ``get_formatted_reviews_context``). Emits ``tool_calls`` until it can set ``final_answer``.
- **shopping_cart_agent**: Cart add/remove/get via Postgres tools; ``user_id`` / ``cart_id``
  come from graph state (typically equal to ``thread_id`` for the Streamlit session).
- **warehouse_manager_agent**: Warehouse availability and reservation (Week 5 capstone path).
  Tools hit ``warehouses.inventory``; see ``tools.py`` and ``scripts/sql/warehouse_management.sql``.
- **coordinator_agent**: Entry planner. Chooses ``next_agent`` and a ``plan``; when it has a
  direct reply (e.g. off-topic), it sets ``final_answer`` and may append an ``AIMessage``.

**Message shaping:** Each node builds OpenAI-style message dicts with
``langchain_core.messages.convert_to_openai_messages`` per LangChain message. That keeps
tool call / tool result pairs in the shape the chat API expects (and avoids subtle bugs
from hand-rolled history).

**Structured outputs:** ``instructor`` constrains completions to Pydantic models (tool_calls,
final_answer, references, etc.) so the graph can route deterministically.

**Observability:** ``@traceable`` sends spans to LangSmith; ``logging`` INFO lines summarize
each node output for Docker logs (iteration, tools, coordinator message tail types).

Course refs: Week 5 notebooks (coordinator, cart, warehouse), Sprint 2 multi-agent videos.
"""
import logging
import instructor
from pathlib import Path
from typing import List

from jinja2 import Template
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langsmith import traceable,get_current_run_tree
from openai import OpenAI

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Path to prompts dir; uses __file__ so it works in both local and Docker (Video 6 fix)
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _summarize_messages_for_log(messages) -> str:
    """Build a short, privacy-safe summary of recent chat history for logs.

    We log only *types* (e.g. user, ai, tool) and roles, never message bodies. That is
    enough to verify ordering: an ``ai`` message with ``tool_calls`` should be followed by
    ``tool`` messages before the next ``ai`` turn—otherwise the provider returns 400.
    """
    if not messages:
        return "n=0"
    parts = []
    for m in messages[-6:]:
        t = getattr(m, "type", None)
        if t is None and isinstance(m, dict):
            t = m.get("role")
        if t is None:
            t = type(m).__name__
        parts.append(str(t))
    return f"n={len(messages)} tail=[{'|'.join(parts)}]"

# Lazy-initialized instructor client (defers API key check to first request)
_instructor_client = None


def _get_instructor_client():
    """Reuse instructor client across requests (avoids per-call instantiation)."""
    global _instructor_client
    if _instructor_client is None:
        _instructor_client = instructor.from_openai(OpenAI())
    return _instructor_client


# --- Pydantic models for structured LLM outputs (Instructor enforces these schemas) ---
# Learning: Instructor (instructor.from_openai) constrains the LLM to return JSON matching
# these Pydantic models. Without it, the LLM might return free text; we need structured
# tool_calls and references for the graph to route correctly.
class ToolCall(BaseModel):
    """Tool call from agent: name + arguments. Agent returns these when it wants to use a tool."""
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class ProductQAAgentResponse(BaseModel):
    """
    Structured output from agent_node. Instructor ensures LLM conforms to this schema.
    - tool_calls: Non-empty when agent wants to call tools (e.g. get_formatted_context).
    - final_answer: True when agent has enough info to answer (no more tool calls).
    - references: Product IDs used in the answer (for enrichment in rag_agent_wrapper).
    """
    answer: str = Field(description="The answer to the question")
    tool_calls: list[ToolCall] = []
    final_answer: bool = False
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    tool_calls: List[ToolCall] = []
    final_answer: bool = False

class Delegation(BaseModel):
    agent: str
    task: str

class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: List[Delegation]
    final_answer: bool = False
    answer: str = ""

class WarehouseManagerAgentResponse(BaseModel):
    """Instructor schema for the warehouse specialist (same shape as other ReAct agents).

    The model must call ``check_warehouse_availability`` before ``reserve_warehouse_items``
    (enforced by prompt + evaluation, not by this class). ``references`` are omitted here
    because this agent does not emit product QA-style citation lists.
    """
    answer: str = Field(description="The answer to the question")
    tool_calls: List[ToolCall] = []
    final_answer: bool = False





# --- Agent Node: ReAct loop LLM (Video 5) ---
@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def product_qa_agent(state) -> dict:
    """
    ReAct agent node: receives messages + tool descriptions, returns answer and/or tool_calls.
    If tool_calls non-empty -> graph routes to tool_node. If final_answer -> graph ends.
    """
    template = prompt_template_config(str(_PROMPTS_DIR / "product_qa_agent.yaml"), "product_qa_agent")
    prompt = template.render(available_tools=state.product_qa_agent.available_tools)

    messages = state.messages

    # One OpenAI-format dict per LangChain message (handles tool_calls + ToolMessage correctly).
    conversation = [convert_to_openai_messages(m) for m in messages]

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ProductQAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }



    ai_message = format_ai_message(response)
    tool_names = [tc.name for tc in response.tool_calls]
    # INFO for docker-compose: correlates with graph edge logs and LangSmith spans.
    logger.info(
        "agent product_qa_agent out iteration=%s final_answer=%s tool_calls=%s tools=%s references=%s",
        state.product_qa_agent.iteration + 1,
        response.final_answer,
        len(response.tool_calls),
        tool_names,
        len(response.references),
    )

    return {
        "messages": [ai_message],
            "product_qa_agent": {
                "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
                "iteration": state.product_qa_agent.iteration + 1,
                "final_answer": response.final_answer,
                "available_tools": state.product_qa_agent.available_tools
            },
            "answer": response.answer,
            "references": response.references
        }


@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent(state) -> dict:
    """
    Shopping cart agent node: handles add/remove/get cart requests.

    Loads prompt from shopping_cart_agent.yaml with user_id and cart_id for session context.
    Returns tool_calls when the user wants to add/remove items; final_answer when confirming.
    Graph routes to shopping_cart_agent_tool_node when tool_calls non-empty.
    """
    template = prompt_template_config(str(_PROMPTS_DIR / "shopping_cart_agent.yaml"), "shopping_cart_agent")

    prompt = template.render(
        available_tools=state.shopping_cart_agent.available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    messages = state.messages
    conversation = [convert_to_openai_messages(m) for m in messages]

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )
    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    ai_message = format_ai_message(response)
    tool_names = [tc.name for tc in response.tool_calls]
    logger.info(
        "agent shopping_cart_agent out iteration=%s final_answer=%s tool_calls=%s tools=%s user_id=%s cart_id=%s",
        state.shopping_cart_agent.iteration + 1,
        response.final_answer,
        len(response.tool_calls),
        tool_names,
        state.user_id,
        state.cart_id,
    )

    return {
        "messages": [ai_message],
        "shopping_cart_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.shopping_cart_agent.available_tools
        },
        "answer": response.answer,
    }




@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def warehouse_manager_agent(state) -> dict:
    """Warehouse ReAct node: check availability, then reserve, then natural-language summary.

    State includes ``warehouse_manager_agent.available_tools`` (JSON tool specs for the prompt).
    The graph routes to ``warehouse_manager_agent_tool_node`` when ``tool_calls`` is non-empty;
    see ``warehouse_manager_agent_tool_edge`` in ``graph.py`` (order of checks matters).
    """
    template = prompt_template_config(str(_PROMPTS_DIR / "warehouse_manager_agent.yaml"), "warehouse_manager_agent")

    prompt = template.render(
        available_tools=state.warehouse_manager_agent.available_tools,
    )

    messages = state.messages
    conversation = [convert_to_openai_messages(m) for m in messages]

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=WarehouseManagerAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    ai_message = format_ai_message(response)
    tool_names = [tc.name for tc in response.tool_calls]
    logger.info(
        "agent warehouse_manager_agent out iteration=%s final_answer=%s tool_calls=%s tools=%s",
        state.warehouse_manager_agent.iteration + 1,
        response.final_answer,
        len(response.tool_calls),
        tool_names,
    )

    return {
        "messages": [ai_message],
        "warehouse_manager_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.warehouse_manager_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.warehouse_manager_agent.available_tools
        },
        "answer": response.answer,
    }





@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def coordinator_agent(state):
    """
    Coordinator agent node: entry point that plans and delegates to specialist agents.

    Chooses among **product_qa_agent**, **shopping_cart_agent**, and **warehouse_manager_agent**
    based on the latest user intent. Returns ``next_agent`` and ``plan``; when it responds
    directly (``final_answer``), it may emit a short ``AIMessage``—otherwise specialists
    append the user-visible text.

    ``trace_id`` is copied from the active LangSmith run when present; else ``""`` so the
    API always returns a string (Streamlit feedback endpoint).
    """
    prompt_template = prompt_template_config(str(_PROMPTS_DIR / "coordinator_agent.yaml"), "coordinator_agent")
    prompt = prompt_template.render()

    messages = state.messages
    conversation = [convert_to_openai_messages(m) for m in messages]

    logger.info(
        "agent coordinator_agent in messages=%s openai_messages=%s",
        _summarize_messages_for_log(messages),
        len(conversation),
    )

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=CoordinatorAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    # Root trace id for human feedback (submit_feedback); missing when tracing is off.
    trace_id = ""
    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    plan_agents = [d.agent for d in response.plan]
    logger.info(
        "agent coordinator_agent out iteration=%s final_answer=%s next_agent=%r plan_len=%s plan_agents=%s",
        state.coordinator_agent.iteration + 1,
        response.final_answer,
        response.next_agent,
        len(response.plan),
        plan_agents,
    )


    # Coordinator only adds a message when it has a final answer (e.g. off-topic response).
    # When delegating, the specialist agent will add the substantive message.
    if response.final_answer:
        ai_message = [AIMessage(content=response.answer,)]
    else:
        ai_message = []

    return {
        "messages": ai_message,
        "answer": response.answer,
        "coordinator_agent": {
          "iteration": state.coordinator_agent.iteration +1,
          "final_answer": response.final_answer,
          "next_agent": response.next_agent,
          "plan": [data.model_dump() for data in response.plan]

        },
        "trace_id": trace_id
    }
