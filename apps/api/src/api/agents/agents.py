"""
ReAct Agent nodes for the LangGraph multi-agent workflow (Sprint 2 / Video 5-6; Week 5).

Implements three LLM nodes invoked by the coordinator-based graph in graph.py:
- product_qa_agent: Answers product questions using retrieval tools (get_formatted_items_context,
  get_formatted_reviews_context). Returns tool_calls when it needs to search; final_answer when done.
- shopping_cart_agent: Manages cart operations (add, remove, get) using cart tools. Receives
  user_id/cart_id from state for multi-session support.
- coordinator_agent: Entry point that plans tasks and delegates to product_qa_agent or
  shopping_cart_agent. Routes based on user intent (product questions vs cart actions).

Uses Instructor for structured outputs (Pydantic models). Each agent returns tool_calls,
final_answer, and agent-specific fields. LangSmith @traceable decorators enable observability.
"""
import instructor
from pathlib import Path
from typing import List

from jinja2 import Template
from langchain_core.messages import AIMessage
from langsmith import traceable,get_current_run_tree
from openai import OpenAI

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message, messages_to_openai
from pydantic import BaseModel, Field

# Path to prompts dir; uses __file__ so it works in both local and Docker (Video 6 fix)
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

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







# --- Agent Node: ReAct loop LLM (Video 5) ---
@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def product_qa_agent(state) -> dict:
    """
    ReAct agent node: receives messages + tool descriptions, returns answer and/or tool_calls.
    If tool_calls non-empty -> graph routes to tool_node. If final_answer -> graph ends.
    """
    template = prompt_template_config(str(_PROMPTS_DIR / "product_qa_agent.yaml"), "product_qa_agent")
    prompt = template.render(available_tools=state.product_qa_agent.available_tools)

    messages = state.messages

    conversation = messages_to_openai(messages)

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
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
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
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

    conversation = messages_to_openai(state.messages)

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    ai_message = format_ai_message(response)

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
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def coordinator_agent(state):
    """
    Coordinator agent node: entry point that plans and delegates to specialist agents.

    Analyzes conversation to decide: product_qa_agent (product questions) or
    shopping_cart_agent (cart add/remove/get). Returns next_agent and plan.
    When final_answer is True with empty plan, the graph ends.
    """
    prompt_template = prompt_template_config(str(_PROMPTS_DIR / "coordinator_agent.yaml"), "coordinator_agent")
    prompt = prompt_template.render()

    conversation = messages_to_openai(state.messages)

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=CoordinatorAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
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

        }
}
