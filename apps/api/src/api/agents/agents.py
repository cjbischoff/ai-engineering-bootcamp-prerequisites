"""
ReAct Agent nodes: agent_node and intent_router_node.

Sprint 2 / Video 5-6: Implements the LLM nodes for the LangGraph workflow.
- agent_node: Main ReAct loop; decides whether to call tools or return final answer.
- intent_router_node: Filters irrelevant queries before invoking the agent.
Uses Instructor for structured outputs (AgentResponse, IntentRouterResponse).
"""
import instructor
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import convert_to_openai_messages
from langsmith import traceable,get_current_run_tree
from openai import OpenAI

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message
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

class AgentResponse(BaseModel):
    """
    Structured output from agent_node. Instructor ensures LLM conforms to this schema.
    - tool_calls: Non-empty when agent wants to call tools (e.g. get_formatted_context).
    - final_answer: True when agent has enough info to answer (no more tool calls).
    - references: Product IDs used in the answer (for enrichment in rag_agent_wrapper).
    """
    answer: str = ""
    tool_calls: list[ToolCall] = []
    final_answer: bool = False
    references: list[RAGUsedContext] = []


class IntentRouterResponse(BaseModel):
    """Intent router output: question_relevant=True routes to agent, False returns answer (e.g. clarification)."""
    question_relevant: bool
    answer: str


# --- Agent Node: ReAct loop LLM (Video 5) ---
@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def agent_node(state) -> dict:
    """
    ReAct agent node: receives messages + tool descriptions, returns answer and/or tool_calls.
    If tool_calls non-empty -> graph routes to tool_node. If final_answer -> graph ends.
    """
    template = prompt_template_config(str(_PROMPTS_DIR / "qa_agent.yaml"), "qa_agent")
    prompt = template.render(available_tools=state.available_tools)

    messages = state.messages

    # Convert LangGraph messages to OpenAI format for the LLM
    # Learning: LangGraph stores AIMessage, ToolMessage, HumanMessage; OpenAI API expects
    # {"role": "user"|"assistant"|"system", "content": "..."}. convert_to_openai_messages
    # handles the translation (including tool_calls if present).
    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
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



    # Convert AgentResponse to AIMessage (with tool_calls if present) for LangGraph
    # tool_call_id_prefix: unique per turn to avoid OpenAI BadRequestError (Week 4 multi-turn)
    ai_message = format_ai_message(response, tool_call_id_prefix=f"call_{state.iteration}")

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }


# --- Intent Router Node: filter irrelevant queries (Video 4) ---
@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def intent_router_node(state):
    """
    Classifies whether the user query is relevant to products in stock.
    Irrelevant -> return answer (e.g. "Please ask about products in stock").
    Relevant -> route to agent_node to process with tools.
    """
    query = ""
    if state.messages:
        m = state.messages[-1]
        query = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")

    template = prompt_template_config(
        str(_PROMPTS_DIR / "intent_router_agent.yaml"),
        "intent_router_agent",
    )
    prompt = template.render(query=query)

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=IntentRouterResponse,
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
            # Expose trace/run id so UI can send feedback (thumbs/comment) to LangSmith for this run (Week 4).
            trace_id = str(getattr(current_run, "trace_id", current_run.id))
    else:
        trace_id = None

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
        "trace_id": trace_id
    }
