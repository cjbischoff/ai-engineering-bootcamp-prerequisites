import instructor
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import convert_to_openai_messages
from langsmith import traceable
from openai import OpenAI

from api.agents.utils.prompt_management import prompt_template_config

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
from api.agents.utils.utils import format_ai_message
from pydantic import BaseModel, Field
from typing import List


class ToolCall(BaseModel):
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class AgentResponse(BaseModel):
    answer: str = ""
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: List[RAGUsedContext] = []



class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def agent_node(state) -> dict:

    template = prompt_template_config(str(_PROMPTS_DIR / "qa_agent.yaml"), "qa_agent")

    prompt = template.render(
        available_tools=state.available_tools
    )

    messages = state.messages

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

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }


# Intent Router Node
@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def intent_router_node(state):
    query = ""
    if state.messages:
        m = state.messages[-1]
        query = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")

    template = prompt_template_config(
        str(_PROMPTS_DIR / "intent_router_agent.yaml"),
        "intent_router_agent",
    )
    prompt = template.render(query=query)

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=IntentRouterResponse,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query}],
        temperature=0.5,
    )

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
    }
