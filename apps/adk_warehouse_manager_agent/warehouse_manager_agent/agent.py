"""ADK root agent for the warehouse manager (LiteLLM + OpenAI via ``LiteLlm``).

Loaded by ``adk web`` as ``warehouse_manager_agent.agent.root_agent``. Run the CLI from
``apps/adk_warehouse_manager_agent`` so this package resolves. Use ``op run`` if
``OPENAI_API_KEY`` in ``.env`` is an ``op://`` reference.
"""
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from warehouse_manager_agent.tools import check_warehouse_availability, reserve_warehouse_items



model = LiteLlm(
    model="openai/gpt-4.1-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

root_agent = Agent(
    name="warehouse_manager_agent",
    model=model,
    tools=[check_warehouse_availability, reserve_warehouse_items],
    description=(
        "An agent that can check the availability of items in the warehouses and reserve them."
    ),
    instruction="""
You are part of the shopping assistant that can manage available inventory in the warehouses.

You will be given a conversation history and a list of tools. Your task is to perform actions
requested by the latest user query. Answer the parts of the query that you can satisfy using
your tools and the information they return.

Instructions:
- You must always check the availability of the items in the warehouses before reserving them.
- Only reserve items in warehouses if the entire order can be reserved or the user has confirmed
  that they want a partial reservation.
- If you cannot reserve any items, return an answer that the order cannot be reserved.
- If you can reserve some items, return an answer that the order can be partially reserved and
  include the details.
- If only a partial quantity can be reserved in some warehouses, try to combine the required
  quantity from different warehouses.
- Try to reserve items from the closest warehouse to the user first if the user's location is
  provided.
""",
)
