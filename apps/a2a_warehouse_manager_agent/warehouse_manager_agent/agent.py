"""ADK agent definition for warehouse availability + reservation (A2A backend).

Same behavioral layer as the Week 6 ADK notebook: a ``google.adk.agents.Agent``
with ``LiteLlm`` and two function tools. The A2A server imports this module and
runs the agent via ``warehouse_manager_agent.agent_executor``.

**Import style:** tools live in this package, so we use a **relative** import
(``from .tools import ...``) so ``uv run appy.py`` from
``apps/a2a_warehouse_manager_agent`` resolves correctly.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from .tools import (
    check_warehouse_availability,
    reserve_warehouse_items,
)


class WarehouseManagerAgent:
    """Factory for the configured ADK agent (model + tools + system instruction)."""

    def __init__(self) -> None:
        self.model = LiteLlm(
            model="openai/gpt-4.1-mini",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.agent = Agent(
            name="warehouse_manager_agent",
            model=self.model,
            tools=[check_warehouse_availability, reserve_warehouse_items],
            description=(
                "An agent that can check the availability of items in the warehouses "
                "and reserve them."
            ),
            instruction="""
You are part of the shopping assistant that can manage available inventory in the warehouses.

You will be given a conversation history and a list of tools. Your task is to perform actions
requested by the latest user query. Answer the parts of the query that you can satisfy using
your tools and the information they return.

Instructions:
- You must always check the availability of the items in the warehouses before reserving them.
 - Only reserve items in warehouses if the entire order can be reserved or the user has confirmed that they want a partial reservation.
 - If you cannot reserve any items, return an answer that the order cannot be reserved.
 - If you can reserve some items, return an answer that the order can be partially reserved and include the details.
- If only a partial quantity can be reserved in some warehouses, try to combine the required quantity from different warehouses.
- Try to reserve items from the closest warehouse to the user first if the user's location is provided.
""",
        )

    def get_agent(self) -> Agent:
        """Return the built agent for :class:`google.adk.runners.Runner`."""
        return self.agent
