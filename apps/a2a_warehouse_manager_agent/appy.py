# ruff: noqa: F401, I001 — scaffold; imports wired as the course progresses.
"""UVicorn entrypoint for the Week 6 warehouse agent as an A2A (Agent2Agent) server.

Builds an :class:`a2a.types.AgentCard`, wraps the existing Google ADK
:class:`warehouse_manager_agent.agent.WarehouseManagerAgent` in a
:class:`warehouse_manager_agent.agent_executor.WarehouseManagerAgentExecutor`,
and serves the JSON-RPC + SSE stack from ``a2a.server`` via Starlette/Uvicorn.

**Course context:** pairs with ``notebooks/week6/03-Warehouse-Agent-A2A.ipynb``.
Run with secrets resolved (e.g. ``make run-a2a-warehouse-agent``) when ``.env``
uses ``op://`` references for ``OPENAI_API_KEY``.
"""
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from warehouse_manager_agent.agent import WarehouseManagerAgent
from warehouse_manager_agent.agent_executor import WarehouseManagerAgentExecutor
from dotenv import load_dotenv
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService



# Merge .env into the process; when the process is started under ``op run``,
# injected vars already override literals—``override=False`` is default.
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "10000"))


def main() -> None:
    """Wire ADK runner → A2A request handler and block on Uvicorn."""

    # ``streaming=True`` advertises that this agent can emit incremental SSE
    # events (status + final artifacts) for long LLM/tool runs.
    capabilities = AgentCapabilities(streaming=True)
    skills_availability = AgentSkill(
        id="ABC",
        name="Check Availability",
        description="Check the availability of items in the warehouses",
        tags=["availability", "warehouse", "inventory"],
        examples=[
            "What is the availability of the item B09W5STB6T?",
            "What is the availability of the item B09W5STB6T in the warehouse ABC?",
            "What is the availability of the item B09W5STB6T in the warehouse ABC for quantity 1?",
        ]
    )
    skills_reservation = AgentSkill(
        id="ABC",
        name="Reserve Items",
        description="Reserve items in the warehouses",
        tags=["reservation", "warehouse", "inventory"],
        examples=[
            "Reserve 1 item B09W5STB6T in the warehouse ABC",
            "Reserve 1 item B09W5STB6T in the warehouse ABC for quantity 1",
        ]
    )
    agent_card = AgentCard(
        name="warehouse_manager_agent",
        description="An agent that can check the availability of items in the warehouses and reserve them.",
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[skills_availability, skills_reservation],
    )

    adk_agent = WarehouseManagerAgent().get_agent()
    runner = Runner(
        agent=adk_agent,
        session_service=InMemorySessionService(),
        app_name=agent_card.name,
        memory_service=InMemoryMemoryService(),
    )
    agent_executor = WarehouseManagerAgentExecutor(runner)

    request_handler= DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Blocks until shutdown; any code after this line does not run.
    logger.info("Starting A2A server on http://%s:%s", HOST, PORT)
    uvicorn.run(server.build(), host=HOST, port=PORT)

if __name__ == "__main__":
    main()
