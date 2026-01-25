"""
FastAPI Endpoints for RAG-based Product Q&A API

This module defines the REST API endpoints for the RAG system.
Currently implements a single POST endpoint for answering product questions.

API Structure:
- Base path: /rag
- Endpoint: POST /rag/ (note the trailing slash)
- Request: JSON with query field
- Response: JSON with request_id and answer fields

The endpoint is organized using FastAPI's APIRouter pattern for modularity
and future extensibility (e.g., adding /rag/health, /rag/feedback endpoints).
"""

import logging

from fastapi import APIRouter, Request

from api.agents.retrieval_generation import rag_pipeline_wrapper
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext

# Configure logging to track API requests and errors
# Format includes timestamp, logger name, level (INFO/ERROR), and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create router for RAG-specific endpoints
# This allows grouping related endpoints and applying common middleware/tags
rag_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> RAGResponse:
    """
    Answer questions about products using RAG pipeline.

    This is the main entry point for the Q&A system. It receives a user question,
    runs it through the complete RAG pipeline (retrieve + generate), and returns
    the answer with a unique request ID for tracking.

    Args:
        request (Request): FastAPI request object containing request metadata
                          The middleware attaches request_id to request.state
        payload (RAGRequest): Pydantic model with user's query
                             Automatically validated by FastAPI

    Returns:
        RAGResponse: Pydantic model with request_id and generated answer
                    Automatically serialized to JSON by FastAPI

    HTTP Details:
        - Method: POST (idempotent for same query, but creates new request_id each time)
        - Content-Type: application/json
        - Status: 200 OK on success, 500 on internal error (unhandled)

    Request Flow:
        1. RequestIDMiddleware generates UUID and attaches to request.state.request_id
        2. FastAPI validates payload against RAGRequest model
        3. Endpoint extracts query and calls rag_pipeline()
        4. RAG pipeline retrieves context and generates answer
        5. Response includes original request_id for request tracing

    Example Request:
        POST /rag/
        {
            "query": "What are the best wireless headphones?"
        }

    Example Response:
        {
            "request_id": "bf802801-da21-4b61-a10c-e700d4aafe2e",
            "answer": "Based on the available products, I recommend..."
        }

    Why request_id:
        - Enables request tracing in distributed systems
        - Allows correlation between logs, errors, and user issues
        - Useful for debugging: "What happened with request XYZ?"
        - Added to response headers (X-Request-ID) by middleware

    Production improvements needed:
        - Add error handling (try/except) for rag_pipeline failures
        - Add rate limiting to prevent abuse
        - Add request/response logging for analytics
        - Add timeout to prevent long-running queries
        - Validate query length and content (prevent injection)
    """
    # Run the complete RAG pipeline with enrichment: retrieve context, generate answer, and fetch metadata
    # Changed in Video 3: Now uses rag_pipeline_wrapper instead of rag_pipeline
    # This wrapper adds product images and prices from Qdrant for rich frontend display
    answer = rag_pipeline_wrapper(payload.query)

    # Return structured response with request ID for tracing and enriched product context
    # request.state.request_id was set by RequestIDMiddleware
    #
    # Changed in Video 3: Added used_context field with product metadata
    # Structure:
    #   - answer (str): Natural language response from LLM
    #   - used_context (list[RAGUsedContext]): Product cards for frontend
    #       Each item contains: image_url, price, description
    #
    # Why this structure?
    #   - Frontend can display visual product cards with images and prices
    #   - Separates enrichment logic (wrapper) from core RAG logic
    #   - Enables rich UI without modifying core pipeline
    #
    # Example response:
    # {
    #   "request_id": "uuid...",
    #   "answer": "The best headphones are...",
    #   "used_context": [
    #     {"image_url": "...", "price": 39.99, "description": "TELSOR Earbuds..."},
    #     ...
    #   ]
    # }
    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer["answer"],
        used_context=[
            RAGUsedContext(**used_context) for used_context in answer["used_context"]
        ],
    )


# Create main API router and mount the RAG router
# This allows multiple routers (e.g., /rag, /admin, /health) to be combined
api_router = APIRouter()

# Mount rag_router under /rag prefix with "rag" tag
# - prefix="/rag": All rag_router endpoints become /rag/*
# - tags=["rag"]: Groups these endpoints in OpenAPI docs under "RAG" section
#
# Final URL structure:
#   POST /rag/ -> handled by rag() function above
#
# Why this pattern:
#   - Separation of concerns: RAG logic isolated in rag_router
#   - Scalability: Easy to add more routers (/admin, /analytics, etc.)
#   - Documentation: Auto-generated OpenAPI docs group by tags
#   - Versioning: Could create /v1/rag, /v2/rag routers separately
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
