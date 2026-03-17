"""
FastAPI Endpoints for the LangGraph multi-agent RAG API (Sprint 2 / Video 5-6; Week 5).

Defines two routers:
- agent router (prefix=/agent): POST /agent/ streams SSE from rag_agent_stream_wrapper.
  Payload: query, thread_id. Yields status updates and final_answer JSON (answer,
  used_context, trace_id, shopping_cart). thread_id enables LangGraph checkpointing.
- feedback router (prefix=/submit_feedback): POST for LangSmith feedback (thumbs, comment).
  Payload: trace_id, feedback_score, feedback_text. Used by frontend to attach feedback
  to LangSmith runs for evaluation.

Uses FastAPI APIRouter pattern for modularity. RequestIDMiddleware attaches request_id
to request.state for tracing.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.agents.graph import rag_agent_stream_wrapper
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext, FeedbackRequest, FeedbackResponse
from api.api.processors.submit_feedback import submit_feedback

# Configure logging to track API requests and errors
# Format includes timestamp, logger name, level (INFO/ERROR), and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create router for RAG-specific endpoints
# This allows grouping related endpoints and applying common middleware/tags
rag_router = APIRouter()
feedback_router = APIRouter()





@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> StreamingResponse:
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
        - Add rate limiting to prevent abuse
        - Add rate limiting to prevent abuse
        - Add request/response logging for analytics
        - Add timeout to prevent long-running queries
        - Validate query length and content (prevent injection)
    """
    # Week 5: LangGraph coordinator-based multi-agent. rag_agent_stream_wrapper runs
    # coordinator -> product_qa_agent | shopping_cart_agent, streams SSE. thread_id
    # enables PostgresSaver checkpointing (same ID = same conversation state).


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
    return StreamingResponse(
        rag_agent_stream_wrapper(payload.query, payload.thread_id),
        media_type="text/event-stream",
    )






@feedback_router.post("/")
def send_feedback(
    request: Request,
    payload: FeedbackRequest,
) -> FeedbackResponse:
    # Log for troubleshooting: confirms trace_id and payload shape before calling LangSmith.
    logger.info(
        "Feedback received: trace_id=%s (present=%s), feedback_score=%s, has_text=%s",
        payload.trace_id,
        payload.trace_id is not None and bool(payload.trace_id),
        payload.feedback_score,
        bool(payload.feedback_text and payload.feedback_text.strip()),
    )
    submit_feedback(
        payload.trace_id,
        payload.feedback_score,
        payload.feedback_text,
        payload.feedback_source_type,
    )
    return FeedbackResponse(
        request_id=request.state.request_id,
        status="success",
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
api_router.include_router(rag_router, prefix="/agent", tags=["agent"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])
