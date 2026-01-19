"""
FastAPI Application Entry Point for RAG Q&A API

This is the main application file that:
1. Creates the FastAPI app instance
2. Configures middleware (request tracing, CORS)
3. Registers API routers (currently just /rag)
4. Sets up logging

The app is launched by uvicorn in Docker:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Application Structure:
    app.py (this file)      → Application setup and middleware
    api/endpoints.py        → API route handlers
    api/models.py          → Request/response schemas
    api/middleware.py      → Custom middleware (request ID)
    agents/retrieval_generation.py → RAG pipeline logic
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.api.endpoints import api_router
from api.api.middleware import RequestIDMiddleware

# Configure application-wide logging
# All loggers in the application will inherit this configuration
# Format: "2026-01-19 20:48:33,696 - api.api.middleware - INFO - Request started: ..."
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application instance
# This is the ASGI application that uvicorn will serve
# FastAPI automatically generates:
#   - OpenAPI schema at /openapi.json
#   - Interactive docs at /docs (Swagger UI)
#   - Alternative docs at /redoc (ReDoc)
app = FastAPI()

# Add RequestIDMiddleware first (runs before CORS)
# Middleware order matters: added first = runs first (outermost layer)
# This ensures every request gets a unique ID before any other processing
app.add_middleware(RequestIDMiddleware)

# Add CORS middleware to allow cross-origin requests
# This enables the Streamlit frontend (port 8501) to call the API (port 8000)
# Without CORS, browsers block requests between different origins (ports/domains)
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"]: Accept requests from any origin
    # Production: Should restrict to specific domains like ["http://localhost:8501"]
    allow_origins=["*"],

    # allow_credentials=True: Allow cookies and auth headers
    # Needed if frontend sends authentication tokens
    allow_credentials=True,

    # allow_methods=["*"]: Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    # Production: Could restrict to ["GET", "POST"] if that's all you need
    allow_methods=["*"],

    # allow_headers=["*"]: Allow all HTTP headers
    # Production: Could restrict to specific headers like ["Content-Type", "Authorization"]
    allow_headers=["*"],
)

# Register API routes from endpoints.py
# api_router contains all endpoints (currently just /rag)
# This mounts the entire API under the root path
#
# URL structure after this line:
#   - api_router has rag_router with prefix="/rag"
#   - Final endpoint: POST /rag/
#
# Why this pattern:
#   - Separation of concerns: routing logic separate from app setup
#   - Scalability: Easy to add more routers (admin, health checks, etc.)
#   - Testing: Can test api_router independently of app setup
app.include_router(api_router)

# Application is now ready to serve requests
# Uvicorn will:
#   1. Load this module (api.app)
#   2. Find the 'app' variable (FastAPI instance)
#   3. Start ASGI server listening on 0.0.0.0:8000
#   4. Route incoming requests through middleware → endpoints
#
# Request flow example:
#   Client (Streamlit) → POST /rag/ with {"query": "..."}
#   → CORSMiddleware (validates origin, adds CORS headers)
#   → RequestIDMiddleware (generates UUID, logs start)
#   → api_router → rag_router → rag() endpoint
#   → rag_pipeline() executes RAG workflow
#   → Response with answer
#   → RequestIDMiddleware (logs completion, adds X-Request-ID header)
#   → CORSMiddleware (adds CORS headers)
#   → Client receives response
