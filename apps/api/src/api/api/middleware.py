"""
Custom FastAPI Middleware for Request Tracing

This module implements RequestIDMiddleware, which assigns a unique UUID to every
incoming HTTP request. The request ID enables distributed tracing and debugging.

Middleware in FastAPI:
- Runs before and after every request
- Can modify requests/responses
- Ideal for cross-cutting concerns (logging, auth, tracing)
- Executes in the order added to app

Request flow with middleware:
1. Request arrives → RequestIDMiddleware generates UUID
2. UUID stored in request.state (accessible to endpoints)
3. Request processed by endpoint
4. Response generated
5. Middleware adds X-Request-ID header to response
6. Response sent to client
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import logging


# Module-level logger for tracking request lifecycle
logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds a unique request ID to each HTTP request.

    This middleware generates a UUID for every incoming request and:
    1. Attaches it to request.state for use in endpoints
    2. Logs request start/completion with the ID
    3. Adds X-Request-ID header to the response

    Benefits of request IDs:
    - Distributed tracing: Track a request across multiple services
    - Debugging: Find all logs related to a specific request
    - Client-side tracking: Client can reference request ID in bug reports
    - Performance monitoring: Measure end-to-end latency per request

    The request ID follows the pattern:
        UUID v4: "bf802801-da21-4b61-a10c-e700d4aafe2e"
        - Universally unique (collision probability ~0)
        - 128-bit random number formatted as 8-4-4-4-12 hex digits
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process each incoming request and outgoing response.

        This method is called for every HTTP request. It runs before the endpoint
        handler (on the way in) and after the endpoint returns (on the way out).

        Args:
            request (Request): The incoming HTTP request object
                              Contains method, URL, headers, body, etc.

            call_next (callable): Coroutine that processes the request
                                 Calls the next middleware or endpoint handler
                                 Returns the response from downstream

        Returns:
            Response: The HTTP response (after adding X-Request-ID header)

        Flow:
            1. Generate UUID before processing request
            2. Attach UUID to request.state (accessible in endpoints)
            3. Log request start with method, path, and ID
            4. Call next handler/endpoint (await = pause until complete)
            5. Add X-Request-ID header to response
            6. Log request completion
            7. Return modified response to client

        Why async/await:
            - FastAPI is async framework (non-blocking I/O)
            - await call_next() allows other requests to process while waiting
            - Critical for scalability under high load

        Example logs:
            INFO: Request started: POST /rag/ (request_id: bf802801...)
            INFO: Request completed: POST /rag/ (request_id: bf802801...)

        Production considerations:
            - Could add request timing: time.time() before/after call_next()
            - Could add error handling: try/except around call_next()
            - Could add request fingerprinting: hash of body for duplicate detection
            - Could store request_id in external tracing system (Jaeger, Zipkin)
        """

        # Step 1: Generate globally unique request ID
        # UUID v4 uses random numbers (not time-based like v1)
        request_id = str(uuid.uuid4())

        # Step 2: Attach request ID to request state
        # request.state is a dict-like object for storing request-scoped data
        # Endpoints can access via: request.state.request_id
        request.state.request_id = request_id

        # Step 3: Log request start with key metadata
        # Includes HTTP method (GET/POST), URL path, and request ID
        logger.info(f"Request started: {request.method} {request.url.path} (request_id: {request_id})")

        # Step 4: Process the request through remaining middleware and endpoint
        # This is where the actual business logic happens (RAG pipeline, etc.)
        # await = asynchronous wait (non-blocking, allows other requests to run)
        response = await call_next(request)

        # Step 5: Add request ID to response headers
        # Standard header name: X-Request-ID (widely used convention)
        # Client can extract this header to reference the request later
        response.headers["X-Request-ID"] = request_id

        # Step 6: Log request completion (pairs with start log)
        # Same format as start log for easy correlation
        logger.info(f"Request completed: {request.method} {request.url.path} (request_id: {request_id})")

        # Step 7: Return modified response to client
        return response
