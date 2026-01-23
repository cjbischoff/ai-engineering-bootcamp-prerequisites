# FastAPI Backend Service

Production-ready RAG (Retrieval-Augmented Generation) API service built with FastAPI, providing Q&A capabilities over product data using vector search and LLM generation.

## Overview

This FastAPI application serves as the backend for a RAG-based chatbot system. It combines semantic search over product embeddings (stored in Qdrant) with LLM-powered answer generation to provide accurate, grounded responses to user questions about products.

## Architecture

```
apps/api/
├── src/api/
│   ├── app.py                 # Application entry point, middleware setup
│   ├── api/                   # API layer (routes, models, middleware)
│   │   ├── endpoints.py       # Route handlers
│   │   ├── models.py          # Pydantic request/response schemas
│   │   └── middleware.py      # Request ID middleware
│   ├── agents/                # Business logic layer
│   │   └── retrieval_generation.py  # RAG pipeline implementation
│   └── core/                  # Configuration
│       └── config.py          # Environment variable management
├── evals/                     # Evaluation scripts (testing RAG quality)
│   ├── __init__.py
│   └── eval_retriever.py      # RAGAS-based evaluation
├── Dockerfile                 # Container image definition
└── pyproject.toml             # Python dependencies
```

## Key Components

### Application Layer (`app.py`)

- **FastAPI Setup**: Creates ASGI application with auto-generated OpenAPI docs
- **Middleware Stack**:
  1. `RequestIDMiddleware` - Generates UUID for request tracing
  2. `CORSMiddleware` - Enables cross-origin requests from frontend
- **Router Registration**: Mounts API routes with `/rag` prefix
- **Logging Configuration**: Structured application-wide logging

**Why This Matters**: Middleware order is critical - RequestID must run before CORS to ensure all requests get traced.

### API Layer (`api/`)

#### Endpoints (`endpoints.py`)
- **`POST /rag/`**: Main Q&A endpoint
  - Accepts: `{"query": "user question"}`
  - Returns: `{"request_id": "uuid", "answer": "generated response"}`
  - Calls RAG pipeline and wraps response with request ID

#### Models (`models.py`)
- **`RAGRequest`**: Validates incoming JSON (required `query` field)
- **`RAGResponse`**: Structured response with request tracking
- **FastAPI Integration**: Auto-validation, 422 errors, OpenAPI schema generation

#### Middleware (`middleware.py`)
- **`RequestIDMiddleware`**: Request tracing implementation
  - Generates UUID v4 for each request
  - Attaches to `request.state.request_id`
  - Adds `X-Request-ID` response header
  - Logs request lifecycle for distributed tracing

### RAG Pipeline (`agents/retrieval_generation.py`)

Complete 5-step RAG workflow:

1. **Embedding Generation** (`get_embedding`)
   - Converts query to 1536-dim vector using OpenAI `text-embedding-3-small`
   - Must match preprocessing embedding model for consistency

2. **Vector Retrieval** (`retrieve_data`)
   - Queries Qdrant for k-nearest neighbors (default k=5)
   - Uses cosine similarity for semantic matching
   - Returns product IDs, descriptions, ratings, similarity scores

3. **Context Formatting** (`process_context`)
   - Converts structured data to human-readable text
   - Format: `- ID: {asin}, rating: {rating}, description: {description}`

4. **Prompt Construction** (`build_prompt`)
   - System role: "shopping assistant"
   - Key constraint: "Only use provided context" (prevents hallucination)
   - Combines instructions + context + question

5. **Answer Generation** (`generate_answer`)
   - Uses OpenAI `gpt-5-nano` with `reasoning_effort="minimal"`
   - Returns natural language answer grounded in retrieved products

**LangSmith Tracing**: All functions decorated with `@traceable` for observability:
- Captures inputs, outputs, execution time, errors
- Manual token tracking via `get_current_run_tree()`
- View traces at https://smith.langchain.com

### Evaluation (`evals/`)

Automated RAG quality testing using RAGAS metrics. See [evals/README.md](evals/README.md) for details.

**4 Evaluation Metrics**:
1. **Faithfulness** (0-1): Answer grounded in context?
2. **Response Relevancy** (0-1): Answer addresses question?
3. **Context Precision** (0-1): Retrieved products relevant?
4. **Context Recall** (0-1): Found all relevant products?

**Run Evaluations**: `make run-evals-retriever` (from project root)

### Configuration (`core/config.py`)

- **Pattern**: `pydantic-settings` with `.env` file loading
- **Environment Variables**:
  - `OPENAI_KEY`: OpenAI API authentication
  - `LANGSMITH_*`: LangSmith tracing configuration
  - `QDRANT_URL`: Vector database connection (default: `http://qdrant:6333`)

## Running Locally

### Docker (Recommended)
```bash
# From project root
make run-docker-compose
# or manually:
docker compose up --build
```

**Service URLs**:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Local Development (Without Docker)
```bash
cd apps/api

# Install dependencies
uv sync

# Run API server
uv run uvicorn api.app:app --reload --port 8000
```

**Note**: Requires Qdrant running separately. Change `QDRANT_URL` to `http://localhost:6333` in `.env`.

## API Usage

### Example Request
```bash
curl -X POST http://localhost:8000/rag/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best wireless headphones under $100?"}'
```

### Example Response
```json
{
  "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "answer": "Based on the products in our catalog, the XYZ Wireless Headphones at $89.99 offer excellent value with noise cancellation and 30-hour battery life. Customers rate them 4.5/5 stars."
}
```

## Dependencies

**Core Framework**:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic-settings` - Configuration management

**RAG Pipeline**:
- `openai` - LLM and embeddings
- `qdrant-client` - Vector database client
- `langsmith` - Observability and tracing

**Evaluation**:
- `ragas` - RAG-specific evaluation metrics
- `langchain-openai` - LangChain integrations

See [pyproject.toml](pyproject.toml) for complete dependency list.

## Docker Integration

**Dockerfile Highlights**:
- Base image: `python:3.12-slim`
- Package manager: `uv` for fast dependency resolution
- Working directory: `/app`
- Port: 8000
- Hot reload: Volume mounts in `docker-compose.yml`

**Volume Mounts** (from project root):
```yaml
volumes:
  - ./apps/api/src:/app/apps/api/src  # Hot reload for development
```

**Environment Variables**: Loaded from `.env` via `env_file` in `docker-compose.yml`

## Development Workflow

1. **Add Dependency**: `uv add --package api <package-name>`
2. **Modify Code**: Changes auto-reload in Docker
3. **Test Locally**: `curl` or API docs at http://localhost:8000/docs
4. **Run Evaluations**: `make run-evals-retriever` to test RAG quality
5. **View Traces**: LangSmith UI for debugging and optimization

## Key Design Patterns

### 1. Middleware Stack
- Added first = runs first (outermost layer)
- RequestIDMiddleware → CORS → Endpoints
- Response flows back in reverse order

### 2. Pydantic Validation
- FastAPI auto-validates request/response
- Returns 422 for invalid inputs (not 500 crashes)
- Type safety and OpenAPI schema generation

### 3. RAG vs Pure LLM
- Pure LLM: May hallucinate, knowledge cutoff limits
- RAG: Grounds answers in actual data, always current
- Trade-off: Requires vector DB but provides verifiable answers

### 4. Observability
- Request IDs for distributed tracing
- LangSmith for LLM-specific metrics (tokens, prompts, similarity scores)
- Structured logging for debugging

### 5. Separation of Concerns
- `app.py`: Application setup
- `api/`: HTTP layer (routes, validation, middleware)
- `agents/`: Business logic (RAG pipeline)
- `core/`: Configuration
- `evals/`: Quality testing

## Known Limitations

**Not Implemented (Intentional MVP Scope)**:
- Error handling: No try/except around RAG pipeline calls
- Rate limiting: API unprotected
- Timeout handling: Long queries could hang
- Input validation: No query length limits
- Connection pooling: New Qdrant client per request (inefficient)
- Caching: Common queries not cached
- Authentication: No API keys
- Query logging: No analytics
- Response streaming: Answers returned all-at-once

**When to Add**:
- Error handling: Before production deployment
- Rate limiting: When opening to public
- Authentication: When controlling access
- Caching: When reducing API costs becomes priority

## Troubleshooting

**Qdrant Connection Errors**:
- In Docker: Use `http://qdrant:6333` (service name)
- Locally: Use `http://localhost:6333`
- Verify collection exists: Check Qdrant dashboard at http://localhost:6333/dashboard

**OpenAI API Errors**:
- Check `OPENAI_KEY` in `.env`
- Rate limits: Reduce concurrent requests
- Token limits: Shorten retrieved context

**LangSmith Tracing Not Working**:
- Verify all 4 `LANGSMITH_*` environment variables set
- Check API key validity
- Set `LANGSMITH_TRACING=true`

## Related Documentation

- **Project Root**: [../../README.md](../../README.md) - Overall architecture
- **Evals**: [evals/README.md](evals/README.md) - Evaluation system
- **Agents**: [src/api/agents/README.md](src/api/agents/README.md) - RAG pipeline
- **Frontend**: [../chatbot_ui/README.md](../chatbot_ui/README.md) - Streamlit UI
- **Notebooks**: [../../notebooks/week1/README.md](../../notebooks/week1/README.md) - Data preprocessing

## Production Considerations

Before deploying to production:
1. Add comprehensive error handling
2. Implement rate limiting (e.g., Redis + FastAPI middleware)
3. Add authentication (API keys, OAuth)
4. Enable HTTPS/TLS
5. Set up monitoring (Prometheus, Grafana)
6. Configure autoscaling
7. Add input sanitization
8. Implement caching (Redis)
9. Connection pooling for Qdrant
10. Set up CI/CD pipeline with evaluation gates
