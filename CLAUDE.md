# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Engineering Bootcamp prerequisites repository featuring a complete chatbot application stack with FastAPI backend, Streamlit frontend, and Qdrant vector database for RAG operations. This is a **uv workspace monorepo** with modular apps.

## Essential Commands

### Environment & Dependencies
```bash
# Install/sync all dependencies (root + all workspace apps)
uv sync

# Add dependency to root workspace
uv add <package-name>

# Add dependency to specific app
uv add --package api <package-name>
uv add --package chatbot-ui <package-name>
```

### Running Services

**Docker Compose (Recommended):**
```bash
# Start all services (API, UI, Qdrant)
make run-docker-compose
# or manually:
uv sync && docker compose up --build

# Detached mode
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Force rebuild
docker compose up --build --force-recreate
```

**Local Development (No Docker):**
```bash
# Run FastAPI backend
cd apps/api
uv run uvicorn api.app:app --reload --port 8000

# Run Streamlit frontend (separate terminal)
cd apps/chatbot_ui
uv run streamlit run src/chatbot_ui/app.py

# Run Jupyter notebooks
uv run jupyter notebook notebooks/
```

### Jupyter Notebooks
```bash
# Launch Jupyter
uv run jupyter notebook notebooks/

# Clean outputs before commit (REQUIRED)
make clean-notebook-outputs
```

### Service URLs
- Chatbot UI: http://localhost:8501
- API Docs: http://localhost:8000/docs
- API Health: http://localhost:8000/health
- Qdrant Dashboard: http://localhost:6333/dashboard
- Qdrant API: http://localhost:6333

## Architecture & Structure

### Workspace Organization
**uv workspace monorepo** with three main components:
- Root workspace: Shared dependencies and workspace config
- `apps/api`: FastAPI backend service
- `apps/chatbot_ui`: Streamlit frontend service

### Key Architectural Patterns

**1. Multi-Provider LLM Abstraction**
- `apps/api/src/api/app.py:25-47` - `run_llm()` function abstracts three providers (OpenAI, Groq, Google GenAI)
- Clients initialized once at module level (lines 20-22) for performance
- Provider selection via POST request, not environment switching

**2. Configuration Management**
- Both apps use `pydantic-settings` with `.env` file loading
- Config classes: `api/core/config.py`, `chatbot_ui/core/config.py`
- Pattern: `AppConfig` class with `model_config = SettingsConfigDict(env_file=".env")`

**3. Service Communication**
- Frontend → Backend via `API_URL` environment variable (defaults to `http://api:8000` in Docker)
- `chatbot_ui/app.py:71` - Direct POST to `/chat` endpoint
- Error handling with popup messages in `api_call()` helper (lines 6-36)

**4. RAG Pipeline (Week 1)**
- **Preprocessing**: `notebooks/week1/02-RAG-preprocessing-Amazon.ipynb`
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Vector Storage**: Qdrant collection `"Amazon-items-collection-00"`
- **Distance Metric**: Cosine similarity for semantic search
- **Data Flow**: Product descriptions → embeddings → Qdrant → semantic retrieval

**5. RAG API Implementation**

This is the production FastAPI implementation of the RAG (Retrieval-Augmented Generation) pipeline for answering product questions.

**Architecture Overview:**
- **Location**: `apps/api/src/api/`
- **Pattern**: FastAPI with modular routing (APIRouter)
- **Request Flow**: Client → Middleware → Router → Endpoint → RAG Pipeline → LLM → Response

**Key Components:**

**a) Application Setup** (`api/app.py`)
- FastAPI instance with auto-generated OpenAPI docs at `/docs`
- Middleware stack (order matters - added first = runs first):
  1. `RequestIDMiddleware` - Generates UUID for request tracing
  2. `CORSMiddleware` - Enables cross-origin requests from frontend
- Router registration: `api_router` mounted at root with `/rag` prefix
- Why CORS: Allows Streamlit frontend (port 8501) to call API (port 8000)

**b) Request Tracing Middleware** (`api/middleware.py`)
- Implements `BaseHTTPMiddleware` pattern
- Generates UUID v4 for every incoming request
- Attaches `request_id` to `request.state` (accessible in endpoints)
- Adds `X-Request-ID` response header for client-side tracking
- Logs request start/completion for distributed tracing
- Benefits: Debugging, correlation across services, client support tickets

**c) Request/Response Models** (`api/models.py`)
- Uses Pydantic `BaseModel` for automatic validation
- `RAGRequest`: Validates incoming JSON with required `query` field
- `RAGResponse`: Structured response with `request_id` and `answer`
- Why Pydantic: Type safety, auto-validation, OpenAPI schema generation
- FastAPI returns 422 Unprocessable Entity if validation fails

**d) API Endpoints** (`api/endpoints.py`)
- `POST /rag/` - Main Q&A endpoint
- Receives `RAGRequest`, calls `rag_pipeline()`, returns `RAGResponse`
- Uses `APIRouter` pattern for modularity (easy to add `/rag/health`, `/rag/feedback`)
- Tags endpoints as "rag" for OpenAPI documentation grouping
- Request ID automatically available via `request.state.request_id`

**e) RAG Pipeline Logic** (`agents/retrieval_generation.py`)

Complete 5-step RAG workflow:

1. **Embedding Generation** (`get_embedding`)
   - Converts text to 1536-dimensional vector using OpenAI
   - Same model as preprocessing: `text-embedding-3-small`
   - Enables semantic similarity matching (not just keyword search)

2. **Vector Retrieval** (`retrieve_data`)
   - Queries Qdrant for k-nearest neighbors (default k=5)
   - Uses cosine similarity to find semantically similar products
   - HNSW algorithm for fast approximate nearest neighbor search
   - Returns: product IDs, descriptions, ratings, similarity scores

3. **Context Formatting** (`process_context`)
   - Converts structured data (lists) into human-readable text
   - Format: `- ID: {asin}, rating: {rating}, description: {description}`
   - Why: LLM needs formatted text, not raw Python data structures
   - Uses `zip()` to iterate through parallel lists

4. **Prompt Construction** (`build_prompt`)
   - Combines system instructions, retrieved context, and user question
   - System role: "shopping assistant"
   - Key constraint: "Only use provided context" (prevents hallucination)
   - Prompt engineering: Clear role → Task → Context → Constraint → Question

5. **Answer Generation** (`generate_answer`)
   - Uses OpenAI `gpt-5-nano` with `reasoning_effort="minimal"`
   - Why nano: Cost-effective for straightforward retrieval-based answers
   - Single system message (no conversation history)
   - Returns natural language answer grounded in retrieved products

**Docker Integration:**
- Qdrant connection: `http://qdrant:6333` (Docker Compose service name)
- Container networking: Services communicate by name, not localhost
- Volume mount: `./qdrant_storage:/qdrant/storage:z` for persistence

**Lessons Learned:**

1. **TypeError with zip() and tuple()**
   - Problem: `zip(tuple(list1, list2, list3))` is invalid syntax
   - Cause: `tuple()` takes one iterable, not multiple arguments
   - Fix: Use `zip(list1, list2, list3)` directly - no tuple wrapper needed
   - When multi-line formatting splits arguments, the error becomes visible
   - Detection: Runtime `TypeError: tuple expected at most 1 argument, got 3`

2. **Qdrant Connection in Docker**
   - Use service name `http://qdrant:6333`, not `http://localhost:6333`
   - Docker Compose creates internal DNS for service-to-service communication
   - Localhost refers to container itself, not the host machine
   - Verify collection exists: `docker compose exec qdrant curl http://localhost:6333/collections`

3. **Middleware Ordering Matters**
   - Middleware added first runs first (outermost layer)
   - RequestIDMiddleware before CORS ensures UUID generation happens first
   - Order: Request → RequestID (generate UUID) → CORS (validate) → Endpoint
   - Response flows back through middleware in reverse order

4. **Pydantic Validation Benefits**
   - FastAPI automatically validates request JSON against model schema
   - Returns 422 with detailed errors if validation fails (not 500 crashes)
   - Field descriptions appear in auto-generated OpenAPI documentation
   - Type hints enable IDE autocomplete and catch bugs early

5. **RAG vs Pure LLM**
   - Pure LLM: May hallucinate product details, knowledge cutoff limits
   - RAG: Grounds answers in actual product data, always current
   - Trade-off: RAG requires vector database setup but provides verifiable answers
   - Best for: Product recommendations, Q&A over specific datasets

6. **OpenAI Embedding Consistency**
   - Critical: Use same embedding model for preprocessing AND runtime queries
   - Preprocessing used `text-embedding-3-small` → runtime must match
   - Different models = different vector spaces = poor retrieval quality
   - Dimension mismatch will cause Qdrant errors

7. **Request Tracing Value**
   - UUID in response body AND header enables multiple use cases
   - Client can display: "Something wrong? Reference request ID: abc123"
   - Logs can be filtered: `grep "request_id: abc123" logs/`
   - Distributed tracing: Track request across multiple services
   - Support tickets: "Request XYZ returned wrong answer"

**Production Considerations:**

Not implemented (intentional MVP scope):
- Error handling: No try/except blocks around rag_pipeline() calls
- Rate limiting: API is unprotected, could be abused
- Timeout handling: Long-running queries could hang
- Input validation: No query length limits or content sanitization
- Connection pooling: New Qdrant client created per request (inefficient)
- Caching: Common queries could be cached to reduce API calls
- Authentication: No API keys, public access
- Query logging: No analytics on what users are asking
- Response streaming: Answers returned all-at-once, not streamed

When to add these:
- Error handling: Before any production deployment
- Rate limiting: When opening to public users
- Authentication: When controlling access or charging for usage
- Caching: When reducing OpenAI API costs becomes priority

**6. LangSmith Observability (Video 5)**

This section covers the observability instrumentation added to the RAG pipeline for debugging, monitoring, and cost tracking.

**Overview:**
- **Tool**: LangSmith - Purpose-built observability platform for LLM applications
- **Integration**: Python SDK (`langsmith` package) with decorator-based instrumentation
- **Environment**: Requires 4 environment variables for LangSmith API authentication
- **Visualization**: Traces viewable at https://smith.langchain.com

**Why LangSmith vs Generic APM:**
- **LLM-Specific**: Captures prompts, completions, and token usage (not just timing)
- **Visual Trace Trees**: Shows RAG pipeline hierarchy (retrieval → formatting → generation)
- **Token Cost Tracking**: Aggregates OpenAI API usage across all calls in a request
- **Debugging**: Inspect exact prompts/responses, similarity scores, and retrieved context
- **Comparison**: Side-by-side trace comparison to debug: "Why did query A fail but query B succeed?"

**Architecture Changes:**

**a) Dependencies Added:**
```toml
# apps/api/pyproject.toml
dependencies = [
    "langsmith>=0.6.4",  # LangSmith Python SDK
    # ... other dependencies
]

# pyproject.toml (root workspace)
[dependency-groups]
dev = [
    "langsmith>=0.6.4",  # Available in all apps
    # ... other dev dependencies
]
```

**b) Environment Variables:**
```env
# env.example
export LANGSMITH_TRACING=true                       # Enable/disable tracing
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com  # LangSmith API endpoint
export LANGSMITH_API_KEY=<your-api-key>            # Authentication
export LANGSMITH_PROJECT="rag-tracing"             # Project name for organizing traces
```

**c) Code Instrumentation:**

Import statements added to `agents/retrieval_generation.py`:
```python
from langsmith import get_current_run_tree, traceable
```

**Decorator Pattern:**
Every function in the RAG pipeline decorated with `@traceable`:
1. `@traceable(name="Get Embedding", run_type="embedding", metadata={...})` on `get_embedding()`
2. `@traceable(name="Retrieve Data", run_type="retriever")` on `retrieve_data()`
3. `@traceable(name="Format Retrieved Context", run_type="prompt")` on `process_context()`
4. `@traceable(name="Build Prompt", run_type="prompt")` on `build_prompt()`
5. `@traceable(name="Generate Answer", run_type="llm", metadata={...})` on `generate_answer()`
6. `@traceable(name="RAG Pipeline")` on `rag_pipeline()` (root span)

**What @traceable Captures Automatically:**
- Function inputs (arguments)
- Function outputs (return values)
- Execution time (start/end timestamps)
- Errors and stack traces
- Parent-child span relationships

**Manual Token Tracking:**

LangSmith doesn't auto-capture OpenAI API token usage, so we manually instrument:

```python
# In get_embedding() and generate_answer()
response = openai.embeddings.create(...)  # or openai.chat.completions.create(...)

current_run = get_current_run_tree()  # Get active LangSmith trace context

if current_run:
    # Manually attach token usage from OpenAI response
    current_run.metadata["usage_metadata"] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,  # Only for chat completions
        "total_tokens": response.usage.total_tokens,
    }
```

**Why manual tracking:**
- OpenAI SDK doesn't integrate with LangSmith automatically
- Token counts needed for cost analysis: `tokens × price_per_1k_tokens`
- LangSmith aggregates tokens across all operations in a trace

**Trace Hierarchy:**

LangSmith visualizes the RAG pipeline as a tree:
```
RAG Pipeline (root span)
├── Retrieve Data
│   └── Get Embedding (child - query embedding)
├── Format Retrieved Context
├── Build Prompt
└── Generate Answer
```

**Observability Benefits:**

1. **Performance Analysis:**
   - See which step is slowest: embedding (50ms), retrieval (100ms), or generation (2s)
   - Identify bottlenecks: Is Qdrant slow or is OpenAI API slow?
   - Compare execution times across different queries

2. **Token Cost Tracking:**
   - Input tokens: Prompt length (instructions + context + question)
   - Output tokens: Generated answer length
   - Total tokens: Aggregated across embedding + generation calls
   - Cost calculation: `total_tokens × $0.00002` per 1K tokens (text-embedding-3-small) + `total_tokens × gpt-5-nano pricing`

3. **Debugging:**
   - Inspect exact prompts sent to LLM: "Is the prompt structure causing poor answers?"
   - View retrieved products and similarity scores: "Why did this query return irrelevant products?"
   - Trace failures: "Which step threw an error?"
   - Compare traces: "Why did query A return good results but query B returned bad results?"

4. **Retrieval Quality Analysis:**
   - Similarity scores visible in `Retrieve Data` span output
   - Low scores (<0.7) indicate poor semantic matches
   - Can identify when to expand the knowledge base or adjust retrieval parameters

5. **Prompt Engineering:**
   - See exact formatted context passed to LLM
   - Iterate on prompt structure by comparing traces with different instructions
   - A/B test different system prompts

**Return Value Changes:**

To support observability and future features, `rag_pipeline()` return value changed:

**Before Video 5:**
```python
return answer  # Just a string
```

**After Video 5:**
```python
return {
    "answer": answer,                      # Natural language response
    "question": question,                  # Original query (for validation)
    "retrieved_context_ids": [...],        # Product ASINs
    "retrieved_context": [...],            # Product descriptions
    "similarity_scores": [...]             # Cosine similarity scores
}
```

**Why this change:**
- **Internal observability**: Metadata available for logging and analysis
- **Debugging**: Can inspect which products influenced the answer
- **Future features**: Frontend could display "Products used in this answer"
- **Analytics**: Track which products are most frequently retrieved

**Endpoint Compatibility:**

The API endpoint extracts only the `answer` field:
```python
# api/endpoints.py
result = rag_pipeline(payload.query)
return RAGResponse(request_id=request.state.request_id, answer=result["answer"])
```

This keeps the API response unchanged (backward compatible) while enriching internal data.

**Lessons Learned:**

1. **LangSmith Trace Activation**
   - Tracing automatically enabled when environment variables are set
   - No explicit initialization code needed (SDK auto-configures)
   - Can toggle tracing on/off via `LANGSMITH_TRACING=true|false`
   - Traces appear in LangSmith UI within seconds (near real-time)

2. **Decorator Order and Nesting**
   - `@traceable` works with any function (sync or async)
   - Nested function calls automatically create parent-child span relationships
   - Example: `rag_pipeline()` calls `retrieve_data()` → nested spans in trace tree
   - No manual span management needed (decorator handles everything)

3. **Token Usage Manual Tracking Required**
   - LangSmith can't auto-capture OpenAI SDK usage stats
   - Must manually call `get_current_run_tree()` and attach metadata
   - Pattern: Call OpenAI API → Get response → Extract usage → Attach to current_run
   - Only needed for functions that call LLM/embedding APIs

4. **Metadata vs Output**
   - **Output**: Function return value (captured automatically)
   - **Metadata**: Custom key-value pairs (requires manual attachment)
   - Use metadata for token counts, model names, custom metrics
   - Output shows in "Output" tab, metadata shows in "Metadata" tab in LangSmith UI

5. **Run Types Matter**
   - `run_type="llm"`: LangSmith treats as LLM call (special UI rendering)
   - `run_type="embedding"`: Categorized as embedding operation
   - `run_type="retriever"`: Categorized as retrieval operation
   - `run_type="prompt"`: Categorized as prompt engineering
   - Enables filtering: "Show me only LLM calls" or "Show me only retrievals"

6. **Project Organization**
   - `LANGSMITH_PROJECT` groups related traces together
   - Use different projects for dev/staging/prod: "rag-tracing-dev", "rag-tracing-prod"
   - Can archive or delete entire projects to clean up old traces
   - Helpful for organizing: "rag-tracing", "chatbot-tracing", "evaluation-runs"

7. **Cost vs Value of Observability**
   - LangSmith has a free tier: 5,000 traces/month
   - Paid plans: $39/month for 50K traces, $199/month for 500K traces
   - Trade-off: Observability cost vs debugging time saved
   - Essential for production: Can't debug LLM issues without traces
   - Development: Can toggle off (`LANGSMITH_TRACING=false`) to save quota

8. **Trace Retention**
   - Free tier: 14-day retention
   - Paid plans: Configurable retention (30 days to forever)
   - Can export traces as JSON for long-term storage
   - Important: Don't rely on LangSmith as system-of-record for logs

**Project Optimizations:**

1. **Connection Pooling for Qdrant**
   - Current: New `QdrantClient` created per request in `rag_pipeline()`
   - Optimization: Create client once at module level, reuse across requests
   - Benefit: Reduce connection overhead, faster queries
   - Implementation: `qdrant_client = QdrantClient(url="http://qdrant:6333")` at module level

2. **Caching for Common Queries**
   - Pattern: Many users ask similar questions ("best headphones", "cheap laptops")
   - Optimization: Cache embeddings and/or full answers for common queries
   - Benefit: Reduce OpenAI API calls, faster responses, lower costs
   - Implementation: Redis or in-memory LRU cache with TTL (time-to-live)
   - Trade-off: Stale data if products change frequently

3. **Batch Embedding for Multi-Query**
   - Current: One embedding call per user query
   - Optimization: If system supports batch queries, embed multiple at once
   - Benefit: OpenAI API supports batch embeddings (more efficient)
   - Implementation: `openai.embeddings.create(input=["query1", "query2", ...])`

4. **Monitoring Metrics**
   - LangSmith provides traces, but not real-time metrics dashboards
   - Optimization: Add Prometheus/Grafana for operational metrics:
     * Requests per second
     * P50/P95/P99 latency
     * Error rate
     * Token usage per hour/day
   - Benefit: Alerting on anomalies, capacity planning
   - Implementation: FastAPI middleware to emit metrics

5. **Retrieval Quality Scoring**
   - Current: Similarity scores logged but not analyzed
   - Optimization: Add alerting when avg similarity < threshold (e.g., 0.6)
   - Benefit: Detect when users ask questions outside knowledge base
   - Implementation: Calculate avg similarity in `retrieve_data()`, log warning if low

6. **A/B Testing Prompts**
   - LangSmith supports prompt versioning and comparison
   - Optimization: Test different system prompts to improve answer quality
   - Benefit: Data-driven prompt engineering (not just intuition)
   - Implementation: LangSmith Playground or custom A/B test framework

**7. Evaluation Dataset Creation (Video 6)**

This section covers creating synthetic evaluation datasets for testing the RAG pipeline's performance.

**Overview:**
- **Tool**: LangSmith Datasets - Structured test cases for evaluating LLM applications
- **Notebook**: `notebooks/week1/04-evaluation-dataset.ipynb`
- **Purpose**: Generate synthetic question-answer pairs with reference contexts for systematic RAG evaluation
- **Data Source**: Existing product data from Qdrant vector database

**Why Evaluation Datasets:**
- **Systematic Testing**: Test RAG pipeline against consistent, repeatable questions
- **Quality Metrics**: Measure answer quality, retrieval accuracy, and consistency over time
- **Regression Prevention**: Detect when code changes degrade performance
- **A/B Testing**: Compare different prompts, models, or retrieval strategies
- **Continuous Improvement**: Identify weak points in the system (bad retrievals, poor answers)

**Architecture:**

**a) Synthetic Data Generation:**
- Uses LLM (GPT-4o) to generate realistic questions based on actual product data
- Structured output via JSON schema to ensure consistent format
- Each generated example includes:
  - `question`: Natural language user query
  - `chunk_ids`: Product IDs that should be retrieved
  - `answer_example`: Expected answer demonstrating quality

**b) LangSmith Dataset Structure:**
```python
# Each example in the dataset has:
{
  "inputs": {"question": "What are the best wireless headphones?"},
  "outputs": {
    "ground_truth": "Based on the products...",  # Expected answer
    "reference_context_ids": ["B09X12ABC", ...],  # Products that should be retrieved
    "reference_descriptions": ["Product 1...", ...]  # Full product descriptions
  }
}
```

**c) Notebook Workflow:**

1. **Environment Setup** (Cells 1-2):
   - Load environment variables with `python-dotenv` (OPENAI_KEY, LANGSMITH_API_KEY)
   - Initialize Qdrant client: `QdrantClient(url="http://localhost:6333")`
   - Initialize LangSmith client: `Client(api_key=os.environ["LANGSMITH_API_KEY"])`

2. **Data Exploration** (Cells 3-7):
   - Fetch sample products from Qdrant collection
   - Understand product structure (parent_asin, title, features, description)
   - Select representative products for question generation

3. **Synthetic Question Generation** (Cells 8-11):
   - Define JSON schema for structured output
   - Schema specifies: question (string), chunk_ids (array), answer_example (string)
   - Use OpenAI `gpt-4o` with `response_format={"type": "json_schema", "json_schema": output_schema}`
   - Prompt engineering: "Generate evaluation questions that test different aspects of RAG"
   - Parse JSON response into `json_output` list

4. **Helper Function** (Cell 16):
   ```python
   def get_description(parent_asin: str) -> str:
       """Fetch full product description from Qdrant by product ID"""
       points = qdrant_client.scroll(
           collection_name="Amazon-items-collection-00",
           scroll_filter=Filter(
               must=[FieldCondition(key="parent_asin", match=MatchValue(value=parent_asin))]
           ),
           limit=100,
           with_payload=True,
           with_vectors=False
       )[0]
       return points[0].payload["description"]
   ```

5. **Dataset Creation** (Cell 20):
   ```python
   dataset_name = "rag-evaluation-dataset"

   # Try to create dataset, if it already exists, read the existing one
   try:
       dataset = client.create_dataset(
           dataset_name=dataset_name,
           description="Dataset for evaluating RAG pipeline"
       )
       print(f"Created new dataset: {dataset_name}")
   except Exception as e:
       if "already exists" in str(e):
           dataset = client.read_dataset(dataset_name=dataset_name)
           print(f"Using existing dataset: {dataset_name}")
       else:
           raise e
   ```
   - Handles 409 Conflict error when dataset already exists (allows re-running notebook)
   - Reads existing dataset instead of failing

6. **Dataset Population** (Cell 21):
   ```python
   for item in json_output:
       print(item["chunk_ids"])  # Track progress
       client.create_example(
           dataset_id=dataset.id,
           inputs={"question": item["question"]},
           outputs={
               "ground_truth": item["answer_example"],
               "reference_context_ids": item["chunk_ids"],
               "reference_descriptions": [get_description(id) for id in item["chunk_ids"]]
           }
       )
   ```
   - Iterates through synthetic questions
   - Creates LangSmith example for each question
   - Fetches full product descriptions for reference context

**Key Implementation Patterns:**

1. **Environment Variable Loading:**
   - Jupyter notebooks don't auto-load `.env` files
   - Must explicitly call `load_dotenv()` from `python-dotenv`
   - Critical for LANGSMITH_API_KEY and OPENAI_KEY

2. **Structured LLM Output:**
   - JSON schema defines exact output structure
   - OpenAI's `response_format` enforces schema compliance
   - Eliminates need for manual parsing/validation
   - Pattern: `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}`

3. **Error Handling for Idempotency:**
   - Wrap dataset creation in try/except
   - Detect "already exists" error and read existing dataset
   - Allows notebook to be re-run without manual cleanup
   - Pattern: Try create → Catch conflict → Read existing

4. **Product Description Retrieval:**
   - Use Qdrant `scroll()` with filter for targeted lookup
   - Filter by `parent_asin` (product ID) for exact match
   - Set `with_vectors=False` to reduce payload size (only need metadata)
   - More efficient than full collection scan

**Lessons Learned:**

1. **Jupyter Notebook JSON Escaping:**
   - Problem: Double-escaped newlines (`\\n\\n`) appeared as literal `\n` in code
   - Cause: Incorrect JSON formatting when manually editing notebook
   - Fix: Jupyter cell source should use single `\n` for newlines, not `\\n\\n`
   - Detection: SyntaxError "unexpected character after line continuation character"
   - Prevention: Use NotebookEdit tool or proper JSON manipulation, not manual string editing

2. **LangSmith Dataset Conflicts:**
   - Problem: 409 Conflict when re-running notebook (dataset already exists)
   - Fix: Try/except pattern with `read_dataset()` fallback
   - Benefit: Idempotent notebook execution (can run multiple times safely)

3. **Synthetic Data Quality:**
   - LLM-generated questions should test diverse RAG aspects:
     * Specific product queries ("best wireless headphones under $100")
     * Comparison questions ("compare X vs Y")
     * Feature-based queries ("headphones with noise cancellation")
     * Constraint-based queries ("laptop with 16GB RAM")
   - Include `answer_example` to show expected response quality
   - Reference context IDs enable retrieval accuracy measurement

4. **Environment Variable Hygiene:**
   - Always use `load_dotenv()` at the start of notebooks
   - Check for missing keys before API calls: `os.environ.get("KEY", "default")`
   - Use `.env.example` to document required variables
   - Never commit actual `.env` files

5. **LangSmith Dataset Best Practices:**
   - Use descriptive dataset names: "rag-evaluation-dataset", not "test1"
   - Include comprehensive descriptions for future reference
   - Store both inputs AND expected outputs for full evaluation
   - Reference contexts enable measuring retrieval accuracy separately from generation quality
   - Versioning: Create new datasets for major changes ("rag-eval-v2")

**Future Enhancements:**

1. **Automated Evaluation Pipeline:**
   - Run RAG pipeline against all dataset questions
   - Compare actual answers vs ground truth
   - Measure metrics: answer similarity, retrieval precision/recall, latency

2. **Human-in-the-Loop Validation:**
   - Review LLM-generated questions for realism
   - Add manually curated edge cases
   - Validate answer_examples for correctness

3. **Continuous Evaluation:**
   - Run evaluation suite on every code change (CI/CD)
   - Track metrics over time (Grafana dashboard)
   - Alert when performance degrades below threshold

4. **Dataset Expansion:**
   - Generate more diverse questions (100+ examples)
   - Include failure cases (questions with no good answer)
   - Multi-turn conversations (follow-up questions)

### Critical Implementation Details

**API Endpoint Structure:**
- `POST /chat` expects: `{"provider": str, "model_name": str, "messages": list[dict]}`
- Returns: `{"message": str}`
- Provider-specific handling in `run_llm()` - Google uses different API pattern (contents array)

**Provider Model Names:**
- OpenAI: `gpt-4o-mini`, `o1-mini` (includes `reasoning_effort="low"` parameter)
- Groq: `llama-3.3-70b-versatile`
- Google: `gemini-2.0-flash-exp`

**Session State Management:**
- Streamlit maintains conversation history in `st.session_state.messages`
- Format: `[{"role": "user"|"assistant", "content": str}]`
- Initial message at `chatbot_ui/app.py:58`

**Docker Volume Mounts:**
- API source: `./apps/api/src:/app/apps/api/src` (hot reload)
- UI source: `./apps/chatbot_ui/src:/app/apps/chatbot_ui/src` (hot reload)
- Qdrant storage: `./qdrant_storage:/qdrant/storage:z` (persistent vectors)

### Data Pipeline Architecture

**Week 1 Dataset Processing:**
1. Raw data: Amazon Electronics reviews (2.5M+ records)
2. Filter: Products from 2022-2023 with valid categories
3. Filter: 100+ ratings threshold → 17,162 products
4. Sample: Reproducible 1,000-item subset (seed=42)
5. Output: `data/meta_Electronics_2022_2023_with_category_ratings_over_100_sample_1000.jsonl`

**RAG Preprocessing Flow:**
1. Load 1,000-item sample
2. Combine `title + features` into rich descriptions
3. Generate embeddings via OpenAI API
4. Create Qdrant points: `PointStruct(id, vector, payload)`
5. Batch upsert to collection (wait=True for indexing)
6. Enable semantic search via `query_points(query_embedding, limit=k)`

## Development Guidelines

### Environment Variables
Required in `.env` (use `env.example` as template):
```env
OPENAI_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
API_URL=http://api:8000  # or http://localhost:8000 for local dev
```

### Adding New LLM Provider
1. Add API key to config classes in both `api/core/config.py` and `chatbot_ui/core/config.py`
2. Initialize client at module level in `api/app.py` (lines 19-22)
3. Add provider case to `run_llm()` function (line 25)
4. Add provider to Streamlit sidebar dropdown (line 44)
5. Add model options for provider (lines 45-50)

### Working with Qdrant
- Collection persists in `./qdrant_storage/` (gitignored)
- To reset: `rm -rf qdrant_storage/` and restart Docker
- Vector dimensions must match embedding model (1536 for text-embedding-3-small)
- Payload stores full product metadata - no separate DB needed

### Notebook Hygiene
**CRITICAL**: Always run `make clean-notebook-outputs` before committing notebooks. Outputs can contain:
- Large embedded data
- API responses
- Execution state

### Testing Strategy
No test suite currently exists. When adding tests:
- Place in `apps/api/tests/` and `apps/chatbot_ui/tests/`
- Use `pytest` as framework (common in Python ecosystem)
- Add `pytest` to root workspace: `uv add --dev pytest`

## Common Patterns

### Extending the API
Add endpoints to `apps/api/src/api/app.py`:
```python
@app.post("/new-endpoint")
def new_handler(payload: RequestModel) -> ResponseModel:
    # Implementation
    return ResponseModel(...)
```

### Modifying UI Components
Streamlit uses imperative rendering - all changes to `apps/chatbot_ui/src/chatbot_ui/app.py` hot reload automatically.

### Adding Dependencies
- Shared by all apps: `uv add <package>`
- API-specific: `uv add --package api <package>`
- UI-specific: `uv add --package chatbot-ui <package>`

## Troubleshooting

**Port conflicts:**
```bash
lsof -i :8000  # API
lsof -i :8501  # Streamlit
lsof -i :6333  # Qdrant
```

**API connection errors from UI:**
- Check `API_URL` in `.env` matches running service
- In Docker: use `http://api:8000` (service name)
- Local dev: use `http://localhost:8000`

**Docker build failures:**
```bash
docker compose down
docker system prune -f
docker compose up --build
```

**Qdrant connection issues:**
- Verify service is running: `docker compose ps`
- Check logs: `docker compose logs qdrant`
- Ensure port 6333 not already in use

## Important Notes

- **Never commit** `.env` files or Jupyter notebook outputs
- **Raw datasets** not included in repo - download from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html)
- **Qdrant storage** persists between Docker restarts - delete `qdrant_storage/` to reset
- **uv workspace** - dependency changes require `uv sync` at root level
- **Git branch strategy** - `main` is the primary branch for all development

---

## 📚 Global Configuration Reference

For global Claude Code configuration (MCP Server Usage Guidelines, Behavioral Rules, etc.), see:
- **Global CLAUDE.md**: `/Users/christopher/.claude/CLAUDE.md`
- **Serena Prompts**: `/Users/christopher/.claude/docs/SERENA-PROMPTS.md`

This separation reduces memory usage by avoiding duplication of framework-wide rules.
