# RAG Pipeline Implementation

Core business logic for the RAG (Retrieval-Augmented Generation) pipeline, combining semantic search with LLM-powered generation to answer product questions.

## Overview

This directory contains the production implementation of the RAG pipeline - a 5-step workflow that takes a user question and returns a natural language answer grounded in actual product data.

## Architecture

```
apps/api/src/api/agents/
├── __init__.py
└── retrieval_generation.py  # Complete RAG pipeline (500+ lines)
```

## RAG Pipeline Workflow

The `retrieval_generation.py` file implements a complete RAG pipeline as a sequence of 5 functions, each handling a specific stage:

```
User Question
    ↓
1. Get Embedding (question → vector)
    ↓
2. Retrieve Data (vector → similar products)
    ↓
3. Process Context (products → formatted text)
    ↓
4. Build Prompt (text + question → LLM prompt)
    ↓
5. Generate Answer (prompt → natural language response)
    ↓
Natural Language Answer
```

## Step-by-Step Breakdown

### 1. Get Embedding (`get_embedding`)

**Purpose**: Convert text query into 1536-dimensional vector for semantic search

**Input**: `query: str` - User's question (e.g., "best wireless headphones")

**Output**: `list[float]` - 1536-dim embedding vector

**How It Works**:
```python
response = openai.embeddings.create(
    model="text-embedding-3-small",  # Must match preprocessing model
    input=query
)
return response.data[0].embedding  # [0.1, -0.3, 0.7, ...]
```

**Why This Model**:
- Same as preprocessing: `text-embedding-3-small` (1536 dimensions)
- Different models = different vector spaces = poor retrieval
- OpenAI's smaller, faster model (vs `text-embedding-3-large`)

**LangSmith Tracing**:
- Decorated with `@traceable(name="Get Embedding", run_type="embedding")`
- Manually tracks token usage: `current_run.metadata["usage_metadata"]`

**Cost**: ~$0.00002 per 1K tokens

---

### 2. Retrieve Data (`retrieve_data`)

**Purpose**: Find k most similar products using cosine similarity

**Input**:
- `query_embedding: list[float]` - Vector from step 1
- `top_k: int = 5` - Number of products to retrieve

**Output**: `dict` with keys:
- `retrieved_context_ids: list[str]` - Product ASINs
- `retrieved_context: list[str]` - Product descriptions
- `similarity_scores: list[float]` - Cosine similarity scores (0-1)

**How It Works**:
```python
results = qdrant_client.query_points(
    collection_name="Amazon-items-collection-00",
    query=query_embedding,
    limit=top_k,
    with_payload=True
)
```

**Qdrant Search**:
- Algorithm: HNSW (Hierarchical Navigable Small World) - approximate nearest neighbor
- Distance Metric: Cosine similarity (measures angle between vectors)
- Speed: ~10-50ms for 1000-item collection

**Return Structure**:
```python
{
    "retrieved_context_ids": ["B09X12ABC", "B08Y34DEF", ...],
    "retrieved_context": ["Wireless headphones...", "Bluetooth speaker...", ...],
    "similarity_scores": [0.89, 0.82, 0.78, 0.71, 0.65]
}
```

**Similarity Scores**:
- 0.9-1.0: Extremely relevant (exact match)
- 0.7-0.9: Very relevant (semantic match)
- 0.5-0.7: Somewhat relevant (related topic)
- <0.5: Low relevance (consider filtering out)

**LangSmith Tracing**:
- Decorated with `@traceable(name="Retrieve Data", run_type="retriever")`
- Captures query embedding, top_k, retrieved product IDs and scores

---

### 3. Process Context (`process_context`)

**Purpose**: Convert structured product data into human-readable text for LLM

**Input**: Parallel lists from step 2:
- `asin_list: list[str]` - Product IDs
- `descriptions_list: list[str]` - Product descriptions
- `ratings_list: list[float]` - Product ratings

**Output**: `str` - Formatted context

**How It Works**:
```python
formatted_lines = []
for asin, rating, desc in zip(asin_list, ratings_list, descriptions_list):
    formatted_lines.append(f"- ID: {asin}, rating: {rating}, description: {desc}")
return "\n".join(formatted_lines)
```

**Example Output**:
```
- ID: B09X12ABC, rating: 4.5, description: Sony WH-1000XM5 Wireless Headphones with noise cancellation
- ID: B08Y34DEF, rating: 4.3, description: Bose QuietComfort 45 Bluetooth Headphones
```

**Why Format as Text**:
- LLMs process text, not structured data (lists/dicts)
- Consistent format makes prompting easier
- Includes key metadata (ID, rating) for verification

**LangSmith Tracing**:
- Decorated with `@traceable(name="Format Retrieved Context", run_type="prompt")`
- Shows transformation from lists to formatted string

---

### 4. Build Prompt (`build_prompt`)

**Purpose**: Construct LLM prompt with system instructions, context, and question

**Input**:
- `formatted_context: str` - From step 3
- `question: str` - Original user question

**Output**: `list[dict]` - OpenAI chat messages format

**Prompt Structure**:
```python
[
    {
        "role": "system",
        "content": """You are a helpful shopping assistant.
Answer the user's question based ONLY on the retrieved product context.
If the context doesn't contain relevant information, say so clearly.

Retrieved Product Context:
{formatted_context}
"""
    },
    {
        "role": "user",
        "content": question
    }
]
```

**Prompt Engineering Decisions**:
1. **System Role**: "Shopping assistant" sets helpful, product-focused tone
2. **Grounding Constraint**: "ONLY on the retrieved product context" prevents hallucination
3. **Honesty Instruction**: "Say so clearly" if context insufficient (vs making up answers)
4. **Context Placement**: In system message (not user message) for stronger adherence

**LangSmith Tracing**:
- Decorated with `@traceable(name="Build Prompt", run_type="prompt")`
- Captures full prompt structure sent to LLM

---

### 5. Generate Answer (`generate_answer`)

**Purpose**: Use LLM to generate natural language answer from prompt

**Input**: `messages: list[dict]` - From step 4

**Output**: `str` - Natural language response

**How It Works**:
```python
response = openai.chat.completions.create(
    model="gpt-5-nano",
    messages=messages,
    reasoning_effort="minimal"
)
return response.choices[0].message.content
```

**Model Selection**:
- **gpt-5-nano**: Smallest, fastest OpenAI model
- **Why Nano**: RAG task is retrieval-based (not creative generation)
- **reasoning_effort="minimal"**: Further optimize for speed/cost
- **Trade-off**: Faster and cheaper, but less creative than larger models

**LangSmith Tracing**:
- Decorated with `@traceable(name="Generate Answer", run_type="llm")`
- Manually tracks token usage:
  - `input_tokens`: Prompt length
  - `output_tokens`: Answer length
  - `total_tokens`: Sum (used for cost calculation)

**Cost**: ~$0.001 per 1K tokens (approximate)

---

### 6. RAG Pipeline (`rag_pipeline`)

**Purpose**: Orchestrate all 5 steps and return comprehensive result

**Input**: `question: str` - User's question

**Output**: `dict` with keys:
```python
{
    "answer": str,                      # Natural language response
    "question": str,                    # Original query (for validation)
    "retrieved_context_ids": list[str], # Product ASINs
    "retrieved_context": list[str],     # Product descriptions
    "similarity_scores": list[float]    # Cosine similarity scores
}
```

**Why Return Extra Metadata**:
- **Internal observability**: Logging and debugging
- **Evaluation**: RAGAS metrics need retrieved context and IDs
- **Future features**: Frontend could display "Products used in this answer"
- **Analytics**: Track which products are most frequently retrieved

**LangSmith Tracing**:
- Decorated with `@traceable(name="RAG Pipeline")` - root span
- All 5 steps appear as child spans in trace tree
- Enables viewing full pipeline execution in LangSmith UI

**Execution Flow**:
```python
def rag_pipeline(question: str) -> dict:
    # Step 1: Question → Embedding
    query_embedding = get_embedding(question)

    # Step 2: Embedding → Similar Products
    retrieval_result = retrieve_data(query_embedding)

    # Step 3: Products → Formatted Text
    formatted_context = process_context(
        retrieval_result["retrieved_context_ids"],
        retrieval_result["retrieved_context"],
        retrieval_result.get("ratings", [])
    )

    # Step 4: Context + Question → Prompt
    messages = build_prompt(formatted_context, question)

    # Step 5: Prompt → Answer
    answer = generate_answer(messages)

    # Return comprehensive result
    return {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieval_result["retrieved_context_ids"],
        "retrieved_context": retrieval_result["retrieved_context"],
        "similarity_scores": retrieval_result.get("similarity_scores", [])
    }
```

## Configuration

### Environment Variables

```env
OPENAI_KEY=sk-...                      # OpenAI API authentication
QDRANT_URL=http://qdrant:6333         # Docker environment
# or
QDRANT_URL=http://localhost:6333      # Local development
```

### Qdrant Connection

**Docker** (API running in container):
```python
qdrant_client = QdrantClient(url="http://qdrant:6333")
```

**Local** (API running on host):
```python
qdrant_client = QdrantClient(url="http://localhost:6333")
```

**Why Different URLs**:
- Docker: `qdrant` is service name in Docker Compose network
- Local: `localhost` refers to host machine
- Trade-off: Need to change URL based on deployment environment

**Better Solution** (not yet implemented):
- Pass `qdrant_url` as parameter to `rag_pipeline()`
- Configure via environment variable
- Default to Docker URL, override for local

### Collection Configuration

```python
collection_name = "Amazon-items-collection-00"
```

**Collection Requirements**:
- Must exist before running pipeline (created in preprocessing notebook)
- Vectors: 1536 dimensions (text-embedding-3-small)
- Payload: `parent_asin`, `description`, `average_rating`, etc.
- Distance: Cosine similarity

## LangSmith Observability

Every function decorated with `@traceable` for comprehensive tracing:

### Trace Hierarchy
```
RAG Pipeline (root span)
├── Get Embedding (50ms)
├── Retrieve Data (100ms)
├── Format Retrieved Context (5ms)
├── Build Prompt (2ms)
└── Generate Answer (2000ms)
```

### What LangSmith Captures
- **Inputs**: Function arguments
- **Outputs**: Return values
- **Execution Time**: Start/end timestamps
- **Token Usage**: Manually tracked for OpenAI calls
- **Errors**: Stack traces if functions fail
- **Metadata**: Custom annotations (model names, run types)

### Manual Token Tracking

```python
current_run = get_current_run_tree()  # Get active trace context
if current_run:
    current_run.metadata["usage_metadata"] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
```

**Why Manual**:
- OpenAI SDK doesn't auto-integrate with LangSmith
- Token counts needed for cost analysis
- LangSmith aggregates tokens across all operations

### Viewing Traces

1. Run RAG pipeline (triggers tracing if `LANGSMITH_TRACING=true`)
2. Open LangSmith UI: https://smith.langchain.com
3. Navigate to project (e.g., "rag-tracing")
4. View trace tree with all 5 steps
5. Inspect inputs, outputs, timing, tokens for each step

## Performance Characteristics

**Typical Latency** (for 1000-item collection):
- Embedding: 50-100ms (OpenAI API call)
- Retrieval: 10-50ms (Qdrant search)
- Formatting: <5ms (text processing)
- Prompt Building: <2ms (string concatenation)
- Generation: 1000-3000ms (OpenAI API call, depends on answer length)
- **Total**: ~1-3 seconds

**Bottlenecks**:
1. **Generation** (80% of total time) - LLM API call
2. **Embedding** (5-10%) - OpenAI API call
3. **Retrieval** (1-5%) - Qdrant search

**Optimization Opportunities**:
1. **Caching embeddings** for common questions
2. **Connection pooling** for Qdrant client
3. **Streaming responses** (return answer as generated, not all-at-once)
4. **Smaller LLM** (gpt-5-nano → gpt-4.1-nano if available)

## Known Issues

### 1. Hardcoded Qdrant URL

**Problem**: URL changes between Docker and local development

**Current**: Manually change in code
```python
qdrant_client = QdrantClient(url="http://qdrant:6333")  # Docker
# vs
qdrant_client = QdrantClient(url="http://localhost:6333")  # Local
```

**Better Solution**: Environment variable
```python
from api.core.config import config
qdrant_client = QdrantClient(url=config.QDRANT_URL)
```

### 2. New Client Per Request

**Problem**: `QdrantClient` created on every `rag_pipeline()` call

**Impact**: Slower due to connection overhead

**Fix**: Create client once at module level
```python
# At top of file (module-level)
qdrant_client = QdrantClient(url=config.QDRANT_URL)

def retrieve_data(query_embedding, top_k=5):
    # Use existing client (no re-initialization)
    results = qdrant_client.query_points(...)
```

### 3. No Error Handling

**Problem**: Functions don't handle errors gracefully

**Impact**: Crashes on API failures, network issues, missing collections

**Example Errors**:
- OpenAI API timeout/rate limit
- Qdrant connection failure
- Empty retrieval results

**Fix** (not implemented):
```python
try:
    response = openai.embeddings.create(...)
except openai.APIError as e:
    logger.error(f"OpenAI API error: {e}")
    raise HTTPException(status_code=503, detail="LLM service unavailable")
```

### 4. No Input Validation

**Problem**: No length limits, sanitization, or validation on `question`

**Impact**: Long queries → high costs, malicious input → injection attacks

**Fix** (not implemented):
```python
def rag_pipeline(question: str) -> dict:
    if len(question) > 500:
        raise ValueError("Question too long (max 500 chars)")
    if not question.strip():
        raise ValueError("Question cannot be empty")
    # ... rest of pipeline
```

## Testing

**No Unit Tests** (intentional MVP scope)

**How to Test**:
1. Run evaluation suite: `make run-evals-retriever`
2. Manual testing via API docs: http://localhost:8000/docs
3. LangSmith traces for debugging individual queries

**When to Add Tests**:
- Before production deployment
- After adding error handling
- When refactoring pipeline logic

## Related Documentation

- **Parent API**: [../../README.md](../../README.md) - API architecture
- **Evaluation**: [../../evals/README.md](../../evals/README.md) - Testing RAG quality
- **Preprocessing**: [../../../../notebooks/week1/README.md](../../../../notebooks/week1/README.md) - Data pipeline
- **Project Root**: [../../../../README.md](../../../../README.md) - Overall architecture

## Further Reading

- **RAG Pattern**: https://arxiv.org/abs/2005.11401
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **LangSmith Tracing**: https://docs.smith.langchain.com/tracing
- **Prompt Engineering**: https://platform.openai.com/docs/guides/prompt-engineering
