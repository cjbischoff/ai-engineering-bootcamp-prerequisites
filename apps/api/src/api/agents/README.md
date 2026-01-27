# RAG Pipeline Implementation

Core business logic for the RAG (Retrieval-Augmented Generation) pipeline, combining semantic search with LLM-powered generation to answer product questions.

## Overview

This directory contains the production implementation of the RAG pipeline - a 5-step workflow that takes a user question and returns a natural language answer grounded in actual product data.

## Architecture

```
apps/api/src/api/agents/
├── __init__.py
├── retrieval_generation.py      # Complete RAG pipeline (500+ lines)
├── utils/
│   ├── __init__.py
│   └── prompt_management.py     # Prompt loading utilities (Video 7)
└── prompts/
    └── retrieval_generation.yaml  # RAG prompt configuration (Video 7)
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
- `formatted_context: str` - From step 3 (preprocessed_context)
- `question: str` - Original user question

**Output**: `str` - Rendered prompt string

**Implementation (Video 7 Refactoring)**:
```python
from api.agents.utils.prompt_management import prompt_template_config

def build_prompt(preprocessed_context, question):
    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )
    prompt = template.render(
        preprocessed_context=preprocessed_context,
        question=question
    )
    return prompt
```

**What Changed (Video 7)**:
- ❌ **Before**: 60+ lines of hardcoded prompt text in Python
- ✅ **After**: 8 lines loading template from YAML configuration file
- ✅ **Benefits**: Version control, easier editing, separation of concerns
- ✅ **Template Engine**: Jinja2 for variable substitution (`{{ variable }}`)

**Prompt Template (YAML)**:
```yaml
# apps/api/src/api/agents/prompts/retrieval_generation.yaml
metadata:
  name: Retrieval Generation Prompt
  version: 1.0.0
  description: Retrieval Generation Prompt for RAG Pipeline
  author: Christoper Bischoff

prompts:
  retrieval_generation: |
    You are a shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - You need to answer the question based on the provided context only.
    - Never use word context and refer to it as the available products.
    - As an output you need to provide:

    * The answer to the question based on the provided context.
    * The list of the IDs of the chunks that were used to answer the question.
    * Short description (1-2 sentences) of the item based on the description.

    - The short description should have the name of the item.
    - The answer should contain detailed information and specification in bullet points.

    Context:
    {{ preprocessed_context }}

    Question:
    {{ question }}
```

**Prompt Engineering Decisions**:
1. **System Role**: "Shopping assistant" sets helpful, product-focused tone
2. **Grounding Constraint**: "based on the provided context only" prevents hallucination
3. **Output Format**: Structured answer with IDs, description, and bullet points
4. **Context Placement**: In template (clearer separation from code logic)

**Jinja2 Template Syntax**:
- `{{ preprocessed_context }}` - Variable substitution
- `{{ question }}` - Variable substitution
- `|` in YAML - Multiline string literal (preserves newlines)

**LangSmith Tracing**:
- Decorated with `@traceable(name="Build Prompt", run_type="prompt")`
- Captures full rendered prompt string sent to LLM

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

## Prompt Configuration Management (Video 7)

### Overview

Video 7 introduced externalized prompt configuration using YAML files and Jinja2 templates, replacing hardcoded prompts with a maintainable, version-controlled system.

### Directory Structure

**`utils/` - Prompt Loading Utilities**
```
apps/api/src/api/agents/utils/
├── __init__.py                    # Makes directory a Python package
└── prompt_management.py           # Centralized prompt loading functions
```

**`prompts/` - YAML Configuration Files**
```
apps/api/src/api/agents/prompts/
└── retrieval_generation.yaml      # RAG prompt with metadata
```

### Utility Functions (prompt_management.py)

**1. `prompt_template_config()` - Load from Local YAML**

```python
import yaml
from jinja2 import Template

def prompt_template_config(yaml_file, prompt_key):
    """Load prompt template from YAML configuration file.

    Args:
        yaml_file: Path to YAML file (relative to project root)
        prompt_key: Key in YAML's 'prompts:' dictionary

    Returns:
        Jinja2 Template object ready for rendering
    """
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)          # Parse YAML

    template_content = config["prompts"][prompt_key]  # Extract template
    template = Template(template_content)      # Create Jinja2 template

    return template
```

**Usage**:
```python
template = prompt_template_config(
    "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
    "retrieval_generation"
)

prompt = template.render(
    preprocessed_context="- Product A\n- Product B",
    question="What is Product A?"
)
```

**2. `prompt_template_registry()` - Load from LangSmith**

```python
from langsmith import Client

ls_client = Client()

def prompt_template_registry(prompt_name):
    """Load prompt from LangSmith prompt registry.

    Args:
        prompt_name: Name of prompt in LangSmith registry

    Returns:
        Jinja2 Template object ready for rendering
    """
    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template
    template = Template(template_content)

    return template
```

**Usage**:
```python
template = prompt_template_registry("retrieval-generation")

prompt = template.render(
    preprocessed_context="...",
    question="..."
)
```

### YAML Configuration Structure

**retrieval_generation.yaml**:
```yaml
metadata:                           # Prompt documentation
  name: Retrieval Generation Prompt  # Human-readable name
  version: 1.0.0                    # Semantic versioning
  description: Retrieval Generation Prompt for RAG Pipeline
  author: Christoper Bischoff       # Author for attribution

prompts:                            # Dictionary of prompt templates
  retrieval_generation: |           # Key for lookup
    You are a shopping assistant that can answer questions about the products in stock.

    Context:
    {{ preprocessed_context }}      # Jinja2 variable

    Question:
    {{ question }}                  # Jinja2 variable
```

**Key Components**:
1. **metadata**: Version control and documentation
2. **prompts**: Dictionary containing multiple prompt templates
3. **Jinja2 syntax**: `{{ variable }}` for variable substitution
4. **YAML `|` operator**: Multiline string literal (preserves formatting)

### Benefits of Externalized Prompts

**Code Quality**:
- ✅ Reduced LOC: 60-line function → 8-line function (-87%)
- ✅ Cleaner code: Logic focused, not prompt text
- ✅ Easier testing: Mock template loader vs multiline string
- ✅ Better reviews: Prompt changes in YAML diffs, not Python diffs

**Collaboration**:
- ✅ Non-engineer friendly: YAML is human-readable
- ✅ Parallel work: Engineers on logic, prompt engineers on prompts
- ✅ Clear ownership: Prompt files owned by prompt engineering team
- ✅ Reduced merge conflicts: Less code overlap

**Versioning**:
- ✅ Semantic versioning: 1.0.0 → 1.1.0 for prompt updates
- ✅ Git history: Clear prompt evolution in YAML file
- ✅ Rollback: Revert to previous YAML version easily
- ✅ Documentation: Metadata tracks author, description, version

**Deployment**:
- ✅ Faster iteration: Change YAML without code deployment
- ✅ A/B testing: Load different prompts at runtime
- ✅ Registry integration: LangSmith for cloud-based management
- ✅ Hot reload: YAML changes picked up by FastAPI auto-reload

### Migration Pattern (Before → After)

**Before (Hardcoded in Python)**:
```python
def build_prompt(preprocessed_context, question):
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
[... 50+ more lines ...]

Context:
{preprocessed_context}

Question:
{question}
"""
    return prompt
```

**After (YAML + Jinja2)**:
```python
from api.agents.utils.prompt_management import prompt_template_config

def build_prompt(preprocessed_context, question):
    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )
    prompt = template.render(
        preprocessed_context=preprocessed_context,
        question=question
    )
    return prompt
```

### Docker Considerations

**File Path Resolution**:
- Working directory: `/app` (in Docker container)
- Volume mount: `./apps/api/src:/app/apps/api/src`
- Same relative path works in both local and Docker environments

**Example**:
```python
# This path works in both local development and Docker
yaml_file = "apps/api/src/api/agents/prompts/retrieval_generation.yaml"
```

**Why It Works**:
- Local: Project root is `/Users/christopher/.../ai-engineering-bootcamp-prerequisites_me/`
- Docker: Project root is `/app/` with volume mount preserving structure
- Relative paths from project root work consistently

### Performance Impact

**YAML Loading Overhead**:
- File I/O: ~1ms per load
- YAML parsing: ~1ms
- Template creation: <1ms
- **Total: ~3ms per request**

**Optimization (Future Enhancement)**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def prompt_template_config_cached(yaml_file, prompt_key):
    """Cached version: loads YAML once, reuses template."""
    # Same implementation as above
    return template
```

**Impact**:
- First call: ~3ms (load + parse)
- Subsequent calls: <0.01ms (cache hit)
- FastAPI hot reload: Cache invalidates automatically

### Testing Prompts

**Unit Test**:
```python
def test_prompt_template_config():
    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )

    prompt = template.render(
        preprocessed_context="Test context",
        question="Test question"
    )

    assert "Test context" in prompt
    assert "Test question" in prompt
    assert "shopping assistant" in prompt.lower()
```

**Smoke Test** (`make smoke-test`):
- Verifies end-to-end RAG pipeline with template-based prompts
- Tests actual API response structure and content
- Located in `scripts/smoke_test.py`

### Further Reading

- **Jinja2 Templates**: https://jinja.palletsprojects.com/templates/
- **YAML Specification**: https://yaml.org/spec/1.2.2/
- **LangSmith Prompts**: https://docs.smith.langchain.com/prompts
- **Semantic Versioning**: https://semver.org/

## Testing

**No Unit Tests** (intentional MVP scope)

**How to Test**:
1. Run evaluation suite: `make run-evals-retriever`
2. Manual testing via API docs: http://localhost:8000/docs
3. LangSmith traces for debugging individual queries
4. Smoke test for end-to-end validation: `make smoke-test`

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
