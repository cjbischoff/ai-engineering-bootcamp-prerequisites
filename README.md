# AI Engineering Bootcamp Prerequisites

This repository contains prerequisite materials and a complete AI chatbot application stack for the AI Engineering Bootcamp, featuring a FastAPI backend and Streamlit frontend.

## Features

- **FastAPI Backend**: Multi-provider LLM API service supporting OpenAI, Groq, and Google GenAI
- **Streamlit Frontend**: Interactive chatbot UI with provider selection
- **Vector Database**: Qdrant for semantic search and RAG operations
- **Docker Support**: Containerized deployment with Docker Compose
- **Workspace Architecture**: Modular monorepo structure with `uv` package manager
- **Jupyter Notebooks**: Interactive tutorials for LLM APIs, dataset exploration, and RAG preprocessing

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker and Docker Compose
- API Keys for:
  - OpenAI (optional, but quota may be exceeded)
  - Groq (recommended)
  - Google GenAI (recommended)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-engineering-bootcamp-prerequisites_me
```

### 2. Configure Environment Variables

```bash
cp env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
API_URL=http://api:8000
```

**⚠️ Important:** Never commit your `.env` file with real API keys!

### 3. Install Dependencies

```bash
uv sync
```

### 4. Run with Docker Compose

```bash
make run-docker-compose
```

Or manually:
```bash
uv sync
docker compose up --build
```

### 5. Access the Applications

- **Chatbot UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Qdrant API**: http://localhost:6333

## Project Structure

```
.
├── apps/
│   ├── api/                        # FastAPI Backend
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── src/api/
│   │       ├── app.py              # Main FastAPI application
│   │       └── core/
│   │           └── config.py       # Configuration management
│   │
│   └── chatbot_ui/                 # Streamlit Frontend
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── src/chatbot_ui/
│           ├── app.py              # Streamlit UI application
│           └── core/
│               └── config.py       # Configuration management
│
├── notebooks/
│   ├── week0/
│   │   └── 01-llm-apis.ipynb       # LLM API tutorials
│   └── week1/
│       ├── 01-explore-amazon-dataset.ipynb  # Dataset exploration
│       ├── 02-RAG-preprocessing-Amazon.ipynb # RAG preprocessing & embeddings
│       ├── 03-RAG-pipeline.ipynb            # RAG pipeline implementation
│       └── 04-evaluation-dataset.ipynb      # Evaluation dataset creation
│
├── qdrant_storage/                 # Qdrant persistent storage (gitignored)
├── docker-compose.yml              # Multi-service orchestration
├── Makefile                        # Common commands
├── pyproject.toml                  # Root workspace configuration
└── .env                            # Environment variables (not tracked)
```

## Week 1: Dataset Exploration

### Sprint 0 / Video 1: Dataset Exploration

Week 1 focuses on exploratory data analysis of the Amazon Electronics reviews dataset.

**Dataset Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html)

**Notebook:** `notebooks/week1/01-explore-amazon-dataset.ipynb`

**Analysis Pipeline:**
1. Load and explore raw metadata (1.61M products)
2. Filter products first observed in 2022 or later
3. Remove products without valid categories
4. Analyze distribution across main categories
5. Filter products with 100+ ratings (17,162 items)
6. Create reproducible 1,000-item sample
7. Extract corresponding review records

**Final Datasets:**
- `meta_Electronics_2022_2023_with_category_ratings_over_100.jsonl` (93MB) - 17,162 products
- `meta_Electronics_2022_2023_with_category_ratings_over_100_sample_1000.jsonl` (5.4MB) - 1,000 products
- `Electronics_2022_2023_with_category_ratings_100_sample_1000.jsonl` (55MB) - Reviews for sample

**Downloading Raw Data:**
To run the complete analysis pipeline, download the raw datasets:
1. Visit https://amazon-reviews-2023.github.io/main.html
2. Download `Electronics.jsonl.gz` and `meta_Electronics.jsonl.gz`
3. Extract to `data/` directory
4. Run the notebook to regenerate all intermediate files

### Sprint 0 / Video 2: RAG Preprocessing & Vector Database

This sprint implements the preprocessing pipeline and vector database infrastructure for Retrieval-Augmented Generation (RAG).

**Notebook:** `notebooks/week1/02-RAG-preprocessing-Amazon.ipynb`

**What Was Done:**

#### 1. Data Preprocessing Pipeline
The notebook implements a complete ETL (Extract, Transform, Load) pipeline for preparing product data for semantic search:

**Data Loading:**
- Reads the 1,000-item sample dataset (`meta_Electronics_2022_2023_with_category_ratings_over_100_sample_1000.jsonl`)
- Uses pandas with `lines=True` parameter for JSONL format
- Preserves all product metadata including ratings, prices, images, and features

**Text Preprocessing:**
- **Description Creation**: Combines product `title` and `features` into a single searchable description
  - Concatenates title with all feature bullet points
  - Creates rich, keyword-dense text for better semantic matching
  - Example: "RAVODOI USB C Cable... 【Fast Charging Cord】... 【Universal Compatibility】..."

- **Image Extraction**: Extracts the first large image URL from each product's image array
  - Uses `.get("large", "")` for safe extraction with fallback
  - Provides thumbnail-quality images for UI display

**Data Sampling:**
- Randomly samples 50 items from the 1,000-item dataset using `random_state=42` for reproducibility
- Selects essential columns: description, image, rating_number, price, average_rating, parent_asin
- Converts to list of dictionaries using `orient="records"` for easy iteration

#### 2. Vector Embedding Generation

**Embedding Function:**
```python
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding
```

**Why OpenAI text-embedding-3-small:**
- **Efficiency**: 1536-dimensional vectors (smaller than text-embedding-3-large's 3072)
- **Cost-effective**: Lower API costs for development/testing
- **Performance**: Excellent balance of speed and semantic quality
- **Use case**: Perfect for product similarity search and recommendation systems

**Embedding Process:**
- Each product description is converted to a 1536-dimensional vector
- Vectors capture semantic meaning, not just keyword matching
- Similar products cluster together in vector space regardless of exact wording
- Enables searches like "waterproof phone case" to find "water-resistant mobile cover"

#### 3. Qdrant Vector Database Setup

**Why Qdrant:**
- **Open Source**: Free, self-hosted vector database
- **Performance**: Fast similarity search with HNSW (Hierarchical Navigable Small World) algorithm
- **Scalability**: Handles millions of vectors efficiently
- **Persistence**: Data survives container restarts via volume mounting
- **Python-native**: Excellent Python client library with type hints

**Docker Compose Configuration:**
```yaml
qdrant:
  image: qdrant/qdrant
  ports:
    - 6333:6333  # HTTP API
    - 6334:6334  # gRPC API
  volumes:
    - ./qdrant_storage:/qdrant/storage:z
  restart: unless-stopped
```

**Port Configuration:**
- **6333**: HTTP REST API for queries and management
- **6334**: gRPC API for high-performance operations

**Storage:**
- Persistent volume at `./qdrant_storage/` preserves vectors across restarts
- `:z` flag enables SELinux compatibility on RHEL/Fedora systems

#### 4. Collection Creation & Configuration

**Collection Setup:**
```python
qdrant_client.create_collection(
    collection_name="Amazon-items-collection-00",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

**Configuration Choices:**

**Vector Size (1536):**
- Must match OpenAI's text-embedding-3-small output dimension
- Fixed at model level - cannot be changed without re-embedding

**Distance Metric (COSINE):**
- **Why COSINE over Euclidean:** Focuses on direction, not magnitude
- Normalized vectors mean distance represents semantic similarity
- Range: 0 (identical) to 2 (opposite meaning)
- Better for text embeddings where vector length varies

**Alternative Metrics (not used):**
- `Distance.EUCLIDEAN`: Better for absolute differences (image vectors)
- `Distance.DOT`: Faster but requires normalized vectors

#### 5. Data Ingestion Pipeline

**Point Structure:**
```python
PointStruct(
    id=i,                              # Unique integer ID
    vector=get_embedding(description),  # 1536-dim embedding
    payload=data                        # Original product data
)
```

**Payload Strategy:**
- Stores complete product metadata alongside vectors
- Enables retrieval of full product details from search results
- No need for separate database lookups
- Fields: description, image, rating_number, price, average_rating, parent_asin

**Batch Upsert:**
```python
qdrant_client.upsert(
    collection_name="Amazon-items-collection-00",
    wait=True,  # Wait for indexing to complete
    points=pointstructs
)
```

**Why Batch Upsert:**
- More efficient than individual inserts (reduces network overhead)
- `wait=True` ensures data is indexed before proceeding
- Returns `UpdateStatus.COMPLETED` for confirmation

#### 6. Semantic Search Implementation

**Retrieval Function:**
```python
def retrieve_data(query, k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=k
    )
    return results
```

**How It Works:**
1. User query (e.g., "gaming headset with mic") → embedding vector
2. Qdrant finds k-nearest neighbors using HNSW index
3. Returns most semantically similar products with scores
4. Scores represent cosine similarity (higher = more relevant)

**Why This Approach:**
- **Semantic Understanding**: "laptop charger" matches "notebook power adapter"
- **Typo Resilient**: Embeddings are robust to spelling errors
- **Multi-language Potential**: Embeddings can handle multiple languages
- **Context Aware**: Understands "wireless" vs "wired" distinctions

**Performance Characteristics:**
- HNSW index: O(log n) search complexity
- 50 items: Near-instant (<10ms) retrieval
- Scalable to millions of items with minimal degradation

#### 7. Infrastructure Architecture

**Complete Stack:**
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FastAPI   │────▶│   OpenAI     │     │   Qdrant    │
│   Backend   │     │  Embeddings  │────▶│   Vector    │
│   (Port     │     │     API      │     │   Database  │
│    8000)    │     └──────────────┘     │  (Port 6333)│
└─────────────┘                          └─────────────┘
      ▲                                         ▲
      │                                         │
      │                                         │
┌─────────────┐                          ┌─────────────┐
│  Streamlit  │                          │  Persistent │
│     UI      │                          │   Storage   │
│  (Port 8501)│                          │  (./qdrant_ │
└─────────────┘                          │   storage/) │
                                         └─────────────┘
```

**Why This Architecture:**
- **Separation of Concerns**: Each service has a single responsibility
- **Scalability**: Services can be scaled independently
- **Reliability**: Container restart doesn't lose vector data
- **Development**: Can develop/test services in isolation

#### 8. Testing & Validation

**Test Point Structure:**
```python
PointStruct(
    id=0,
    vector=get_embedding("Test text"),
    payload={"text": "Test text", "model": "text-embedding-3-small"}
)
```

**Validation Steps:**
1. Test single embedding generation
2. Verify point structure creation
3. Validate batch embedding pipeline
4. Confirm successful upsert operation
5. Test retrieval with sample queries

**Outputs:**
- All 50 products successfully embedded and stored
- Collection ready for semantic search queries
- Data persisted to `./qdrant_storage/` directory

**Why This Matters for RAG:**
- **Retrieval**: Semantic search finds relevant products for user queries
- **Augmentation**: Retrieved product data augments LLM context
- **Generation**: LLM generates responses using product information
- **Foundation**: This preprocessing enables the complete RAG pipeline

**Next Steps:**
- Integrate semantic search with FastAPI endpoints
- Connect retrieval results to LLM context
- Build product recommendation features
- Implement filtering (price, ratings, categories)

### Sprint 0 / Video 3: RAG Pipeline Implementation

This sprint implements the complete Retrieval-Augmented Generation (RAG) pipeline, enabling semantic product search combined with LLM-powered response generation.

**Notebook:** `notebooks/week1/03-RAG-pipeline.ipynb`

**What Was Done:**

#### 1. RAG Architecture Overview

The RAG pipeline implements a four-stage architecture for intelligent product recommendations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline Flow                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User Query                                                      │
│     "What kind of earphones can I get with ratings above 4.5?"     │
│                              ↓                                      │
│  2. Retrieval (Semantic Search)                                    │
│     ┌─────────────────────────────────────────┐                   │
│     │ Query → Embedding Model                 │                   │
│     │ Vector → ANN Search (Cosine Similarity) │                   │
│     │ Results → Top-K Products                │                   │
│     └─────────────────────────────────────────┘                   │
│                              ↓                                      │
│  3. Augmentation (Context Building)                                │
│     ┌─────────────────────────────────────────┐                   │
│     │ Format retrieved products               │                   │
│     │ Build structured prompt                 │                   │
│     │ Combine with user query                 │                   │
│     └─────────────────────────────────────────┘                   │
│                              ↓                                      │
│  4. Generation (LLM Response)                                      │
│     ┌─────────────────────────────────────────┐                   │
│     │ Prompt → GPT-4o-mini                   │                   │
│     │ Generate product recommendations        │                   │
│     │ Return natural language answer          │                   │
│     └─────────────────────────────────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why This Architecture:**
- **Retrieval**: Semantic search finds relevant products based on meaning, not just keywords
- **Augmentation**: LLM receives concrete product data as context, reducing hallucinations
- **Generation**: LLM synthesizes natural language recommendations from real data
- **Grounding**: All recommendations are backed by actual products in the database

#### 2. Embedding Function

**Implementation:**
```python
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding
```

**Why This Function:**
- **Reusability**: Used for both product descriptions (indexing) and user queries (retrieval)
- **Consistency**: Same model ensures query vectors match product vectors in semantic space
- **Simplicity**: Single-purpose function with clear interface
- **Model Parameter**: Allows testing with different embedding models (text-embedding-3-large, etc.)

**Key Characteristics:**
- Returns 1536-dimensional vector for text-embedding-3-small
- Synchronous API call (suitable for notebook usage)
- No batching (fine for query-time embedding generation)

#### 3. Retrieval Function

**Implementation:**
```python
def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }
```

**Why This Design:**

**Structured Return Value:**
- Returns dictionary with explicit keys for easy access
- Separates IDs, descriptions, ratings, and scores for flexible usage
- Enables downstream filtering or ranking adjustments

**Payload Extraction:**
- Extracts `parent_asin` for product identification and linking
- Retrieves `description` for LLM context (already formatted with title + features)
- Includes `average_rating` for quality assessment
- Captures similarity `score` for relevance ranking

**ANN Search Strategy:**
- Uses `query_points()` for fast approximate nearest neighbor search
- Cosine similarity metric matches collection configuration
- `limit=k` parameter allows flexible result count (default 5)
- HNSW index provides O(log n) search complexity

**How Retrieval Works:**
1. Query text → 1536-dim embedding vector
2. Qdrant compares query vector against all product vectors using cosine similarity
3. HNSW graph algorithm efficiently finds k-nearest neighbors
4. Returns products ordered by similarity score (higher = more relevant)

#### 4. Context Formatting Function

**Implementation:**
```python
def process_context(context):
    formatted_context = ""

    for id, chunk, rating in zip(
        tuple(context["retrieved_context_ids"]),
        context["retrieved_context"],
        context["retrieved_context_ratings"]
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context
```

**Why This Format:**

**Structured Text Representation:**
- Bullet-point list provides clear separation between products
- Includes product ID for traceability and linking
- Shows rating upfront for LLM to assess quality
- Description contains full product details (title + features)

**LLM-Friendly Design:**
- Plain text format is easy for LLMs to parse
- Consistent structure helps LLM extract relevant information
- Newlines separate products clearly
- Compact format minimizes token usage while preserving information

**Example Output:**
```
- ID: B0C142QS8X, rating: 4.5, description: TUNEAKE Kids Headphones...
- ID: B0B67ZFRPC, rating: 3.7, description: QearFun Cat Earbuds...
- ID: B08XYZMQ2Y, rating: 4.6, description: Sony WH-1000XM4...
```

#### 5. Prompt Construction Function

**Implementation:**
```python
def build_prompt(preprocessed_context, question):
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""
    return prompt
```

**Why This Prompt Design:**

**System Role Definition:**
- "Shopping assistant" sets clear expectation for tone and purpose
- Establishes domain expertise in product recommendations

**Explicit Instructions:**
- "Based on the provided context only" prevents hallucinations
- Grounds responses in actual product data
- "Never use word context" ensures natural language ("available products" vs "the context")

**Structured Sections:**
- Clear separation between context and question
- Easy for LLM to identify data source vs. user intent
- F-string interpolation allows dynamic content injection

**Prompt Engineering Principles:**
- **Specificity**: Clear instructions reduce ambiguous responses
- **Constraint**: "Context only" limitation ensures factual accuracy
- **Natural Language**: Avoids technical jargon in output
- **Few-shot Not Needed**: Simple task doesn't require examples

#### 6. Answer Generation Function

**Implementation:**
```python
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content
```

**Why GPT-4o-mini:**
- **Cost-Effective**: Significantly cheaper than GPT-4 (~80% cost reduction)
- **Fast**: Lower latency for real-time chat applications
- **Sufficient Quality**: Product recommendations don't require reasoning-heavy capabilities
- **Availability**: High rate limits suitable for development/testing

**Message Structure:**
- Uses `system` role to provide context and instructions
- Single message contains full prompt (context + question)
- No conversation history needed for stateless recommendations

**API Parameters:**
- `model="gpt-4o-mini"`: Selected for balance of quality and cost
- No `temperature` override (defaults to 1.0 for creative responses)
- No `max_tokens` limit (allows complete responses)
- Note: `reasoning_effort` parameter only available for o1-series models

#### 7. Complete RAG Pipeline Function

**Implementation:**
```python
def rag_pipeline(question, top_k=5):
    qdrant_client = QdrantClient(url="http://localhost:6333")

    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    return answer
```

**Why This Orchestration:**

**Single Entry Point:**
- One function call executes entire RAG pipeline
- Hides implementation complexity from end users
- Easy to integrate into web applications or APIs

**Pipeline Stages:**
1. **Connection**: Initialize Qdrant client (localhost during development)
2. **Retrieval**: Semantic search for top-k relevant products
3. **Formatting**: Convert results to LLM-friendly text format
4. **Prompt Building**: Construct structured prompt with context
5. **Generation**: LLM produces natural language recommendation

**Parameter Design:**
- `question`: User's natural language query
- `top_k=5`: Configurable result count (balances context size vs. relevance)
- Returns: Complete answer string ready for display

**Usage Example:**
```python
answer = rag_pipeline("What kind of earphones can I get with ratings above 4.5?")
print(answer)
```

**Expected Output:**
```
You can get the TUNEAKE Kids Headphones (ID: B0C142QS8X) which have a rating
of 4.5. These are over-ear headphones designed for kids, featuring
volume-limiting technology for hearing protection, a comfortable fit, and a
foldable design for easy storage. They are compatible with all devices that
have a 3.5mm jack.
```

#### 8. RAG Pipeline Benefits

**Compared to Pure LLM:**
- **Factual Accuracy**: Responses based on real product data, not training data
- **Up-to-Date**: Works with current inventory without model retraining
- **Traceable**: Product IDs enable verification and linking
- **Cost-Efficient**: Smaller context than fine-tuning entire product catalog

**Compared to Pure Search:**
- **Semantic Understanding**: "waterproof" matches "water-resistant"
- **Natural Language**: Users can ask questions naturally
- **Synthesis**: LLM combines multiple products into coherent recommendation
- **Context-Aware**: Understands user intent ("for kids", "with mic", etc.)

**Compared to Keyword Search:**
- **Synonym Handling**: "headphones" matches "earbuds", "earphones"
- **Typo Resilient**: Embeddings robust to spelling variations
- **Conceptual Search**: "gaming" finds products with "low latency", "microphone"
- **Multi-Language Potential**: Embeddings can bridge language gaps

#### 9. Testing & Validation

**Test Queries:**
```python
# Rating-based filtering
rag_pipeline("What kind of earphones can I get with ratings above 4.5?")

# Product type search
rag_pipeline("What kids earphones can I get?", top_k=10)

# Feature-based search
rag_pipeline("Wireless headphones with noise cancellation")
```

**Validation Approach:**
1. **Retrieval Quality**: Verify similarity scores are meaningful
2. **Context Formatting**: Ensure all product details are preserved
3. **Prompt Structure**: Validate LLM receives clear instructions
4. **Answer Quality**: Check responses are accurate and helpful
5. **Traceability**: Confirm product IDs match retrieved items

**Performance Characteristics:**
- **Query Latency**: ~200-500ms total (embedding + search + generation)
- **Embedding Generation**: ~100ms (OpenAI API call)
- **Vector Search**: <10ms (Qdrant HNSW index)
- **LLM Generation**: ~100-400ms (GPT-4o-mini)
- **Scalability**: Can handle millions of products with minimal latency increase

#### 10. Integration with Existing Stack

**Connection to FastAPI Backend:**
- RAG functions can be imported into FastAPI endpoints
- Replace hardcoded LLM responses with RAG-enhanced answers
- Maintain existing multi-provider support (OpenAI, Groq, Google)

**Connection to Streamlit UI:**
- Chatbot can display product recommendations with IDs
- UI can render product cards with images and ratings
- Users can click IDs to view full product details

**Production Considerations:**
- **Error Handling**: Add try-except for API failures and empty results
- **Caching**: Cache embeddings for common queries
- **Async Operations**: Use async OpenAI client for better throughput
- **Rate Limiting**: Implement request throttling for API cost control
- **Monitoring**: Track retrieval quality and LLM response accuracy

#### 11. Key Learnings & Next Steps

**What We Built:**
- Complete RAG pipeline from query to response
- Semantic search over product embeddings
- LLM-powered natural language recommendations
- Reusable functions for each pipeline stage

**Next Steps:**
- Integrate RAG pipeline into FastAPI `/chat` endpoint
- Add product filtering (price range, categories, brands)
- Implement conversation history for follow-up questions
- Add product images and links to UI responses
- Experiment with different embedding models and LLMs
- Implement hybrid search (semantic + keyword + filters)
- Add user feedback loop for recommendation quality

**Architecture Foundation:**
This RAG implementation provides the foundation for advanced features:
- **Personalization**: User preference vectors for personalized search
- **Multi-modal**: Image-based product search and comparison
- **Conversational**: Multi-turn dialogue with context retention
- **Analytics**: Track popular queries and products for insights

### Sprint 0 / Video 4: Production RAG API Implementation

This sprint implements the production-ready FastAPI backend for the RAG pipeline, integrating all components from the notebooks into a deployable web service.

**Files:**
- `apps/api/src/api/app.py` - FastAPI application setup and middleware
- `apps/api/src/api/api/endpoints.py` - API route handlers
- `apps/api/src/api/api/models.py` - Request/response schemas
- `apps/api/src/api/api/middleware.py` - Custom middleware (request tracing)
- `apps/api/src/api/agents/retrieval_generation.py` - RAG pipeline implementation

#### 1. Architecture Overview

The production API implements a layered architecture with clear separation of concerns:

```
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Application Stack                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Client Request (POST /rag/)                                      │
│         ↓                                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │ Middleware Layer                                │             │
│  │  1. RequestIDMiddleware (UUID generation)       │             │
│  │  2. CORSMiddleware (cross-origin support)       │             │
│  └─────────────────────────────────────────────────┘             │
│         ↓                                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │ Validation Layer (Pydantic)                     │             │
│  │  - RAGRequest: Validates query field            │             │
│  │  - Auto-rejects malformed requests (422)        │             │
│  └─────────────────────────────────────────────────┘             │
│         ↓                                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │ Routing Layer (APIRouter)                       │             │
│  │  - POST /rag/ → rag() endpoint handler          │             │
│  │  - Extracts query from validated request        │             │
│  └─────────────────────────────────────────────────┘             │
│         ↓                                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │ RAG Pipeline Layer                              │             │
│  │  1. get_embedding(query) → vector               │             │
│  │  2. retrieve_data() → semantic search           │             │
│  │  3. process_context() → format results          │             │
│  │  4. build_prompt() → construct LLM prompt       │             │
│  │  5. generate_answer() → LLM response            │             │
│  └─────────────────────────────────────────────────┘             │
│         ↓                                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │ Response Layer                                  │             │
│  │  - RAGResponse: Serializes answer + request_id  │             │
│  │  - Middleware adds X-Request-ID header          │             │
│  └─────────────────────────────────────────────────┘             │
│         ↓                                                          │
│  Client Response (JSON)                                           │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

#### 2. Key Components

**a) Application Setup ([app.py](apps/api/src/api/app.py))**

The FastAPI application is configured with:
- **Auto-generated Documentation**: OpenAPI schema at `/docs` (Swagger UI) and `/redoc` (ReDoc)
- **Middleware Stack** (order matters - first added = first executed):
  1. `RequestIDMiddleware`: Generates UUID for every request for distributed tracing
  2. `CORSMiddleware`: Enables cross-origin requests from Streamlit frontend (port 8501)
- **Router Registration**: Mounts `api_router` with all RAG endpoints

**Why CORS:**
- Browser security blocks requests between different origins (different ports = different origins)
- Without CORS, Streamlit (port 8501) cannot call API (port 8000)
- Production should restrict `allow_origins` to specific domains, not `["*"]`

**b) Request Tracing Middleware ([middleware.py](apps/api/src/api/api/middleware.py))**

Implements distributed tracing via UUID generation:
- **Pattern**: `BaseHTTPMiddleware` with async `dispatch()` method
- **UUID Generation**: Uses `uuid.uuid4()` for globally unique request IDs
- **Storage**: Attaches ID to `request.state.request_id` (accessible in endpoints)
- **Response Header**: Adds `X-Request-ID` header for client-side tracking
- **Logging**: Records request start/completion with method, path, and request ID

**Benefits:**
- **Debugging**: Filter logs by request ID to trace issues
- **Client Support**: Users can reference request ID in bug reports
- **Distributed Tracing**: Track requests across multiple microservices
- **Performance Monitoring**: Measure end-to-end latency per request

**c) Request/Response Models ([models.py](apps/api/src/api/api/models.py))**

Uses Pydantic for automatic validation and serialization:

```python
class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")

class RAGUsedContext(BaseModel):
    """Product metadata for frontend display (Video 3 enhancement)."""
    image_url: Optional[str] = Field(None, description="The URL of the image of the item")
    price: Optional[float] = Field(None, description="The price of the item")
    description: str = Field(..., description="The description of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The answer to the query")
    used_context: list[RAGUsedContext] = Field(
        ..., description="Information about the items used to answer the query"
    )
```

**Why Pydantic:**
- **Automatic Validation**: FastAPI validates JSON against schema before calling endpoint
- **Type Safety**: Catches type errors at runtime, not in production
- **OpenAPI Generation**: Field descriptions appear in auto-generated API documentation
- **Error Messages**: Returns 422 Unprocessable Entity with detailed validation errors

**Video 3 Enhancement - Rich Product Context:**
- **RAGUsedContext Model**: Represents enriched product information (images, prices, descriptions)
- **Optional Fields**: `image_url` and `price` are Optional to handle nullable Qdrant data gracefully
  - Qdrant may not have images/prices for all products
  - Frontend can show placeholders when fields are None
  - Prevents ValidationError on None values for required float fields
- **Frontend Integration**: Enables visual product cards with images and pricing in the UI
- **Grounding**: Shows users the actual products backing the LLM's recommendations

**d) API Endpoints ([endpoints.py](apps/api/src/api/api/endpoints.py))**

The main RAG endpoint (Video 3 enhanced with product enrichment):

```python
@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> RAGResponse:
    answer = rag_pipeline_wrapper(payload.query)  # Video 3: Uses wrapper for enrichment
    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer["answer"],
        used_context=[
            RAGUsedContext(**used_context) for used_context in answer["used_context"]
        ],
    )
```

**Design Decisions:**
- **APIRouter Pattern**: Groups related endpoints for modularity (easy to add `/rag/health`, `/rag/feedback`)
- **Request Object**: Access middleware-injected `request_id` from `request.state`
- **Return Type**: Pydantic `RAGResponse` automatically serialized to JSON
- **Error Handling**: Not implemented (production would need try/except blocks)

**Video 3 Changes:**
- **Wrapper Function**: Uses `rag_pipeline_wrapper()` instead of `rag_pipeline()` for product metadata enrichment
- **Response Structure**: Returns dict with `answer` and `used_context` fields
- **Context Construction**: Unpacks dict items into `RAGUsedContext` Pydantic models using `**used_context` spread
- **Frontend Data**: Provides image URLs and prices for visual product cards

**e) RAG Pipeline ([retrieval_generation.py](apps/api/src/api/agents/retrieval_generation.py))**

Production implementation of the 5-step RAG workflow from the notebook:

**1. Embedding Generation:**
- Function: `get_embedding(text, model="text-embedding-3-small")`
- Model: OpenAI text-embedding-3-small (1536 dimensions)
- Critical: Must match preprocessing model for semantic space consistency

**2. Vector Retrieval:**
- Function: `retrieve_data(query, qdrant_client, k=5)`
- Connection: `http://qdrant:6333` (Docker Compose service name, not localhost)
- Search: Cosine similarity via `query_points()` with HNSW index
- Returns: Product IDs, descriptions, ratings, similarity scores

**3. Context Formatting:**
- Function: `process_context(context)`
- Format: `- ID: {asin}, rating: {rating}, description: {description}\n`
- Uses: `zip(list1, list2, list3)` - NO tuple() wrapper (TypeError fix)

**4. Prompt Construction:**
- Function: `build_prompt(preprocessed_context, question)`
- Role: "Shopping assistant"
- Constraint: "Only use provided context" (prevents hallucination)
- Structure: System instructions → Context → Question

**5. Answer Generation:**
- Function: `generate_answer(prompt)`
- Model: OpenAI `gpt-5-nano` with `reasoning_effort="minimal"`
- Why nano: Cost-effective for straightforward retrieval-based Q&A
- Message: Single system message with full prompt

**6. Pipeline Orchestration:**
- Function: `rag_pipeline(question, top_k=5)`
- Entry point: Single function call executes entire workflow
- Connection: Creates new Qdrant client per request (inefficient, needs pooling)

**7. Product Enrichment Wrapper (Video 3 Enhancement):**

The `rag_pipeline_wrapper()` function enriches RAG responses with product metadata for rich frontend display:

```python
def rag_pipeline_wrapper(question: str, top_k: int = 5) -> dict:
    """
    Enriches RAG pipeline results with product metadata (images and prices).

    Wrapper pattern separates presentation enrichment from core RAG logic.
    Returns dict with 'answer' (str) and 'used_context' (list of product metadata).
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = rag_pipeline(question, top_k)

    used_context = []
    dummy_vector = np.zeros((1536,)).tolist()

    for item in result.get("references", []):
        # Query Qdrant by product ID using filter
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-00",
            query=dummy_vector,
            limit=1,
            with_payload=True,
            query_filter=Filter(must=[
                FieldCondition(key="parent_asin", match=MatchValue(value=item.id))
            ])
        ).points[0].payload

        used_context.append({
            "image_url": payload.get("image"),
            "price": payload.get("price"),
            "description": item.description
        })

    return {
        "answer": result["answer"],
        "used_context": used_context
    }
```

**Why This Approach:**

- **Wrapper Pattern**: Keeps core `rag_pipeline()` logic unchanged while adding presentation-layer enrichment
- **Separation of Concerns**: RAG logic (retrieval + generation) separated from frontend data fetching
- **Instructor Integration**: Uses structured outputs from `generate_answer()` with `RAGGenerationResponse` model
  - LLM returns answer + list of product references with IDs and descriptions
  - Structured outputs via instructor library ensure reliable JSON parsing

**Technical Implementation:**

- **Qdrant Filtering by ID**: Uses dummy zero vector with `query_filter` to fetch by `parent_asin`
  - Why dummy vector: Qdrant `query_points()` requires a query vector for API compatibility
  - Filter ensures only exact ID match is returned (limit=1)
  - More efficient than semantic search when ID is known

- **Docker Networking**: Uses `http://qdrant:6333` service name, not localhost
  - Docker Compose DNS resolves service names to container IPs
  - Localhost in container context refers to container itself, not other services

- **Graceful Degradation**: Uses `.get()` for nullable fields (image, price)
  - Qdrant data quality varies: some products lack images/prices
  - Returns None instead of KeyError
  - Pydantic Optional[] fields handle None values without validation errors

- **LangSmith Tracing**: Decorated with `@traceable` for observability
  - Tracks enrichment performance separately from core RAG
  - Helps identify bottlenecks in Qdrant metadata fetching

**Performance Considerations:**

- **N+1 Query Problem**: One Qdrant query per product (5 queries for top_k=5)
  - Could be optimized with batch `scroll()` or `retrieve()` if IDs are known upfront
  - Current approach prioritizes code clarity for educational purposes

- **Client Pooling**: Creates new QdrantClient per request
  - Production should use connection pooling for efficiency
  - Consider singleton pattern or dependency injection

**Data Flow:**

1. Call `rag_pipeline()` → Get LLM answer + structured product references (IDs + descriptions)
2. For each product reference → Query Qdrant by ID to fetch image_url and price
3. Construct `used_context` list with enriched product metadata
4. Return dict with `answer` (str) and `used_context` (list) for API response

#### 3. Docker Integration

**Service Communication:**
- API container connects to Qdrant using service name: `http://qdrant:6333`
- Docker Compose creates internal DNS for service-to-service communication
- Localhost would refer to container itself, not Qdrant container

**Volume Mounts for Hot Reload:**
- `./apps/api/src:/app/apps/api/src` - Code changes reflect immediately without rebuild
- `./qdrant_storage:/qdrant/storage:z` - Vector database persists between restarts

#### 4. Lessons Learned

**TypeError with zip() and tuple():**
- **Problem**: `zip(tuple(list1, list2, list3))` is invalid syntax
- **Root Cause**: `tuple()` constructor accepts one iterable, not multiple arguments
- **Fix**: Use `zip(list1, list2, list3)` directly - no tuple wrapper
- **Detection**: Runtime error: `TypeError: tuple expected at most 1 argument, got 3`
- **When**: Multi-line formatting can hide this error until code execution

**Qdrant Connection in Docker:**
- Use service name `http://qdrant:6333`, not `http://localhost:6333`
- Localhost in container context refers to the container itself
- Docker Compose DNS resolves service names to container IPs

**Middleware Order:**
- Middleware added first runs first (outermost layer of onion)
- RequestIDMiddleware before CORS ensures UUID exists before CORS validation
- Response flows back through middleware in reverse order

**Pydantic Validation:**
- FastAPI automatically returns 422 (not 500) for invalid requests
- Field descriptions improve auto-generated documentation quality
- Type hints catch bugs early during development

**Instructor response_model Parameter (Video 3):**
- **Problem**: `KeyError: 'answer'` when instructor doesn't return structured output
- **Root Cause**: Missing `response_model` parameter in `create_with_completion()` call
- **Fix**: Explicitly pass `response_model=RAGGenerationResponse` to instructor
- **Why**: Instructor needs the Pydantic model to know what structure to extract from LLM
- **Detection**: Runtime KeyError when accessing expected dictionary keys

**Pydantic Optional Fields for Nullable Data (Video 3):**
- **Problem**: `ValidationError: price - Input should be a valid number [type=float_type, input_value=None]`
- **Root Cause**: Qdrant data has nullable fields (image, price) but Pydantic expected required values
- **Fix**: Use `Optional[float]` and `Optional[str]` with `Field(None, ...)` for nullable fields
- **Why**: Qdrant data quality varies - some products lack images/prices
- **Benefit**: Graceful degradation - API returns partial data instead of failing validation
- **Frontend Impact**: UI can show placeholders when fields are None

**Qdrant Filter-Based Queries with Dummy Vectors (Video 3):**
- **Technique**: Use `np.zeros((1536,)).tolist()` as query vector with `query_filter`
- **Why Needed**: `query_points()` requires a query vector but we're filtering by exact ID
- **Filter**: `Filter(must=[FieldCondition(key="parent_asin", match=MatchValue(value=id))])`
- **Alternative**: Could use `scroll()` or `retrieve()` for ID-based lookup without vector
- **Trade-off**: Slightly inefficient but maintains API consistency with semantic search

**Import Statement Syntax (Video 3):**
- **Problem**: `import qdrant_client.models import Filter` causes SyntaxError
- **Root Cause**: Invalid Python syntax - mixing import styles
- **Fix**: Use `from qdrant_client.models import Filter, FieldCondition, MatchValue`
- **Detection**: Immediate SyntaxError on file load, not runtime
- **Prevention**: Careful transcription from images, IDE syntax highlighting

**RAG vs Pure LLM:**
- Pure LLM may hallucinate product details or have outdated knowledge
- RAG grounds answers in actual product data from vector database
- Trade-off: Requires vector database setup but provides verifiable, current answers

**Embedding Model Consistency:**
- Critical: Use same model for preprocessing AND query-time embedding
- Different models = different vector spaces = poor retrieval quality
- Dimension mismatch causes Qdrant errors

**Request Tracing Value:**
- UUID in both response body and header enables multiple use cases
- Clients can display: "Error? Reference request ID: abc-123"
- Logs filterable: `grep "request_id: abc-123" logs/`
- Essential for debugging distributed systems

#### 5. Production Considerations

**Not Implemented (Intentional MVP Scope):**
- **Error Handling**: No try/except blocks around API calls or pipeline
- **Rate Limiting**: API unprotected, vulnerable to abuse
- **Timeout Handling**: Long-running queries could hang indefinitely
- **Input Validation**: No query length limits or content sanitization
- **Connection Pooling**: New Qdrant client created per request (inefficient)
- **Caching**: Common queries could be cached to reduce API costs
- **Monitoring**: No metrics on retrieval quality, answer accuracy, latency
- **Authentication**: No API keys or access control
- **Response Streaming**: Answers returned all-at-once, not token-by-token

**When to Add:**
- Error handling: Before ANY production deployment
- Rate limiting: When opening to public users
- Monitoring: When analyzing system performance and quality
- Authentication: When controlling access or implementing billing
- Caching: When reducing OpenAI API costs becomes priority

#### 6. Testing the API

**Using curl:**
```bash
curl -X POST http://localhost:8000/rag/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best wireless headphones?"}'
```

**Using Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:8000/rag/",
    json={"query": "What are the best wireless headphones?"}
)
print(response.json())
```

**Expected Response:**
```json
{
  "request_id": "bf802801-da21-4b61-a10c-e700d4aafe2e",
  "answer": "Based on the available products, I recommend the Sony WH-1000XM4 wireless headphones (ID: B08XYZMQ2Y) with a rating of 4.6. These headphones feature industry-leading noise cancellation, exceptional sound quality, and up to 30 hours of battery life."
}
```

**Validation:**
- Request ID appears in both response body and `X-Request-ID` header
- Answer references actual product IDs from Qdrant collection
- Product details match retrieved context (rating, features)

#### 7. API Documentation

FastAPI auto-generates interactive API documentation:

**Swagger UI (`/docs`):**
- Interactive API explorer with "Try it out" functionality
- Auto-generated from Pydantic models and route definitions
- Shows request/response schemas, field descriptions, validation rules

**ReDoc (`/redoc`):**
- Alternative documentation UI with cleaner layout
- Better for reading and sharing with stakeholders
- Same content as Swagger UI, different presentation

**OpenAPI Schema (`/openapi.json`):**
- Machine-readable API specification
- Can be imported into Postman, Insomnia, or other API clients
- Useful for generating client SDKs in other languages

#### 8. Next Steps

**Immediate Improvements:**
- Add comprehensive error handling to pipeline
- Implement request timeout and retry logic
- Add logging for debugging and monitoring
- Create health check endpoint for orchestration

**Feature Additions:**
- Product filtering by price range, category, rating
- Conversation history for follow-up questions
- Multi-turn dialogue with context retention
- Product image URLs in responses

**Optimization:**
- Connection pooling for Qdrant client
- Caching layer for common queries
- Async OpenAI client for better throughput
- Response streaming for real-time UI updates

**Production Readiness:**
- API key authentication
- Rate limiting per user/IP
- Request/response validation
- Comprehensive test suite
- CI/CD pipeline integration

### Sprint 0 / Video 6: Evaluation Dataset Creation

This sprint implements synthetic evaluation dataset creation for systematic RAG pipeline testing using LangSmith.

**Notebook:** `notebooks/week1/04-evaluation-dataset.ipynb`

**What Was Done:**

#### 1. Overview: Why Evaluation Datasets Matter

**The Problem:**
- RAG pipelines are complex systems with multiple failure points (embedding, retrieval, generation)
- Manual testing is time-consuming and inconsistent
- Hard to measure improvements or detect regressions
- No way to compare different approaches systematically

**The Solution: Evaluation Datasets:**
- **Structured test cases** with known questions and expected answers
- **Repeatable testing** against the same questions over time
- **Objective metrics** for retrieval accuracy and answer quality
- **A/B testing** to compare prompts, models, or retrieval strategies
- **Regression detection** when code changes degrade performance

**Real-World Benefits:**
- Catch bugs before production (e.g., "retrieval returns wrong products")
- Compare GPT-4o vs GPT-5-nano objectively (cost vs quality trade-offs)
- Test prompt changes without guessing ("this prompt reduced errors by 15%")
- Detect when embeddings or vector DB changes break retrieval

#### 2. LangSmith Integration

**What is LangSmith?**
- **Observability platform** specifically built for LLM applications
- Created by LangChain team for debugging and evaluating AI systems
- **Datasets feature** stores test cases for evaluation
- **Traces feature** monitors production RAG pipeline execution (added in Video 5)

**Why Use LangSmith Datasets (vs CSV files)?**
- **Structured storage**: Inputs and outputs clearly separated
- **Versioning**: Track dataset changes over time
- **Integration**: Works with LangSmith evaluation framework
- **Collaboration**: Team can share datasets across projects
- **Web UI**: View and edit datasets visually at smith.langchain.com

#### 3. Synthetic Data Generation with LLMs

**Why Synthetic (LLM-generated) vs Manual?**
- **Speed**: Generate 50 questions in minutes vs hours of manual writing
- **Diversity**: LLM explores product combinations you might not think of
- **Consistency**: Maintains format and quality standards automatically
- **Scalability**: Easy to generate 100s or 1000s of test cases

**The Approach:**
1. Load actual product data from Qdrant
2. Use GPT-4o to generate realistic user questions about these products
3. Include expected answers and reference product IDs
4. Store in LangSmith for systematic evaluation

#### 4. Notebook Implementation

**Cell 1: Environment Setup**
```python
import openai, os, json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()  # CRITICAL: Jupyter doesn't auto-load .env
```

**Why `load_dotenv()`:**
- Jupyter notebooks don't automatically load environment variables from `.env` files
- Must explicitly call `load_dotenv()` to access `OPENAI_KEY`, `LANGSMITH_API_KEY`
- Without this, `KeyError: 'LANGSMITH_API_KEY'` occurs

**Cell 2: Client Initialization**
```python
qdrant_client = QdrantClient(url="http://localhost:6333")  # Local development
client = Client(api_key=os.environ["LANGSMITH_API_KEY"])  # LangSmith client
```

**Cells 3-7: Data Exploration**
- Fetch sample products from Qdrant collection
- Inspect product structure (title, features, ratings, ASINs)
- Select representative products for question generation

**Cells 8-11: Synthetic Question Generation**

**JSON Schema Definition:**
```python
output_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Suggested question"},
            "chunk_ids": {"type": "array", "items": {"type": "string"}},
            "answer_example": {"type": "string", "description": "Expected answer"}
        }
    }
}
```

**Why JSON Schema:**
- **Structured Output**: OpenAI's `response_format` enforces exact format
- **No Parsing Needed**: Direct JSON parsing, no regex or manual extraction
- **Type Safety**: Ensures arrays, strings, objects match expected types
- **Consistency**: Every generated question follows same structure

**LLM Call with Structured Output:**
```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "system", "content": "Generate evaluation questions..."}],
    response_format={"type": "json_schema", "json_schema": output_schema}
)

json_output = json.loads(response.choices[0].message.content)
```

**Why GPT-4o (not GPT-4o-mini):**
- **Higher Quality**: Better at understanding prompt instructions
- **JSON Schema Support**: Reliable structured output generation
- **Diversity**: Generates more creative and varied test cases
- **Worth the Cost**: One-time generation, not repeated per user query

**Cell 16: Helper Function**
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
        with_vectors=False  # Don't need embeddings, only metadata
    )[0]
    return points[0].payload["description"]
```

**Why This Function:**
- **Context Enrichment**: Synthetic data only has product IDs, need full descriptions
- **Efficient Retrieval**: Filtered query is faster than full collection scan
- **Payload Only**: `with_vectors=False` reduces response size (don't need embeddings)

**Cell 20: Dataset Creation with Conflict Handling**
```python
dataset_name = "rag-evaluation-dataset"

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

**Why Try/Except Pattern:**
- **Idempotency**: Notebook can be re-run without manual cleanup
- **409 Conflict**: LangSmith returns error if dataset name exists
- **Fallback**: Read existing dataset instead of failing
- **Developer Experience**: No need to delete dataset before each run

**Common Error Without This:**
```
LangSmithConflictError: 409 Client Error: Conflict
Detail: Dataset with this name already exists.
```

**Cell 21: Dataset Population**
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

**Dataset Structure:**
- **Inputs**: What the RAG pipeline receives (user question)
- **Outputs**: What we expect the RAG pipeline to produce:
  - `ground_truth`: Example of a good answer
  - `reference_context_ids`: Products that SHOULD be retrieved
  - `reference_descriptions`: Full product text for validation

**Why This Structure:**
- **Retrieval Evaluation**: Compare retrieved IDs vs `reference_context_ids`
- **Answer Evaluation**: Compare generated answer vs `ground_truth`
- **Debugging**: See exactly what products influenced the answer
- **Transparency**: Full descriptions available for human review

#### 5. Key Learnings

**Lesson 1: Jupyter Environment Variables**
- **Problem**: `KeyError: 'LANGSMITH_API_KEY'` even though key is in `.env`
- **Root Cause**: Jupyter doesn't auto-load environment files
- **Solution**: Always call `load_dotenv()` at start of notebook
- **Best Practice**: Check for missing keys before API calls

**Lesson 2: Notebook JSON Escaping**
- **Problem**: `SyntaxError: unexpected character after line continuation character`
- **Root Cause**: Double-escaped newlines (`\\n\\n`) in notebook JSON
- **Solution**: Jupyter cell source should use `\n` for newlines, not `\\n\\n`
- **Prevention**: Use NotebookEdit tool or proper JSON manipulation

**Lesson 3: LangSmith Dataset Idempotency**
- **Problem**: 409 Conflict when re-running notebook (dataset exists)
- **Solution**: Try/except with `read_dataset()` fallback
- **Benefit**: Notebook can be safely re-executed

**Lesson 4: Structured LLM Output**
- **JSON Schema** is more reliable than prompt engineering for format
- Eliminates parsing errors and validation logic
- Ensures consistency across all generated examples

#### 6. Evaluation Workflow (Future)

Once the dataset is created, here's how it's used:

**Step 1: Run RAG Pipeline Against Dataset**
```python
for example in dataset:
    question = example.inputs["question"]
    actual_answer = rag_pipeline(question)
    expected_answer = example.outputs["ground_truth"]

    # Compare actual vs expected
    # Measure similarity, retrieval accuracy, etc.
```

**Step 2: Measure Metrics**
- **Retrieval Precision**: % of retrieved products that are in `reference_context_ids`
- **Retrieval Recall**: % of `reference_context_ids` that were actually retrieved
- **Answer Similarity**: Semantic similarity between actual and ground_truth answers
- **Answer Correctness**: Binary score (correct product recommendations or not)

**Step 3: A/B Testing**
- Run pipeline with Prompt A vs Prompt B
- Compare metrics to determine which performs better
- Data-driven decision making vs guessing

**Step 4: Continuous Evaluation**
- Run evaluation suite on every code change (CI/CD integration)
- Track metrics over time (did the last update improve or degrade quality?)
- Alert when metrics drop below threshold

#### 7. Dataset Quality Considerations

**Good Evaluation Questions:**
- Test diverse scenarios (specific products, comparisons, feature-based, price-based)
- Cover edge cases (no results, ambiguous queries, multiple valid answers)
- Represent actual user behavior (real questions users would ask)
- Include varying difficulty (easy exact matches → complex multi-constraint queries)

**Example Dataset Diversity:**
```
- "What are the best wireless headphones?" (Broad search)
- "Headphones under $50 with good bass" (Constraint-based)
- "Compare Sony WH-1000XM4 vs Bose QC45" (Comparison)
- "Gaming headset with detachable mic" (Feature-specific)
- "Kids headphones with volume limiting" (Safety feature)
```

#### 8. Production Enhancements

**Future Improvements:**
1. **Human Review**: Validate LLM-generated questions for realism
2. **Larger Datasets**: Generate 100-500 examples for comprehensive coverage
3. **Automated Evaluation**: CI/CD pipeline runs evaluation on every PR
4. **Metric Dashboards**: Grafana/Prometheus to track evaluation metrics over time
5. **Failure Analysis**: Detailed reports on which questions fail and why

**Integration with Video 5 Observability:**
- Evaluation runs create LangSmith traces (same as production)
- Can debug evaluation failures using trace inspection
- Compare evaluation traces vs production traces to find discrepancies

#### 9. Tools & Technologies

**Required Environment Variables:**
```env
OPENAI_KEY=sk-...                              # For embeddings + generation
LANGSMITH_API_KEY=lsv2_pt_...                  # For dataset storage
LANGSMITH_PROJECT=rag-evaluation               # Project organization
LANGSMITH_TRACING=true                         # Enable tracing (optional)
```

**Python Dependencies:**
- `openai` - Embedding and LLM generation
- `langsmith` - Dataset storage and evaluation framework
- `qdrant-client` - Vector database access for product data
- `python-dotenv` - Environment variable loading

**LangSmith Dashboard:**
- View datasets: https://smith.langchain.com
- Navigate: Projects → rag-evaluation → Datasets → rag-evaluation-dataset
- Features: Add/edit/delete examples via web UI

#### 10. Why This Matters

**Before Evaluation Datasets:**
- Manual testing: "Does this answer look good?"
- Subjective quality assessment
- No way to measure improvement objectively
- Regressions go unnoticed until production

**After Evaluation Datasets:**
- Automated testing: "78% of questions answered correctly"
- Objective quality metrics
- Data-driven decisions on model/prompt changes
- Regressions caught in CI/CD before deployment

**Real Impact:**
- **Development Speed**: Faster iteration with automated feedback
- **Quality Assurance**: Systematic testing catches more bugs
- **Cost Optimization**: Compare expensive vs cheap models objectively
- **Team Confidence**: Data shows improvements, not guesses

### Sprint 0 / Video 7: RAG Evaluation with RAGAS Metrics

This sprint implements comprehensive evaluation of the RAG pipeline using RAGAS (RAG Assessment) metrics to measure retrieval quality, answer accuracy, and system performance.

**Notebook:** `notebooks/week1/05-RAG-Evals.ipynb`

**What Was Done:**

#### 1. Overview: Why Evaluate RAG Systems?

**The Challenge:**
- RAG systems have multiple failure modes: bad retrieval, hallucinated answers, irrelevant responses
- Difficult to know if code changes improve or degrade quality
- Subjective assessment ("this looks good") doesn't scale
- Can't compare different approaches objectively (different prompts, models, retrieval strategies)

**The Solution: RAGAS Metrics:**
- **Systematic Evaluation**: Measure specific aspects of RAG quality (retrieval precision, answer faithfulness, relevance)
- **Objective Scores**: Numeric metrics (0-1 scale) for quantitative comparison
- **Repeatable Testing**: Run same evaluation suite after every code change
- **Data-Driven Decisions**: "Prompt A improved faithfulness by 12%" vs "I think this prompt is better"

#### 2. RAGAS Framework

**What is RAGAS?**
- **RAG Assessment (RAGAS)**: Open-source framework specifically designed for evaluating RAG systems
- Created by Exploding Gradients team
- Provides specialized metrics that understand RAG architecture (retrieval + generation)
- Integrates with LangSmith, LangChain, and other LLM observability tools

**Why RAGAS (vs Generic Metrics)?**
- **RAG-Specific**: Metrics designed for retrieval-augmented systems, not just LLM outputs
- **Component-Level**: Separate metrics for retrieval quality vs generation quality
- **Reference-Based**: Can use ground truth data for accurate evaluation
- **No Manual Labeling**: Uses LLMs to evaluate outputs automatically (LLM-as-a-judge pattern)

#### 3. Implemented Metrics

**a) Faithfulness**
```python
scorer = Faithfulness(llm=ragas_llm)
score = await scorer.single_turn_ascore(sample)
```

**What It Measures:**
- Whether the generated answer is grounded in the retrieved context
- Detects hallucinations (LLM making up information not in context)
- Range: 0 (completely unfaithful) to 1 (perfectly grounded)

**How It Works:**
1. Extract claims from the generated answer
2. Check each claim against retrieved context
3. Score = (verified claims) / (total claims)

**Why It Matters:**
- Prevents LLM from inventing product details
- Ensures recommendations are based on actual product data
- Critical for trustworthy e-commerce applications

**b) Answer Relevancy**
```python
scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
score = await scorer.single_turn_ascore(sample)
```

**What It Measures:**
- How relevant the answer is to the user's question
- Whether the LLM addressed what was actually asked
- Range: 0 (irrelevant) to 1 (perfectly relevant)

**How It Works:**
1. Generate hypothetical questions that the answer could address
2. Compare semantic similarity between original question and hypothetical questions
3. Higher similarity = more relevant answer

**Why It Matters:**
- Catches cases where LLM provides correct but off-topic information
- Example: User asks "wireless headphones", LLM talks about wired headphones
- Ensures answers actually help the user

**c) ID-Based Context Precision**
```python
scorer = IDBasedContextPrecision()
score = await scorer.single_turn_ascore(sample)
```

**What It Measures:**
- How many retrieved products are actually relevant to the question
- Precision = (relevant retrieved items) / (total retrieved items)
- Range: 0 (no relevant items retrieved) to 1 (all retrieved items relevant)

**How It Works:**
1. Compare retrieved product IDs against reference product IDs from evaluation dataset
2. Count matches vs total retrieved
3. Measures pure retrieval quality (independent of LLM generation)

**Why It Matters:**
- Isolates retrieval quality from generation quality
- Fast evaluation (no LLM calls, just ID comparison)
- Directly measures semantic search effectiveness

#### 4. Implementation Details

**RAGAS API Evolution:**

The notebook navigates RAGAS's API changes from older versions to the modern API:

**Modern LLM Initialization:**
```python
from openai import OpenAI
from ragas.llms import llm_factory

openai_client = OpenAI()
ragas_llm = llm_factory("gpt-4o-mini", client=openai_client)
```

**Why This Approach:**
- `llm_factory()` is the modern API (deprecated: `LangchainLLMWrapper`)
- Requires explicit `OpenAI` client instance (text-only mode removed)
- Returns `InstructorLLM` type compatible with all RAGAS metrics

**Embeddings Wrapper Requirement:**
```python
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)
```

**Why LangchainEmbeddingsWrapper:**
- `AnswerRelevancy` metric requires embeddings with `embed_query()` and `embed_documents()` methods
- RAGAS's native `OpenAIEmbeddings` uses different method names (`embed_text`, `embed_texts`)
- LangChain wrapper provides compatible interface

**Evaluation Functions:**

Each metric implemented as async function:

```python
async def ragas_faithfulness(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"]
    )
    scorer = Faithfulness(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)

async def ragas_response_relevancy(run, example):
    sample = SingleTurnSample(
        user_input=run["question"],
        response=run["answer"],
        retrieved_contexts=run["retrieved_context"]
    )
    scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return await scorer.single_turn_ascore(sample)

async def ragas_context_precision_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run["retrieved_context_ids"],
        reference_context_ids=example["reference_context_ids"]
    )
    scorer = IDBasedContextPrecision()
    return await scorer.single_turn_ascore(sample)
```

#### 5. Evaluation Workflow

**Step 1: Load Evaluation Dataset**
```python
from langsmith import Client

client = Client()
dataset = client.read_dataset(dataset_name="rag-evaluation-dataset")
examples = list(client.list_examples(dataset_id=dataset.id, limit=10))
```

**Step 2: Run RAG Pipeline**
```python
reference_input = examples[0].inputs
reference_output = examples[0].outputs

result = rag_pipeline(reference_input["question"], top_k=5)
```

**Step 3: Evaluate with RAGAS Metrics**
```python
faithfulness_score = await ragas_faithfulness(result, reference_output)
relevancy_score = await ragas_response_relevancy(result, reference_output)
precision_score = await ragas_context_precision_id_based(result, reference_output)
```

**Step 4: Interpret Scores**
- **Faithfulness Score**: How well answer is grounded in context
  - Example: 0.71 = 71% of claims in answer are verified by retrieved context
- **Relevancy Score**: How well answer addresses the question
  - Example: 0.0 = Answer completely off-topic (indicates problem with generation)
- **Precision Score**: How many retrieved products are relevant
  - Example: 0.2 = Only 20% of retrieved products match reference set (poor retrieval)

#### 6. Key Learnings

**Lesson 1: RAGAS API Migration**
- **Problem**: `AttributeError: 'Faithfulness' object has no attribute 'single_turn_ascore'`
- **Root Cause**: Importing from deprecated `ragas.metrics.collections`
- **Solution**: Import from `ragas.metrics` directly
- **Modern Pattern**: `from ragas.metrics import Faithfulness, AnswerRelevancy, IDBasedContextPrecision`

**Lesson 2: Embeddings Interface Compatibility**
- **Problem**: `AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'`
- **Root Cause**: RAGAS metrics expect LangChain-style embedding interface
- **Solution**: Use `LangchainEmbeddingsWrapper` around `OpenAIEmbeddings`
- **Why**: Different embedding providers use different method names

**Lesson 3: LLM Factory Requirements**
- **Problem**: `ValueError: llm_factory() requires a client instance`
- **Root Cause**: Modern RAGAS API removed text-only mode
- **Solution**: Explicitly instantiate `OpenAI()` client and pass to `llm_factory()`
- **Benefit**: More control over API configuration (timeouts, retries, etc.)

#### 7. Benefits of Systematic Evaluation

**Before Evaluation:**
- "This answer looks good" (subjective)
- No way to measure improvement
- Regressions go unnoticed
- Can't compare different approaches

**After Evaluation:**
- "Faithfulness improved from 0.65 to 0.78" (objective)
- Track metrics over time
- Catch regressions in CI/CD
- Data-driven decisions on model/prompt changes

**Real-World Use Cases:**
- **A/B Testing**: Compare GPT-4o-mini vs GPT-5-nano (cost vs quality trade-off)
- **Prompt Engineering**: Test different system prompts objectively
- **Retrieval Tuning**: Measure impact of changing top_k parameter
- **Model Selection**: Evaluate different embedding models
- **Regression Detection**: Alert when code changes degrade metrics

#### 8. Integration with LangSmith

**Dataset-Driven Evaluation:**
- Evaluation dataset created in Video 6 (`rag-evaluation-dataset`)
- Contains reference questions and expected product IDs
- RAGAS metrics compare RAG outputs against reference data

**Observability Integration:**
- LangSmith tracing (from Video 5) works during evaluation
- Can inspect traces for failed evaluation cases
- Debug why specific questions scored low

**Complete Evaluation Loop:**
```
LangSmith Dataset (Video 6)
      ↓
RAG Pipeline (Video 4)
      ↓
RAGAS Metrics (Video 7)
      ↓
Scores + Insights
```

#### 9. Production Considerations

**Not Implemented (Future Work):**
- **Batch Evaluation**: Run metrics on entire dataset, not just one example
- **Metric Aggregation**: Calculate mean/median/p95 scores across dataset
- **Automated Reports**: Generate evaluation reports with charts
- **CI/CD Integration**: Run evaluation on every PR, block if scores drop
- **Threshold Alerts**: Alert when metrics fall below acceptable levels
- **Historical Tracking**: Store scores in database, visualize trends over time

**When to Add:**
- Batch evaluation: After validating metrics work on individual examples
- CI/CD integration: When moving to production deployment
- Monitoring dashboards: When tracking system quality over time

#### 10. Next Steps

**Immediate:**
- Evaluate entire dataset (all 43 examples)
- Calculate aggregate metrics (mean faithfulness, mean relevancy, etc.)
- Identify failure patterns (which types of questions score poorly?)

**Advanced:**
- Implement additional RAGAS metrics (ContextRecall, ContextUtilization)
- A/B test different prompts and compare scores
- Experiment with different LLMs (GPT-4o vs GPT-5-nano)
- Test retrieval strategies (top_k=3 vs top_k=10)
- Add human evaluation for qualitative insights

**Tools & Dependencies:**
```bash
# Added in this sprint
uv add ragas>=0.4.3        # RAGAS evaluation framework
uv add langgraph>=1.0.7    # Required dependency for RAGAS
```

**Required Environment Variables:**
```env
OPENAI_API_KEY=sk-...              # For embeddings + LLM evaluation
LANGSMITH_API_KEY=lsv2_pt_...      # For dataset access
LANGSMITH_PROJECT=rag-tracing      # Project organization
```

## API Endpoints

### FastAPI Backend (`http://localhost:8000`)

#### `GET /`
Welcome endpoint

**Response:**
```json
{
  "message": "Welcome to the AI Chat API",
  "status": "running"
}
```

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy"
}
```

#### `POST /chat`
Chat with AI providers

**Request Body:**
```json
{
  "provider": "Groq",
  "model_name": "llama-3.3-70b-versatile",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}
```

**Response:**
```json
{
  "message": "Hello! How can I help you today?"
}
```

**Supported Providers:**
- `OpenAI`: Models like `gpt-4o-mini`, `o1-mini`
- `Groq`: Models like `llama-3.3-70b-versatile`
- `Google`: Models like `gemini-2.0-flash-exp`

## Usage

### Using the Chatbot UI

1. Open http://localhost:8501 in your browser
2. Select a provider (OpenAI, Groq, or Google) from the sidebar
3. Choose a model from the dropdown
4. Type your message and press Enter
5. View the AI's response in the chat interface

### Using the API Directly

**With curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "Groq",
    "model_name": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**With Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "provider": "Groq",
        "model_name": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json())
```

### Running Jupyter Notebooks

```bash
uv run jupyter notebook notebooks/
```

Or with the activated virtual environment:
```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

### Cleaning Notebook Outputs

Before committing notebooks, clean their outputs:
```bash
make clean-notebook-outputs
```

## Development

### Workspace Structure

This project uses a `uv` workspace with multiple packages:

- **Root workspace**: Common dependencies and workspace configuration
- **apps/api**: FastAPI backend service
- **apps/chatbot_ui**: Streamlit frontend service

### Adding Dependencies

**To root workspace:**
```bash
uv add <package-name>
```

**To specific app:**
```bash
uv add --package api <package-name>
uv add --package chatbot-ui <package-name>
```

### Local Development (without Docker)

**Run API:**
```bash
cd apps/api
uv run uvicorn api.app:app --reload --port 8000
```

**Run Chatbot UI:**
```bash
cd apps/chatbot_ui
uv run streamlit run src/chatbot_ui/app.py
```

### Docker Commands

**Build services:**
```bash
docker compose build
```

**Run in detached mode:**
```bash
docker compose up -d
```

**View logs:**
```bash
docker compose logs -f
```

**Stop services:**
```bash
docker compose down
```

**Rebuild and restart:**
```bash
docker compose up --build --force-recreate
```

## Dependencies

### Root Workspace
- `openai>=2.15.0` - OpenAI API client
- `google-genai>=1.57.0` - Google Generative AI client
- `groq>=1.0.0` - Groq API client
- `streamlit>=1.52.2` - Streamlit web framework
- `pydantic>=2.12.5` - Data validation
- `jupyter>=1.1.1` - Jupyter notebook support
- `python-dotenv>=1.2.1` - Environment variable management
- `qdrant-client>=1.12.1` - Qdrant vector database client
- `pandas>=2.2.0` - Data manipulation and analysis

### API Service
- `fastapi>=0.128.0` - FastAPI framework
- `uvicorn>=0.40.0` - ASGI server

### Chatbot UI Service
- `streamlit>=1.52.2` - Streamlit framework
- `requests>=2.32.0` - HTTP client

## Makefile Commands

```bash
make run-docker-compose       # Sync dependencies and run Docker Compose
make clean-notebook-outputs   # Clear Jupyter notebook outputs
```

## Data Management

### Included Datasets
The repository includes processed, analysis-ready datasets in the `data/` directory:
- Final filtered product metadata (17K items with 100+ ratings)
- Sampled subset for focused analysis (1,000 items)
- Corresponding review records for sampled products

### Downloading Raw Data
Raw datasets are not included in the repository due to size (~26GB total). To obtain them:

1. Visit [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html)
2. Download Electronics category files:
   - `Electronics.jsonl.gz` (~21GB uncompressed)
   - `meta_Electronics.jsonl.gz` (~5GB uncompressed)
3. Extract to the `data/` directory:
   ```bash
   gunzip data/Electronics.jsonl.gz
   gunzip data/meta_Electronics.jsonl.gz
   ```
4. Run `notebooks/week1/01-explore-amazon-dataset.ipynb` to regenerate intermediate files

### Dataset Citation
```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

## Security Notes

### Environment Variables
- The `.env` file is gitignored to prevent accidental exposure of API keys
- Use `env.example` as a template
- Never commit real API keys to version control

### API Key Rotation
- Rotate your API keys immediately if they are exposed
- Monitor your API usage for unusual activity
- Use different keys for development and production

### Docker Security
- Services run as non-root users
- Environment variables are passed securely via `.env` file
- No secrets are baked into Docker images

## Troubleshooting

### Port Already in Use
If ports 8000 or 8501 are already in use:
```bash
# Find process using the port
lsof -i :8000
lsof -i :8501

# Kill the process or change ports in docker-compose.yml
```

### API Connection Errors
- Ensure the API service is running: `docker compose ps`
- Check API logs: `docker compose logs api`
- Verify `API_URL` in `.env` is set to `http://api:8000`

### Missing Dependencies
```bash
# Reinstall all dependencies
uv sync --reinstall
```

### Docker Build Issues
```bash
# Clean Docker cache and rebuild
docker compose down
docker system prune -f
docker compose up --build
```

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Clean notebook outputs: `make clean-notebook-outputs`
4. Commit your changes
5. Push and create a pull request

## License

This project is for educational purposes as part of the AI Engineering Bootcamp.

## Support

For questions or issues, please open an issue in the repository or contact the bootcamp instructors.
