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
│       └── 02-RAG-preprocessing-Amazon.ipynb # RAG preprocessing & embeddings
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
