# Week 1: RAG System Implementation

Jupyter notebooks covering the complete RAG (Retrieval-Augmented Generation) pipeline - from data preprocessing and embeddings to production deployment and evaluation.

## Overview

Week 1 builds a production-ready RAG system for product Q&A using Amazon Electronics data. You'll learn the complete workflow: data exploration, embedding generation, vector storage, retrieval-augmented generation, observability, evaluation dataset creation, and quality measurement.

## Notebooks

### 01-explore-amazon-dataset.ipynb

**Purpose**: Data exploration and filtering of Amazon Electronics reviews dataset

**What You'll Learn**:
- Loading large JSONL datasets with pandas
- Filtering by date range (2022-2023)
- Handling missing data (categories, dates)
- Statistical analysis (rating thresholds)
- Reproducible sampling (seed=42)

**Key Outputs**:
- Understanding of dataset structure (2.5M+ records)
- Filtered dataset (17,162 products with 100+ ratings)
- Sampled subset (1,000 products) for development

**Data Pipeline**:
```
Raw Data (2.5M products)
    ↓ Filter: 2022-2023 only
    ↓ Filter: Has valid category
    ↓ Filter: 100+ ratings (quality threshold)
17,162 products
    ↓ Sample: 1,000 products (seed=42, reproducible)
Final Dataset: data/meta_Electronics_2022_2023_with_category_ratings_over_100_sample_1000.jsonl
```

**Why 100+ Ratings**:
- Quality signal (products with real user feedback)
- Reduces noise from new/obscure products
- Still maintains diversity (17k+ products)

**Why 1,000 Sample**:
- Fast development iteration
- Lower embedding costs (~$0.03 vs $0.50)
- Representative subset for testing

---

### 02-RAG-preprocessing-Amazon.ipynb

**Purpose**: Generate embeddings and populate Qdrant vector database

**What You'll Learn**:
- Creating embeddings with OpenAI API
- Setting up Qdrant vector database
- Creating collections with proper schema
- Batch upserting vectors efficiently
- Verifying collection with test queries

**Key Steps**:
1. **Load Dataset**: 1,000-item sample from notebook 01
2. **Combine Fields**: `title + features` → rich product descriptions
3. **Generate Embeddings**: OpenAI `text-embedding-3-small` (1536 dims)
4. **Create Qdrant Collection**:
   - Name: `"Amazon-items-collection-00"`
   - Vector size: 1536
   - Distance: Cosine similarity
5. **Upsert Points**: Batch insert with product metadata
6. **Verify**: Test semantic search

**Embedding Model**:
- **Model**: `text-embedding-3-small`
- **Dimensions**: 1536
- **Cost**: ~$0.00002 per 1K tokens
- **Total Cost**: ~$0.03 for 1,000 products

**Why This Model**:
- OpenAI's smaller, faster embedding model
- Good balance of quality and cost
- 1536 dims sufficient for product similarity

**Qdrant Collection Schema**:
```python
{
    "vectors": 1536-dim embedding,
    "payload": {
        "parent_asin": str,        # Product ID
        "title": str,              # Product name
        "description": str,        # Combined title + features
        "average_rating": float,   # User rating
        "features": list[str],     # Bullet points
        # ... other metadata
    }
}
```

**Distance Metric**:
- **Cosine Similarity**: Measures angle between vectors (0-1)
- 1.0 = identical, 0.0 = orthogonal
- Why cosine: Works well for text (normalized by length)

---

### 03-RAG-pipeline.ipynb

**Purpose**: Implement and test the complete RAG workflow

**What You'll Learn**:
- Converting queries to embeddings
- Querying Qdrant for similar products
- Formatting context for LLM prompts
- Prompt engineering for grounded answers
- Generating responses with OpenAI

**5-Step RAG Workflow**:
1. **Get Embedding**: Query → 1536-dim vector
2. **Retrieve Data**: Vector → top-k similar products (k=5)
3. **Process Context**: Products → formatted text
4. **Build Prompt**: System instructions + context + question
5. **Generate Answer**: Prompt → natural language response

**Prompt Engineering**:
```python
system_message = """You are a helpful shopping assistant.
Answer the user's question based ONLY on the retrieved product context.
If the context doesn't contain relevant information, say so clearly.

Retrieved Product Context:
{formatted_context}
"""
```

**Why This Prompt Structure**:
- Clear role: "Shopping assistant"
- Grounding constraint: "ONLY on the retrieved product context"
- Honesty instruction: "Say so clearly" if insufficient context
- Context placement: In system message for stronger adherence

**Testing**:
- Example queries with expected results
- Similarity score analysis
- Answer quality validation

---

### 04-evaluation-dataset.ipynb

**Purpose**: Generate synthetic evaluation dataset using LLM

**What You'll Learn**:
- Creating test cases with structured LLM output (JSON schema)
- Using GPT-4o for question generation
- Building LangSmith datasets
- Fetching reference data from Qdrant
- Idempotent dataset creation (handles "already exists")

**Synthetic Data Generation**:
```python
# Define JSON schema for structured output
output_schema = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "chunk_ids": {"type": "array", "items": {"type": "string"}},
        "answer_example": {"type": "string"}
    }
}

# Generate with GPT-4o
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": generation_prompt}],
    response_format={"type": "json_schema", "json_schema": output_schema}
)
```

**LangSmith Dataset Structure**:
```python
{
    "inputs": {"question": str},
    "outputs": {
        "ground_truth": str,              # Expected answer
        "reference_context_ids": list,    # Product IDs that should be retrieved
        "reference_descriptions": list    # Full product descriptions
    }
}
```

**Why Synthetic Data**:
- Fast generation (43 questions in minutes)
- Covers diverse question types (specific, comparison, constraint-based)
- Includes reference answers for quality comparison
- Reference IDs enable retrieval accuracy measurement

**Dataset Statistics**:
- **Total Examples**: 43 questions
- **Question Types**: Specific queries, comparisons, feature-based, constraint-based
- **Reference Products**: 2-5 products per question
- **Use Case**: Systematic RAG pipeline testing

---

### 05-RAG-Evals.ipynb

**Purpose**: Run RAGAS evaluation metrics against RAG pipeline

**What You'll Learn**:
- Setting up RAGAS evaluation framework
- Implementing 4 evaluation metrics
- Running evaluations via LangSmith
- Interpreting evaluation results
- Identifying quality bottlenecks

**4 Evaluation Metrics**:

1. **Faithfulness** (0-1):
   - Question: "Is the answer grounded in retrieved context?"
   - Method: LLM judge verifies claims against context
   - Good Score: >0.8 (answers are trustworthy)

2. **Response Relevancy** (0-1):
   - Question: "Does the answer address the question?"
   - Method: Generate hypothetical questions, compare via embeddings
   - Good Score: >0.7 (answers are on-topic)

3. **Context Precision** (0-1):
   - Question: "What % of retrieved products are relevant?"
   - Formula: `|retrieved ∩ reference| / |retrieved|`
   - Good Score: >0.6 (more than half relevant)

4. **Context Recall** (0-1):
   - Question: "What % of relevant products were retrieved?"
   - Formula: `|retrieved ∩ reference| / |reference|`
   - Good Score: >0.5 (found at least half)

**Running Evaluations**:
```python
results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"]),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based
    ],
    experiment_prefix="retriever",
    max_concurrency=5
)
```

**LangSmith Results**:
- Aggregate scores per metric
- Per-example scores with trace links
- Score distribution histograms
- Comparison across multiple runs

**Interpreting Results**:
- **Low Faithfulness**: Prompt issue (LLM not grounded)
- **Low Relevancy**: Prompt or LLM quality issue
- **Low Precision**: Too much noise (increase threshold, reduce top_k)
- **Low Recall**: Missing relevant items (increase top_k, lower threshold)

## Running the Notebooks

### From Project Root
```bash
# Launch Jupyter
uv run jupyter notebook notebooks/week1/

# or specific notebook
uv run jupyter notebook notebooks/week1/01-explore-amazon-dataset.ipynb
```

### Prerequisites

1. **Environment Variables** (`.env` file):
   ```env
   OPENAI_KEY=sk-...
   LANGSMITH_API_KEY=...
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_PROJECT=rag-tracing
   LANGSMITH_TRACING=true
   ```

2. **Qdrant Running**:
   ```bash
   # Docker Compose (recommended)
   docker compose up qdrant

   # or manually
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Dependencies Installed**:
   ```bash
   uv sync
   ```

### Notebook Execution Order

**Must run in order**:
1. **01-explore-amazon-dataset.ipynb** - Creates filtered dataset
2. **02-RAG-preprocessing-Amazon.ipynb** - Populates Qdrant
3. **03-RAG-pipeline.ipynb** - Tests RAG workflow
4. **04-evaluation-dataset.ipynb** - Creates test dataset
5. **05-RAG-Evals.ipynb** - Runs evaluations

**Why Order Matters**:
- Later notebooks depend on outputs from earlier ones
- Qdrant must be populated before running RAG pipeline
- Evaluation dataset must exist before running evals

## Data Files

Created by notebooks (in `data/` directory):

```
data/
├── meta_Electronics_2022_2023_with_category.jsonl          # Notebook 01 intermediate
├── meta_Electronics_2022_2023_with_category_ratings_over_100.jsonl  # Notebook 01 filtered
└── meta_Electronics_2022_2023_with_category_ratings_over_100_sample_1000.jsonl  # Notebook 01 final
```

**Note**: Some intermediate files excluded from git (see `.gitignore`).

## Qdrant Collections

Created by notebook 02:

**Collection Name**: `Amazon-items-collection-00`
- **Vectors**: 1,000 × 1536 dimensions
- **Distance**: Cosine similarity
- **Payload**: Product metadata (ASIN, title, description, rating)
- **Storage**: `qdrant_storage/` (gitignored, persists between runs)

**Accessing Qdrant**:
- **Dashboard**: http://localhost:6333/dashboard
- **API**: http://localhost:6333

**Resetting Collection**:
```bash
# Stop Qdrant
docker compose down

# Delete storage
rm -rf qdrant_storage/

# Restart and re-run notebook 02
docker compose up qdrant
```

## Cost Analysis

**Embeddings** (Notebook 02):
- 1,000 products × avg 100 tokens = 100K tokens
- Cost: ~$0.002 per 1K tokens = **~$0.20**

**Evaluation** (Notebook 05):
- 43 examples × 4 metrics
- Faithfulness & Relevancy require LLM judge
- Cost: ~$0.25 per full evaluation run (see [../../apps/api/evals/README.md](../../apps/api/evals/README.md))

**Total for Week 1**: ~$0.50-1.00 (depending on experimentation)

## Key Learnings

### 1. Embedding Consistency
- **Critical**: Use same model for preprocessing AND runtime queries
- Different models = different vector spaces = poor retrieval
- Always verify model name matches

### 2. Prompt Engineering Matters
- Grounding constraint prevents hallucination
- System message more powerful than user message for instructions
- Clear role and honesty instructions improve quality

### 3. Evaluation is Essential
- Can't improve what you don't measure
- Objective metrics reveal bottlenecks
- Synthetic datasets enable continuous testing

### 4. Precision vs Recall Trade-off
- High top_k → Better recall, worse precision
- Low top_k → Better precision, worse recall
- Optimal balance depends on use case

### 5. Qdrant Best Practices
- Use batching for large upserts
- Wait for indexing (`wait=True`)
- Cosine similarity for text embeddings
- Test queries to verify collection

## Notebook Hygiene

**CRITICAL**: Always clean outputs before committing:
```bash
# From project root
make clean-notebook-outputs

# or manually
uv run jupyter nbconvert --clear-output --inplace notebooks/week1/*.ipynb
```

**Why**:
- Outputs contain API responses, potentially sensitive data
- Large embedded images/data bloat git history
- Clean notebooks easier to diff and review

## Troubleshooting

**Qdrant Connection Errors**:
- Verify Qdrant running: `curl http://localhost:6333/collections`
- Check port 6333 not in use: `lsof -i :6333`
- In notebooks: Use `http://localhost:6333` (not `http://qdrant:6333`)

**OpenAI API Errors**:
- Check `OPENAI_KEY` in `.env`
- Verify API key has credits
- Rate limits: Add delays between requests

**LangSmith Not Working**:
- Verify all 4 `LANGSMITH_*` variables in `.env`
- Check API key validity
- Ensure `load_dotenv()` called in notebook

**Missing Dependencies**:
- Run `uv sync` from project root
- Restart Jupyter kernel after installing

**Notebook Kernel Crashes**:
- Large datasets may exceed memory
- Reduce sample size (e.g., 500 instead of 1000)
- Restart kernel and re-run cells

## Related Documentation

- **Week 0**: [../week0/README.md](../week0/README.md) - LLM API fundamentals
- **API Implementation**: [../../apps/api/src/api/agents/README.md](../../apps/api/src/api/agents/README.md) - Production RAG pipeline
- **Evaluation**: [../../apps/api/evals/README.md](../../apps/api/evals/README.md) - Automated evaluation system
- **Project Root**: [../../README.md](../../README.md) - Overall architecture

## External Resources

- **RAG Paper**: https://arxiv.org/abs/2005.11401
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **RAGAS Docs**: https://docs.ragas.io
- **LangSmith Docs**: https://docs.smith.langchain.com
- **Amazon Reviews Dataset**: https://amazon-reviews-2023.github.io/
