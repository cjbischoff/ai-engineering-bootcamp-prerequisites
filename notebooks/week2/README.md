# Week 2: Advanced RAG - Structured Outputs & Hybrid Retrieval

This folder contains Jupyter notebooks for Week 2 of the AI Engineering Bootcamp, focusing on advanced RAG techniques including structured outputs, grounding, hybrid search, and reranking.

## 📚 Notebook Overview

### Prerequisites

Before starting these notebooks, ensure you have:
- Qdrant running locally (`docker compose up -d` from project root)
- Required API keys in `.env` file:
  - `OPENAI_API_KEY` - For embeddings and LLM generation
  - `COHERE_API_KEY` - For reranking (Video 6 only)
- Product data indexed in Qdrant (run data preprocessing scripts from Week 1)

### Learning Path

The notebooks build on each other progressively. **Follow in order**:

```
Video 1: Structured Outputs Intro
    ↓
Video 2: Grounding with References
    ↓
Video 5: Hybrid Search (Dense + Sparse)
    ↓
Video 6: Reranking with Cross-Encoders
```

---

## 📓 Video 1: Structured Outputs with Instructor

**Notebook:** `01-Strucutured-Ouputs-Intro.ipynb`

### What You'll Learn

- Using the **instructor** library to enforce structured LLM outputs
- Defining Pydantic models for type-safe responses
- Patching OpenAI client with `instructor.from_openai()`
- Using `create_with_completion()` for structured outputs
- Integrating structured outputs into RAG pipeline

### Key Concepts

**Before (Week 1):**
```python
# Returns raw string
response = openai.chat.completions.create(...)
answer = response.choices[0].message.content  # String (no validation)
```

**After (Week 2):**
```python
# Returns validated Pydantic model
class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")

instructor_client = instructor.from_openai(openai.OpenAI())
response, raw = instructor_client.chat.completions.create_with_completion(
    response_model=RAGGenerationResponse,
    ...
)
answer = response.answer  # Type-safe access
```

### Why This Matters

- **Type Safety**: Guaranteed data structure for downstream processing
- **Validation**: Automatic validation against Pydantic schema
- **Reliability**: Reduces parsing errors and inconsistent formats
- **Integration**: Easy integration with existing Python code

### Code Structure

1. **Part 1**: Baseline OpenAI response (raw text)
2. **Part 2**: Add instructor for structured outputs
3. **Part 3**: Integrate with RAG pipeline from Week 1

### Dependencies

```python
openai
instructor
pydantic
qdrant_client
```

---

## 📓 Video 2: Grounding with References

**Notebook:** `02-Structures-Outputs-RAG-Pipeline.ipynb`

### What You'll Learn

- **Grounding**: Linking generated answers to source documents
- Nested Pydantic models for complex structured outputs
- Prompt engineering for grounded responses
- Distinguishing retrieved vs. used context

### Key Concepts

**Nested Pydantic Models:**

```python
class RAGUsedContext(BaseModel):
    id: str = Field(description="Product ASIN")
    description: str = Field(description="Short product description")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="Products used in answer")
```

**Why Grounding Matters:**
- **Transparency**: Users can see which products influenced the answer
- **Verification**: Users can validate the AI's reasoning
- **Trust**: Providing sources increases confidence
- **Debugging**: Developers can trace why certain products were chosen

### Enhanced Prompt Engineering

The prompt explicitly instructs the LLM to:
1. Answer the question based on context
2. Identify which products were **actually used** (not all retrieved)
3. Provide product IDs for those products
4. Give concise descriptions with product names

### Code Structure

1. **Part 1**: Baseline RAG pipeline (from Video 1)
2. **Part 2**: Enhanced RAG with grounding context
3. **Analysis**: Comparing retrieved vs. used products

### Key Insight

**Retrieved Context ≠ Used Context**

- **Retrieved**: All products fetched from vector database (e.g., 10 products)
- **Used/References**: Only products the LLM actually used in the answer (e.g., 2-3 products)
- **LLM performs filtering**: Selects relevant products based on specific question

### Dependencies

```python
openai
instructor
pydantic
qdrant_client
```

---

## 📓 Video 5: Hybrid Search

**Notebook:** `03-Hybrid-Search.ipynb`

### What You'll Learn

- **Hybrid search**: Combining dense (semantic) and sparse (keyword/BM25) retrieval
- Qdrant dual vector architecture (named vectors)
- **Prefetch mechanism** for multi-stage retrieval
- **RRF (Reciprocal Rank Fusion)** for merging ranked lists
- BM25 sparse vectors with Qdrant's built-in support

### Key Concepts

**Dual Vector Architecture:**

```python
# Collection with TWO vector types
qdrant_client.create_collection(
    collection_name="Amazon-items-collection-01-hybrid-search",
    vectors_config={
        "text-embedding-3-small": VectorParams(size=1536, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "bm25": SparseVectorParams(modifier=models.Modifier.IDF)
    }
)
```

**Point Structure with Dual Vectors:**

```python
PointStruct(
    id=i,
    vector={
        "text-embedding-3-small": embedding,  # Dense: 1536 floats
        "bm25": Document(text=description, model="qdrant/bm25")  # Sparse: auto BM25
    },
    payload=data
)
```

**Hybrid Retrieval with Prefetch + RRF:**

```python
results = qdrant_client.query_points(
    collection_name="Amazon-items-collection-01-hybrid-search",
    prefetch=[
        Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
        Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20)
    ],
    query=FusionQuery(fusion="rrf"),
    limit=k
)
```

### Why Hybrid Search?

**Dense (Semantic) Search:**
- ✅ Understands synonyms ("waterproof" = "water-resistant")
- ✅ Captures context and meaning
- ❌ May miss exact keyword matches
- ❌ Struggles with rare terms and product codes

**Sparse (BM25) Search:**
- ✅ Excellent for exact matches (product codes, model numbers)
- ✅ Good for acronyms and technical terms
- ❌ Doesn't understand synonyms or context
- ❌ Keyword-only matching

**Hybrid (Dense + Sparse):**
- ✅ Combines strengths of both approaches
- ✅ More robust across diverse query types
- ✅ ~20% higher recall than dense-only
- Minimal latency increase (~15ms)

### RRF (Reciprocal Rank Fusion)

**Why RRF?**
- Dense scores (~0.85) and sparse scores (~127.3) can't be directly combined
- Rank-based approach avoids normalization problems
- Products ranked highly in **both** methods score best
- Formula: `RRF_score = Σ (1 / (k + rank_i))` where k=60

### Performance Characteristics

**Memory per Product:**
- Dense vector: 1536 floats × 4 bytes = 6KB
- Sparse vector: ~100 terms × 8 bytes = 800 bytes
- Payload: ~500 bytes
- **Total: ~7.4 KB per product**

**Query Performance:**
- Dense search: O(log N) with HNSW index
- Sparse search: O(T × log N) where T = query terms
- Fusion: O(K1 + K2) where K = prefetch limits
- **Total latency: ~115ms** (OpenAI embedding is bottleneck)

### Code Structure

1. **Setup**: Create Qdrant collection with dual vectors
2. **Indexing**: Batch embed and upload 1000 products
3. **Retrieval**: Implement hybrid search with prefetch + RRF
4. **Testing**: Compare hybrid vs dense-only results

### Dependencies

```python
openai
qdrant_client
pandas
fastembed  # Optional alternative to OpenAI embeddings
```

---

## 📓 Video 6: Reranking with Cross-Encoders

**Notebook:** `04-Reranking.ipynb`

### What You'll Learn

- **Two-stage retrieval**: Hybrid search + reranking
- **Bi-encoder vs cross-encoder** architectures
- Cohere Rerank API integration
- Performance and cost trade-offs
- When to use reranking vs when to skip it

### Key Concepts

**Two-Stage Retrieval Pipeline:**

```
User Query
    ↓
Stage 1: Hybrid Search (Bi-Encoder)
  - Dense + sparse vectors with RRF fusion
  - Fast initial retrieval (~115ms)
  - Top 20 candidates
  - Good recall (~90%), moderate precision (~70%)
    ↓
Stage 2: Reranking (Cross-Encoder)
  - Cohere rerank-v4.0-pro
  - Slower but more accurate (~500ms)
  - Top 5-20 reordered results
  - Excellent precision (~95%)
    ↓
Final Results (Highly Relevant)
```

**Bi-Encoder vs Cross-Encoder:**

| Aspect | Bi-Encoder (Retrieval) | Cross-Encoder (Reranking) |
|--------|------------------------|---------------------------|
| **Architecture** | Query and docs encoded separately | Query + doc encoded together |
| **Speed** | Fast (~100ms for 1M docs) | Slow (~25ms per doc) |
| **Accuracy** | Good (70-80% precision) | Excellent (90-95% precision) |
| **Scalability** | Millions of docs | Hundreds of docs max |
| **Pre-compute** | Yes (doc embeddings) | No (query-dependent) |
| **Use Case** | Initial retrieval | Final refinement |

**Cohere Rerank Integration:**

```python
import cohere
cohere_client = cohere.ClientV2()

# Stage 1: Hybrid search (k=20 for reranking)
results = retrieve_data(query, qdrant_client, k=20)
to_rerank = results["retrieved_context"]

# Stage 2: Rerank
response = cohere_client.rerank(
    model="rerank-v4.0-pro",
    query=query,
    documents=to_rerank,
    top_n=20
)

# Reconstruct in new order
reranked_results = [to_rerank[result.index] for result in response.results]
```

### Performance and Cost Analysis

**Latency:**

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Query embedding | ~100ms | 100ms |
| Dense prefetch | <10ms | 110ms |
| Sparse prefetch | <5ms | 115ms |
| RRF fusion | <1ms | 116ms |
| **Reranking (20 docs)** | **~500ms** | **~616ms** |

**Cost (1000 queries/day, 30 days):**

| Component | Cost per Query | Monthly Cost |
|-----------|---------------|--------------|
| OpenAI embeddings | $0.0002 | $6 |
| Cohere reranking | $0.002 | $60 |
| **Total** | **$0.0022** | **$66** |

**Key Insight:** Reranking dominates both latency (500ms of 616ms) and cost ($60 of $66)

### When to Use Reranking

**✅ Use Reranking When:**
- Precision is critical (customer support, legal, medical)
- Small final result set (top 5-10)
- Have budget for API costs ($2 per 1K queries)
- Latency budget allows ~500ms overhead
- Quality improvements justify 10x cost increase

**❌ Skip Reranking When:**
- Need sub-200ms response times
- Large result sets (50+ results)
- Cost-sensitive application (<$0.50 per 1K queries)
- Hybrid search already provides sufficient precision
- High volume use case (millions of queries/day)

### Comparison of Approaches

| Approach | Latency | Cost/1K Queries | Precision | Recall | Best For |
|----------|---------|-----------------|-----------|--------|----------|
| **Dense only** | 50ms | $0.20 | 60% | 70% | High volume, cost-sensitive |
| **Hybrid** | 115ms | $0.20 | 70% | 90% | General purpose, balanced |
| **Hybrid + Rerank** | 616ms | $2.20 | 95% | 90% | High precision, low volume |

**Quality Improvement:**
- Dense-only → Hybrid: +10% precision, +20% recall
- Hybrid → Hybrid+Rerank: +25% precision, same recall
- Dense-only → Hybrid+Rerank: +35% precision, +20% recall

### Code Structure

1. **Stage 1**: Hybrid search retrieval (from Video 5)
2. **Stage 2**: Cohere reranking implementation
3. **Analysis**: Performance metrics and cost-benefit analysis
4. **Key Takeaways**: Production considerations

### Dependencies

```python
openai
qdrant_client
cohere  # NEW: For reranking
pandas
```

---

## 🎯 Learning Outcomes

After completing all Week 2 notebooks, you will be able to:

### Structured Outputs & Grounding
- ✅ Enforce type-safe LLM outputs with instructor + Pydantic
- ✅ Create nested Pydantic models for complex structures
- ✅ Implement grounding to link answers to source documents
- ✅ Distinguish between retrieved and used context

### Hybrid Search
- ✅ Configure Qdrant collections with dual vectors (dense + sparse)
- ✅ Understand bi-encoder embeddings for semantic search
- ✅ Use BM25 sparse vectors for keyword matching
- ✅ Implement prefetch mechanism for multi-stage retrieval
- ✅ Apply RRF (Reciprocal Rank Fusion) to merge ranked lists
- ✅ Make informed decisions about when to use hybrid vs dense-only

### Reranking
- ✅ Understand difference between bi-encoders and cross-encoders
- ✅ Implement two-stage retrieval (hybrid search + reranking)
- ✅ Integrate Cohere Rerank API into RAG pipeline
- ✅ Analyze latency and cost trade-offs
- ✅ Make data-driven decisions about when to use reranking

### Production Skills
- ✅ Batch process embeddings efficiently
- ✅ Optimize retrieval pipelines for performance
- ✅ Balance quality, latency, and cost
- ✅ Monitor and evaluate RAG system quality
- ✅ Debug and troubleshoot vector search issues

---

## 📊 Quick Reference: Approach Comparison

### When to Use Each Approach

**Dense-Only Search (Week 1):**
- Simple queries with natural language
- All semantic understanding, no product codes
- Ultra-low latency requirements (<100ms)
- Cost-sensitive (embeddings only: $0.20/1K queries)

**Hybrid Search (Video 5):**
- Diverse query types (keywords + descriptions)
- Product codes, model numbers, technical terms
- Need both semantic and exact matching
- Same cost as dense-only, 15ms extra latency

**Hybrid + Reranking (Video 6):**
- Precision is critical (support, legal, medical)
- Small result sets (top 5-10)
- Budget allows $2/1K queries
- Latency budget ~600ms
- Quality justifies 10x cost

### Performance Comparison Matrix

| Metric | Dense-Only | Hybrid | Hybrid+Rerank |
|--------|-----------|--------|---------------|
| **Latency** | 100ms | 115ms | 616ms |
| **Cost (1K queries)** | $0.20 | $0.20 | $2.20 |
| **Precision** | 60% | 70% | 95% |
| **Recall** | 70% | 90% | 90% |
| **Scalability** | 1M+ products | 1M+ products | Top-K only |
| **Complexity** | Low | Medium | High |

### Use Case Decision Tree

```
Start
  ↓
Do you need product code matching?
  ├─ Yes → Use Hybrid (Videos 5-6)
  └─ No → Continue
       ↓
Is latency critical (<100ms)?
  ├─ Yes → Use Dense-Only (Week 1)
  └─ No → Continue
       ↓
Is precision critical (>90%)?
  ├─ Yes → Use Hybrid+Rerank (Video 6)
  └─ No → Use Hybrid (Video 5)
```

---

## 🔧 Common Issues and Solutions

### Issue: ModuleNotFoundError for instructor

**Solution:**
```bash
pip install instructor
# or
uv add instructor
```

### Issue: Qdrant connection refused

**Solution:**
```bash
# Start Qdrant in Docker
docker compose up -d

# Verify it's running
docker compose ps

# Check logs
docker compose logs qdrant
```

### Issue: Missing COHERE_API_KEY for Video 6

**Solution:**
1. Get API key at https://dashboard.cohere.com/api-keys
2. Add to `.env`:
   ```
   COHERE_API_KEY=your_api_key_here
   ```
3. Restart notebook kernel

### Issue: Collection not found

**Solution:**
- Week 1 notebooks use: `Amazon-items-collection-00`
- Video 5-6 use: `Amazon-items-collection-01-hybrid-search`
- Run the data preprocessing cells in each notebook to create collections

### Issue: OpenAI rate limit errors

**Solution:**
```python
# Reduce batch size in get_embeddings_batch()
embeddings = get_embeddings_batch(text_to_embed, batch_size=50)  # Default: 100
```

### Issue: Notebook cells take too long

**Solution:**
- Reduce dataset size (first 100 rows for testing):
  ```python
  df_items = df_items.head(100)
  ```
- Use smaller `k` values for retrieval:
  ```python
  results = retrieve_data(query, k=5)  # Instead of k=20
  ```

---

## 🚀 Next Steps

### After completing Week 2 notebooks:

1. **Integrate with FastAPI Backend**
   - Add reranking to RAG endpoint (optional flag)
   - Implement structured outputs in production API
   - Add grounding context to API responses

2. **Evaluate Quality with RAGAS**
   - Run evaluation scripts in `scripts/` folder
   - Compare dense-only vs hybrid vs hybrid+rerank
   - A/B test different approaches

3. **Optimize for Production**
   - Implement caching for repeated queries
   - Add monitoring for latency and costs
   - Set up alerting for quality degradation
   - Consider self-hosted reranker for cost savings

4. **Continue to Week 3** (Coming Soon)
   - Prompt engineering and configuration
   - Prompt registries and versioning
   - Multi-modal RAG (text + images)
   - Advanced evaluation metrics

---

## 📚 Additional Resources

### Documentation
- **Instructor Library**: https://python.useinstructor.com/
- **Pydantic**: https://docs.pydantic.dev/
- **Qdrant Hybrid Search**: https://qdrant.tech/documentation/concepts/search/#hybrid-search
- **Cohere Rerank**: https://docs.cohere.com/docs/reranking

### Research Papers
- **Instructor**: "Instructor: Low-Code LLM Programming with Structured Outputs"
- **RRF**: "Reciprocal Rank Fusion for Search Result Merging" (Cormack et al.)
- **BM25**: "Okapi at TREC-3" (Robertson et al., 1994)
- **Cross-Encoders**: "Sentence-BERT for Efficient Dense Retrieval" (Reimers & Gurevych, 2019)

### Community
- **Qdrant Discord**: https://discord.gg/qdrant
- **Instructor GitHub**: https://github.com/jxnl/instructor
- **Cohere Discord**: https://discord.gg/co-mmunity

---

## 💡 Tips for Success

1. **Run Notebooks in Order**: Each notebook builds on concepts from previous ones
2. **Start Qdrant First**: Always ensure Qdrant is running before starting notebooks
3. **Check API Keys**: Verify `.env` file has required keys before running cells
4. **Use Small Datasets**: Test with 100 products before scaling to 1000
5. **Read Comments**: All code cells have detailed explanatory comments
6. **Experiment**: Try different parameters (k, batch_size, top_n) to understand impact
7. **Monitor Costs**: OpenAI embeddings are cheap, but Cohere reranking adds up
8. **Save Checkpoints**: Save Qdrant collections to avoid re-embedding
9. **Clear Outputs**: Run `make clean-notebook-outputs` before committing
10. **Ask Questions**: Use the community resources if you get stuck

---

**Last Updated:** 2026-01-26
**Next Review:** After Week 3 content is added
