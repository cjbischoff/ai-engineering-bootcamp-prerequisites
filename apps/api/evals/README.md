# RAG Evaluation System

Automated evaluation system for measuring RAG pipeline quality using RAGAS (RAG Assessment) metrics. This directory contains scripts that systematically test retrieval accuracy and generation quality.

## Overview

This evaluation system provides objective, quantifiable metrics for RAG performance by running the pipeline against a test dataset and measuring 4 key dimensions:
1. **Retrieval Quality**: Are we finding the right products?
2. **Generation Faithfulness**: Is the LLM grounded in context?
3. **Answer Relevancy**: Does the answer address the question?
4. **End-to-End Performance**: How well does the full pipeline work?

## Why Evaluation Matters

**Without Evaluation**:
- Subjective assessment: "This answer looks good"
- Blind to regressions: Code changes may degrade quality unnoticed
- No optimization guidance: Can't identify bottlenecks (retrieval vs generation)

**With Evaluation**:
- Objective metrics: Quantifiable scores (0-1 scale)
- Regression detection: Catch quality degradation before production
- Bottleneck identification: Pinpoint if issues are retrieval or generation
- A/B testing: Compare prompts, models, retrieval strategies scientifically
- Continuous improvement: Track quality improvements over time

## Architecture

```
apps/api/evals/
├── __init__.py              # Makes evals a Python package
└── eval_retriever.py        # Main evaluation script (317 lines)
```

## Evaluation Metrics

### 1. Faithfulness (LLM-Based)

**Question**: "Is the LLM making things up or staying grounded in retrieved context?"

**How It Works**:
- LLM judge (gpt-4.1-mini) checks if answer statements can be verified from retrieved products
- Breaks answer into atomic claims
- Verifies each claim against context
- Score: Claims verified / Total claims

**Score Range**: 0 (hallucination) to 1 (fully grounded)

**Example**:
- Context: "Product X has 8GB RAM"
- Good Answer: "This product has 8GB RAM" → **Score: 1.0**
- Bad Answer: "This product has 16GB RAM" → **Score: 0.0**

**Why It Matters**: Prevents LLM from inventing product features, ensures trustworthy answers.

**Cost**: Requires OpenAI API call per evaluation (gpt-4.1-mini)

### 2. Response Relevancy (LLM + Embeddings)

**Question**: "Does the answer actually address the user's question?"

**How It Works**:
- Generate hypothetical questions from the answer (using LLM)
- Compare these questions to the original using embeddings
- High similarity = relevant answer, low similarity = off-topic

**Score Range**: 0 (off-topic) to 1 (highly relevant)

**Example**:
- Question: "What are the best wireless headphones under $100?"
- Good Answer: "Based on the products, the XYZ headphones at $89..." → **High score**
- Bad Answer: "Headphones are audio devices worn on the head..." → **Low score**

**Why It Matters**: Prevents generic or evasive responses, ensures answers are useful.

**Cost**: Requires OpenAI API calls (gpt-4.1-mini + text-embedding-3-small)

### 3. Context Precision (ID-Based)

**Question**: "What percentage of retrieved products are actually relevant?"

**Formula**: `|retrieved ∩ reference| / |retrieved|`

**How It Works**:
- Compare retrieved product IDs to reference (ground truth) IDs
- Calculate: relevant_retrieved / total_retrieved

**Score Range**: 0 (all noise) to 1 (all relevant)

**Example**:
- Retrieved: [A, B, C, D, E] (5 products)
- Reference (relevant): [A, B, F, G]
- Precision: 2/5 = **0.4** (only A and B were relevant)

**Why It Matters**: High precision = user sees mostly relevant products, less noise.

**Cost**: No API calls (just ID comparison)

### 4. Context Recall (ID-Based)

**Question**: "What percentage of relevant products did we successfully retrieve?"

**Formula**: `|retrieved ∩ reference| / |reference|`

**How It Works**:
- Compare retrieved product IDs to reference (ground truth) IDs
- Calculate: relevant_retrieved / total_relevant

**Score Range**: 0 (missed everything) to 1 (found all relevant items)

**Example**:
- Retrieved: [A, B, C, D, E] (5 products)
- Reference (relevant): [A, B, F, G] (4 relevant products exist)
- Recall: 2/4 = **0.5** (found A and B, but missed F and G)

**Why It Matters**: High recall = user doesn't miss good product options.

**Cost**: No API calls (just ID comparison)

### Precision vs Recall Trade-off

**Understanding the Balance**:
- **High top_k** → Better recall (find more), worse precision (more noise)
- **Low top_k** → Better precision (less noise), worse recall (miss some)
- **Optimal**: Balance based on use case (e.g., top_k=5-10 for e-commerce)

**Tuning Strategy**:
1. Start with baseline (current settings)
2. Adjust `top_k` in retrieval step
3. Run evaluations to measure impact
4. Find sweet spot for your use case

## Running Evaluations

### Quick Start

From project root:
```bash
make run-evals-retriever
```

### Manual Execution

```bash
# 1. Sync dependencies (ensure ragas, langsmith installed)
uv sync

# 2. Run evaluation with proper environment setup
PYTHONPATH=${PWD}/apps/api/src:$PYTHONPATH:${PWD} \
  uv run --env-file .env \
  python apps/api/evals/eval_retriever.py
```

### What Happens

1. **Connects to LangSmith**: Fetches "rag-evaluation-dataset" (43 examples)
2. **Runs RAG Pipeline**: Calls `rag_pipeline(question)` for each example
3. **Computes Scores**: Runs all 4 evaluators on each result
4. **Stores Results**: 172 total scores (43 examples × 4 metrics) in LangSmith
5. **Prints URL**: LangSmith experiment URL for viewing results

**Output Example**:
```
View results at: https://smith.langchain.com/o/.../datasets/.../compare?selectedSessions=...
```

## Understanding Results

### LangSmith UI

**What You Can See**:
- **Aggregate Metrics**: Average scores across all examples
- **Per-Example Scores**: Individual scores for each question
- **Trace Details**: Full RAG pipeline execution (retrieval, prompts, answers)
- **Score Distribution**: Histogram of scores for each metric
- **Failed Examples**: Filter by low scores to debug issues

**Key Views**:
1. **Comparison View**: Side-by-side comparison of multiple runs
2. **Trace Explorer**: Drill into specific examples to see:
   - Retrieved products and similarity scores
   - Exact prompts sent to LLM
   - Token usage and costs
   - Execution time per step

### Interpreting Scores

**Good Scores** (System is Working Well):
- **Faithfulness > 0.8**: Answers grounded in context
- **Response Relevancy > 0.7**: Answers address questions
- **Precision > 0.6**: More than half of retrieved products relevant
- **Recall > 0.5**: Retrieving at least half of relevant products

**Warning Signs** (System Needs Improvement):
- **Low Faithfulness (<0.5)**: LLM hallucinating, not using context → Fix prompt
- **Low Relevancy (<0.5)**: Answers generic or off-topic → Improve prompt or LLM
- **Low Precision (<0.4)**: Too much noise → Increase similarity threshold or reduce top_k
- **Low Recall (<0.3)**: Missing relevant products → Increase top_k or lower threshold

## Implementation Details

### Evaluator Function Pattern

Each metric is a Python function following this pattern:

```python
def ragas_faithfulness(run, example):
    """
    Args:
        run: LangSmith Run object with RAG pipeline outputs
        example: LangSmith Example object with reference data

    Returns:
        float: Score between 0 and 1
    """
    # 1. Extract outputs safely (defensive programming)
    outputs = run.outputs if isinstance(run.outputs, dict) else {}

    # 2. Create RAGAS sample structure
    sample = SingleTurnSample(
        user_input=outputs.get("question", ""),
        response=outputs.get("answer", ""),
        retrieved_contexts=outputs.get("retrieved_context", [])
    )

    # 3. Initialize scorer and compute score
    scorer = Faithfulness(llm=ragas_llm)
    return scorer.single_turn_score(sample)
```

**Key Patterns**:
- **Defensive Extraction**: Use `.get()` with defaults (pipeline might error)
- **Type Checking**: Verify `run.outputs` is dict before accessing
- **Synchronous**: Regular functions (not async) - LangSmith runs in thread pools
- **RAGAS Data Structure**: Must use `SingleTurnSample` for RAGAS compatibility

### LangSmith Integration

```python
results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"]),        # What to evaluate
    data="rag-evaluation-dataset",                # Test dataset
    evaluators=[                                  # Scoring functions
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based
    ],
    experiment_prefix="retriever",                # Results grouping
    max_concurrency=5                             # Parallel execution
)
```

**Concurrency Tuning**:
- Default: 5 (balance speed and API rate limits)
- Lower (2-3): More stable, less likely to hit rate limits
- Higher (10+): Faster but may cause connection errors

## Environment Variables

Required in `.env`:

```env
# LangSmith (evaluation platform)
LANGSMITH_API_KEY=<your-key>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=rag-tracing
LANGSMITH_TRACING=true

# OpenAI (for RAGAS LLM judge and embeddings)
OPENAI_KEY=sk-...

# Qdrant (vector database)
QDRANT_URL=http://localhost:6333  # Local execution (not Docker)
```

**Important**: `QDRANT_URL` must be `http://localhost:6333` for local evaluation script (not `http://qdrant:6333` which only works in Docker).

## Lessons Learned

### 1. Module Structure Requirements
- **Problem**: `ModuleNotFoundError: No module named 'api.evals'`
- **Cause**: Missing `__init__.py` in `apps/api/evals/` directory
- **Fix**: Create `__init__.py` (can be empty or with docstring)
- **Prevention**: Every Python package directory needs `__init__.py`

### 2. Async/Sync Execution Context
- **Problem**: `RuntimeError: There is no current event loop in thread`
- **Cause**: Evaluator functions were `async def` but LangSmith uses thread pools
- **Fix**: Use regular `def` (not `async def`) and `scorer.single_turn_score()` (not `await ...ascore()`)
- **Takeaway**: LangSmith evaluators must be synchronous

### 3. LangSmith Run Object Structure
- **Problem**: `TypeError: 'RunTree' object is not subscriptable` with `run["question"]`
- **Cause**: `run` is a RunTree object, not a dict
- **Fix**: Access via `run.outputs["question"]` to get outputs dictionary
- **Pattern**: `run.outputs` = pipeline return value, `run.inputs` = pipeline inputs

### 4. Defensive Data Access
- **Problem**: `KeyError: 'question'` when pipeline errors on some examples
- **Cause**: If RAG pipeline fails, `run.outputs` might not have expected keys
- **Fix**: Use `outputs = run.outputs if isinstance(run.outputs, dict) else {}` and `.get()` with defaults
- **Benefit**: Evaluation continues even if some examples fail

### 5. Qdrant Connection in Local Execution
- **Problem**: `qdrant:6333` doesn't resolve when running locally
- **Cause**: `qdrant` is Docker Compose service name, not hostname
- **Fix**: Changed `retrieval_generation.py` to use `http://localhost:6333`
- **Trade-off**: Works for local eval but breaks in Docker
- **Better Solution**: Pass `qdrant_url` as parameter (not implemented)

### 6. Transient API Errors
- **Problem**: `openai.APIConnectionError: Connection error` during evaluation
- **Cause**: Network instability or OpenAI rate limits
- **Status**: Not a code bug - evaluation continues and processes most examples
- **Mitigation**: Reduce `max_concurrency` (from 5 to 2-3) if frequent
- **Impact**: Only affects `ragas_response_relevancy` (requires OpenAI calls)

### 7. RAGAS Dependencies
- **Required**: `ragas`, `langsmith`, `langchain-openai` packages
- **Models**: gpt-4.1-mini (LLM judge), text-embedding-3-small (embeddings)
- **Why Same Embedding Model**: Must match preprocessing embeddings for consistency
- **Cost**: Evaluation calls OpenAI API (judge LLM + similarity embeddings)

## Cost Analysis

**Per Evaluation Run** (43 examples):
1. **Faithfulness**: 43 × gpt-4.1-mini calls = ~$0.10
2. **Response Relevancy**: 43 × (gpt-4.1-mini + embeddings) = ~$0.15
3. **Context Precision**: Free (ID comparison only)
4. **Context Recall**: Free (ID comparison only)

**Total Cost**: ~$0.25 per evaluation run

**Optimization**:
- Reduce evaluation frequency (weekly vs daily)
- Sample subset of dataset (10 examples vs 43)
- Use cheaper LLM judge (gpt-4.1-nano if available)

## Next Steps

### 1. Optimize Retrieval Parameters
- Experiment with `top_k` (default 5)
- Experiment with similarity threshold filtering
- Re-run evaluations after each change
- Track scores in spreadsheet or dashboard

### 2. Improve Prompts
- Modify system prompt in `build_prompt()`
- Test different instruction phrasing
- Compare results in LangSmith

### 3. Expand Dataset
- Add more test questions (current: 43)
- Include edge cases and failure scenarios
- Test multi-turn conversations

### 4. Continuous Evaluation
- Run `make run-evals-retriever` before every commit
- Track scores over time (manual or automated)
- Create baseline scores for comparison
- Set up CI/CD gate (fail if scores drop >10%)

## Related Documentation

- **Parent API**: [../README.md](../README.md) - API architecture
- **RAG Pipeline**: [../src/api/agents/README.md](../src/api/agents/README.md) - Implementation details
- **Evaluation Dataset**: [../../../notebooks/week1/README.md](../../../notebooks/week1/README.md#04-evaluation-dataset) - Dataset creation
- **Project Root**: [../../../README.md](../../../README.md) - Overall architecture

## RAGAS Resources

- **Documentation**: https://docs.ragas.io
- **Metrics Guide**: https://docs.ragas.io/en/latest/concepts/metrics/
- **LangSmith Integration**: https://docs.smith.langchain.com/evaluation

## Troubleshooting

**ModuleNotFoundError**:
- Ensure `__init__.py` exists in `apps/api/evals/`
- Verify PYTHONPATH includes `apps/api/src`

**Async Event Loop Errors**:
- Use `def` not `async def` for evaluator functions
- Use `.single_turn_score()` not `await .ascore()`

**Qdrant Connection Errors**:
- Local execution: `http://localhost:6333`
- Docker: `http://qdrant:6333`
- Verify Qdrant is running: `curl http://localhost:6333/collections`

**OpenAI API Errors**:
- Check `OPENAI_KEY` in `.env`
- Reduce `max_concurrency` if rate limited
- Verify API key has credits

**LangSmith Not Showing Results**:
- Verify all 4 `LANGSMITH_*` environment variables set
- Check API key validity at https://smith.langchain.com
- Ensure `LANGSMITH_TRACING=true`
