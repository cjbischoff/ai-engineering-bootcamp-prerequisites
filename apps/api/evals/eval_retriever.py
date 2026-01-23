"""
RAG Pipeline Evaluation Script using RAGAS Metrics

This script evaluates the quality of our RAG (Retrieval-Augmented Generation) pipeline
by running it against a test dataset and measuring performance with RAGAS metrics.

Purpose:
    - Systematically test RAG pipeline against known question-answer pairs
    - Measure retrieval quality (are we finding the right products?)
    - Measure generation quality (are answers faithful and relevant?)
    - Track performance over time to prevent regressions

Key Concepts:
    - LangSmith: Platform for running evaluations and viewing results
    - RAGAS: Framework providing RAG-specific evaluation metrics
    - Evaluator Functions: Custom functions that score each RAG output
    - Dataset: Collection of test questions with expected answers/context

How to Run:
    make run-evals-retriever

Results Location:
    View at the LangSmith URL printed when the script runs
"""

# Import our RAG pipeline that we want to evaluate
from api.agents.retrieval_generation import rag_pipeline

# LangSmith client for running evaluations and storing results
from langsmith import Client

# OpenAI models for RAGAS metrics (RAGAS uses LLMs to judge quality)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# RAGAS wrappers to make LangChain models work with RAGAS
# Note: These are deprecated but still functional. Modern approach uses llm_factory()
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# RAGAS data structure for representing a single evaluation example
from ragas.dataset_schema import SingleTurnSample

# RAGAS metrics for measuring RAG quality
# Each metric answers a specific question about RAG performance
from ragas.metrics import (
    IDBasedContextPrecision,  # "Did we retrieve the RIGHT products?"
    IDBasedContextRecall,     # "Did we retrieve ALL relevant products?"
    Faithfulness,             # "Is the answer grounded in retrieved context?"
    ResponseRelevancy         # "Does the answer actually address the question?"
)


# Initialize LangSmith client to run evaluations
# This connects to your LangSmith account and enables:
# - Running evaluations against datasets
# - Storing evaluation results
# - Viewing results in the web UI
ls_client = Client()

# Configure LLM for RAGAS metrics that need to judge quality
# Why gpt-4.1-mini? Balance of quality and cost for evaluation judgments
# RAGAS uses this LLM to:
# - Determine if answers are faithful to context
# - Generate test questions to check answer relevancy
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))

# Configure embeddings for RAGAS metrics that need semantic similarity
# Why text-embedding-3-small? Same model used in our RAG pipeline (consistency)
# RAGAS uses these embeddings to:
# - Compare semantic similarity between questions and answers
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)


def ragas_faithfulness(run, example):
    """
    Evaluator Function: Measures if the answer is faithful to retrieved context

    Question Answered: "Is the LLM making things up or staying grounded in the retrieved products?"

    How It Works:
        1. Extracts question, answer, and retrieved context from the RAG run
        2. Uses an LLM judge to determine if statements in the answer can be
           verified from the retrieved context
        3. Returns a score from 0 (hallucination) to 1 (fully grounded)

    Why This Matters:
        - Prevents the LLM from inventing product features
        - Ensures answers are trustworthy and verifiable
        - Critical for e-commerce where accuracy is essential

    Example:
        Context: "Product X has 8GB RAM"
        Good Answer: "This product has 8GB RAM" (score: 1.0)
        Bad Answer: "This product has 16GB RAM" (score: 0.0)

    Args:
        run: LangSmith Run object containing RAG pipeline outputs
        example: LangSmith Example object containing test data (unused here)

    Returns:
        float: Faithfulness score between 0 and 1
    """
    # Extract outputs safely from the run object
    # Why safe extraction? run.outputs might not be a dict if the pipeline errored
    outputs = run.outputs if isinstance(run.outputs, dict) else {}

    # Create a RAGAS sample with the data needed for faithfulness scoring
    # SingleTurnSample = one question-answer exchange (not a conversation)
    sample = SingleTurnSample(
        user_input=outputs.get("question", ""),           # The original question
        response=outputs.get("answer", ""),               # The LLM-generated answer
        retrieved_contexts=outputs.get("retrieved_context", [])  # Product descriptions
    )

    # Create a faithfulness scorer with our configured LLM judge
    # This scorer will use the LLM to check if the answer is grounded
    scorer = Faithfulness(llm=ragas_llm)

    # Compute and return the faithfulness score
    # single_turn_score() is synchronous (blocks until complete)
    return scorer.single_turn_score(sample)


def ragas_response_relevancy(run, example):
    """
    Evaluator Function: Measures if the answer actually addresses the question

    Question Answered: "Is the LLM answering the right question or going off-topic?"

    How It Works:
        1. Uses the answer to generate hypothetical questions
        2. Compares these questions to the original question using embeddings
        3. High similarity = relevant answer, low similarity = off-topic

    Why This Matters:
        - Ensures the LLM doesn't provide generic responses
        - Checks if the answer is useful to the user
        - Detects when the LLM avoids the question

    Example:
        Question: "What are the best wireless headphones under $100?"
        Good Answer: "Based on the products, the XYZ headphones at $89..." (high score)
        Bad Answer: "Headphones are audio devices worn on the head..." (low score)

    Args:
        run: LangSmith Run object containing RAG pipeline outputs
        example: LangSmith Example object containing test data (unused here)

    Returns:
        float: Relevancy score between 0 and 1
    """
    # Extract outputs safely (same pattern as faithfulness)
    outputs = run.outputs if isinstance(run.outputs, dict) else {}

    # Create sample for relevancy scoring
    sample = SingleTurnSample(
        user_input=outputs.get("question", ""),
        response=outputs.get("answer", ""),
        retrieved_contexts=outputs.get("retrieved_context", [])
    )

    # Create relevancy scorer with both LLM and embeddings
    # Why both? LLM generates test questions, embeddings compare similarity
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    # Compute and return the relevancy score
    return scorer.single_turn_score(sample)


def ragas_context_precision_id_based(run, example):
    """
    Evaluator Function: Measures retrieval precision using product IDs

    Question Answered: "What percentage of retrieved products are actually relevant?"

    How It Works:
        1. Compares retrieved product IDs to reference (ground truth) IDs
        2. Calculates: relevant_retrieved / total_retrieved
        3. Higher score = less noise in retrieval results

    Why This Matters:
        - High precision = user sees mostly relevant products
        - Low precision = user wastes time on irrelevant results
        - Helps tune retrieval parameters (top_k, similarity threshold)

    Example:
        Retrieved: [A, B, C, D, E] (5 products)
        Reference (relevant): [A, B, F, G]
        Precision: 2/5 = 0.4 (only A and B were relevant)

    Formula:
        Precision = |retrieved ∩ reference| / |retrieved|

    Args:
        run: LangSmith Run object containing RAG pipeline outputs
        example: LangSmith Example object containing reference data

    Returns:
        float: Precision score between 0 and 1
    """
    # Extract outputs from both run and example
    # Why example? We need the ground truth reference IDs
    outputs = run.outputs if isinstance(run.outputs, dict) else {}
    example_outputs = example.outputs if isinstance(example.outputs, dict) else {}

    # Create sample with ID lists for comparison
    sample = SingleTurnSample(
        retrieved_context_ids=outputs.get("retrieved_context_ids", []),      # What we retrieved
        reference_context_ids=example_outputs.get("reference_context_ids", [])  # What we should have retrieved
    )

    # Create precision scorer (no LLM needed - just ID comparison)
    scorer = IDBasedContextPrecision()

    # Compute and return precision score
    return scorer.single_turn_score(sample)


def ragas_context_recall_id_based(run, example):
    """
    Evaluator Function: Measures retrieval recall using product IDs

    Question Answered: "What percentage of relevant products did we successfully retrieve?"

    How It Works:
        1. Compares retrieved product IDs to reference (ground truth) IDs
        2. Calculates: relevant_retrieved / total_relevant
        3. Higher score = fewer missed relevant products

    Why This Matters:
        - High recall = user doesn't miss good product options
        - Low recall = relevant products hidden from user
        - Helps determine if top_k is too low

    Example:
        Retrieved: [A, B, C, D, E] (5 products)
        Reference (relevant): [A, B, F, G] (4 relevant products exist)
        Recall: 2/4 = 0.5 (we found A and B, but missed F and G)

    Formula:
        Recall = |retrieved ∩ reference| / |reference|

    Precision vs Recall Trade-off:
        - High k → better recall, worse precision (retrieve more, more noise)
        - Low k → better precision, worse recall (retrieve less, miss relevant items)

    Args:
        run: LangSmith Run object containing RAG pipeline outputs
        example: LangSmith Example object containing reference data

    Returns:
        float: Recall score between 0 and 1
    """
    # Extract outputs from both run and example (same as precision)
    outputs = run.outputs if isinstance(run.outputs, dict) else {}
    example_outputs = example.outputs if isinstance(example.outputs, dict) else {}

    # Create sample with ID lists for comparison
    sample = SingleTurnSample(
        retrieved_context_ids=outputs.get("retrieved_context_ids", []),
        reference_context_ids=example_outputs.get("reference_context_ids", [])
    )

    # Create recall scorer (no LLM needed - just ID comparison)
    scorer = IDBasedContextRecall()

    # Compute and return recall score
    return scorer.single_turn_score(sample)


# Run the evaluation
# This is the main execution that ties everything together
results = ls_client.evaluate(
    # Target function: What we're evaluating
    # Lambda takes dataset input (dict with "question" key) and calls our RAG pipeline
    # Why lambda? Allows us to adapt the pipeline to LangSmith's expected interface
    lambda x: rag_pipeline(x["question"]),

    # Dataset: Test cases to evaluate against
    # This references a LangSmith dataset created in the 04-evaluation-dataset notebook
    # Contains 43 question-answer pairs with reference product IDs
    data="rag-evaluation-dataset",

    # Evaluators: List of scoring functions to run on each example
    # Each evaluator receives the run output and computes a score
    # All four evaluators run on every example in the dataset
    evaluators=[
        ragas_faithfulness,              # Generation quality: Is answer grounded?
        ragas_response_relevancy,        # Generation quality: Is answer on-topic?
        ragas_context_precision_id_based,  # Retrieval quality: Precision
        ragas_context_recall_id_based      # Retrieval quality: Recall
    ],

    # Experiment prefix: Naming for organizing results in LangSmith
    # Results appear as "retriever-{timestamp}" in the UI
    # Helps group related evaluation runs together
    experiment_prefix="retriever",

    # Concurrency: How many examples to evaluate in parallel
    # Why 5? Balance between speed and API rate limits
    # Lower = slower but more stable, Higher = faster but may hit rate limits
    # Recommendation: Start with 2-3 if you hit API connection errors
    max_concurrency=5
)

# After evaluation completes, results are available at:
# https://smith.langchain.com/... (URL printed during execution)
#
# In LangSmith UI you can:
# - View individual run traces (see what the RAG pipeline did)
# - Compare scores across evaluators (which metrics are high/low?)
# - Track scores over time (are changes improving quality?)
# - Filter by score range (find examples with low scores to debug)
# - Export results to CSV for further analysis
