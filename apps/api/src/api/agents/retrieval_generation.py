"""
RAG (Retrieval-Augmented Generation) Pipeline for Product Q&A

This module implements a complete RAG pipeline for answering questions about Amazon Electronics products.
The pipeline follows these steps:
1. Convert user query to embeddings (semantic representation)
2. Retrieve similar products from Qdrant vector database
3. Format retrieved context into a readable structure
4. Build a prompt with context and question
5. Generate answer using OpenAI's language model

Architecture:
- Embeddings: OpenAI text-embedding-3-small (1536 dimensions)
- Vector Store: Qdrant with cosine similarity
- LLM: OpenAI GPT-5-nano with minimal reasoning effort
- Data: Amazon Electronics products (2022-2023, 50 items)

Observability (Video 5):
- LangSmith Integration: Distributed tracing for RAG pipeline
- Decorator Pattern: @traceable marks functions for automatic instrumentation
- Trace Hierarchy: Each step becomes a span in the trace tree
- Token Tracking: Captures OpenAI API usage (prompt_tokens, completion_tokens, total_tokens)
- Run Metadata: Enriches traces with model names, providers, and usage stats
- Why LangSmith: Purpose-built for LLM observability vs generic APM tools
  * Traces show prompt/response pairs, not just timing
  * Visualizes RAG workflow: retrieval → formatting → generation
  * Tracks token costs across multiple LLM calls
  * Enables debugging: "Why did this query return a poor answer?"
"""

import openai
from langsmith import get_current_run_tree, traceable
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from pydantic import BaseModel, Field
import instructor
import numpy as np

from api.api.models import RAGResponse

# Define nested Pydantic models for grounded structured outputs


class RAGUsedContext(BaseModel):
    """
    Represents a single product reference (grounding context).

    This model captures information about one product that was used
    to generate the answer. Multiple instances of this model will be
    returned in the references list.

    Fields:
        id: The product ASIN (Amazon Standard Identification Number)
            - Unique identifier for the product
            - Can be used to link back to full product details
            - Example: "B09X12ABC"

        description: Short human-readable summary (1-2 sentences)
            - Should include the product name
            - Brief description of what the product is
            - Example: "GREPHONE USB-C to Lightning Cable - 6ft MFi certified charging cable"

    Why this structure?
        - Balances detail vs. brevity
        - ID enables programmatic lookup
        - Description enables human understanding
    """

    id: str = Field(description="The unique identifier to the answer the question")
    description: str = Field(
        description="Short description of the item used to answer the question"
    )


class RAGGenerationResponse(BaseModel):
    """
    Complete structured response with grounding context.

    This is the TOP-LEVEL model that instructor will validate and return.
    It contains both the answer and the list of source references.

    Fields:
        answer: The full natural language answer to the user's question
            - Should be detailed and informative
            - Should reference products by name
            - Should include specifications in bullet points

        references: List of RAGUsedContext objects (grounding)
            - CRITICAL: Only include products actually used in the answer
            - Not all retrieved products will be used
            - The LLM must decide which products are relevant
            - Each reference links the answer back to its source

    Why nested structure?
        - Instructor automatically validates nested Pydantic models
        - Type-safe: references must be list[RAGUsedContext]
        - Self-documenting: structure makes relationship clear
        - Enables rich tooling: IDEs can autocomplete fields

    Common Pitfall:
        Don't confuse retrieved_context (all products fetched from DB)
        with references (only products used in answer). The LLM performs
        this filtering based on relevance to the specific question.
    """

    answer: str = Field(description="The answer to the question")
    references: list[RAGUsedContext] = Field(
        description="List of items used to generate the answer"
    )







@traceable(
    name="Get Embedding",  # Display name in LangSmith trace UI
    run_type="embedding",  # Categorizes this span as an embedding operation
    metadata={"ls_model_name": "text-embedding-3-small", "ls_provider": "openai"},  # Static metadata attached to all traces
)
def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate embeddings for text using OpenAI's embedding model.

    Embeddings are dense vector representations that capture semantic meaning.
    Similar texts will have similar embeddings (measured by cosine similarity).

    Args:
        text (str): The text to embed (query or document)
        model (str): OpenAI embedding model name. Default is text-embedding-3-small
                     which produces 1536-dimensional vectors

    Returns:
        list[float]: A 1536-dimensional vector representing the semantic meaning of the text

    Why: Embeddings allow us to find semantically similar products, not just keyword matches.
         "affordable laptop" will match products describing "budget-friendly computers"

    Observability (Video 5):
        The @traceable decorator automatically:
        - Creates a span in the LangSmith trace tree
        - Captures function inputs (text) and outputs (embedding vector)
        - Records execution time and any errors
        - Links to parent spans (e.g., retrieve_data or rag_pipeline)

        Manual token tracking added:
        - get_current_run_tree() retrieves the active LangSmith run context
        - We manually add usage_metadata with token counts from OpenAI response
        - This enables cost tracking: input_tokens × $0.00002 per 1K tokens
        - LangSmith aggregates tokens across all embedding calls in a trace
    """
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    # Manual instrumentation: Capture OpenAI API usage for cost tracking
    # LangSmith doesn't auto-capture embeddings API usage, so we add it manually
    current_run = get_current_run_tree()

    if current_run:
        # Attach token usage to the current trace span
        # This appears in LangSmith UI under "Metadata" for this embedding call
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,  # Number of tokens in input text
            "total_tokens": response.usage.total_tokens,  # Total tokens processed (input only for embeddings)
        }

    return response.data[0].embedding


@traceable(name="Retrieve Data", run_type="retriever")  # LangSmith categorizes as a retrieval operation
def retrieve_data(query, qdrant_client, k=5):
    """
    Retrieve top-k most relevant products from Qdrant vector database.

    This implements the "Retrieval" part of RAG. It converts the user's query to an
    embedding and finds products with similar embeddings (semantic search).

    Args:
        query (str): User's question (e.g., "best headphones for gaming")
        qdrant_client (QdrantClient): Connected Qdrant client instance
        k (int): Number of similar products to retrieve. Default is 5

    Returns:
        dict: Contains four parallel lists:
            - retrieved_context_ids: Product ASINs (Amazon IDs)
            - retrieved_context: Product descriptions
            - retrieved_context_ratings: Average customer ratings
            - similarity_scores: Cosine similarity scores (0-1, higher = more similar)

    How it works:
        1. Convert query text to embedding vector
        2. Search Qdrant collection for nearest neighbor vectors
        3. Qdrant uses HNSW algorithm for fast approximate nearest neighbor search
        4. Extract product metadata from matching results

    Why k=5: Balance between providing enough context and keeping prompt concise

    Observability (Video 5):
        The @traceable decorator captures:
        - Input: user query text and k parameter
        - Output: dictionary with retrieved products and similarity scores
        - Nested span: Calls get_embedding() which creates a child span in the trace
        - Retrieval quality: Similarity scores visible in LangSmith for debugging
          * Low scores (<0.7) may indicate poor matches
          * Can analyze: "Why did this query return irrelevant products?"

        Trace hierarchy in LangSmith:
        RAG Pipeline (parent)
        └── Retrieve Data (this function)
            └── Get Embedding (child - embedding the query)
    """
    # Step 1: Convert user query to embedding (same 1536-dim space as products)
    query_embedding = get_embedding(query)

    # Step 2: Search Qdrant for k nearest neighbors using cosine similarity
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",  # Collection created in preprocessing notebook
        query=query_embedding,
        limit=k,
    )

    # Step 3: Initialize lists to store retrieved data
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    # Step 4: Extract metadata from each matching product
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])  # Product ID
        retrieved_context.append(
            result.payload["description"]
        )  # Combined title + features
        retrieved_context_ratings.append(
            result.payload["average_rating"]
        )  # Customer rating
        similarity_scores.append(result.score)  # How similar to query (0-1)

    # Return structured data for downstream processing
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


@traceable(name="Format Retrieved Context", run_type="prompt")  # LangSmith categorizes as prompt formatting
def process_context(context):
    """
    Format retrieved product data into a human-readable string for the LLM.

    This converts structured data (lists of IDs, descriptions, ratings) into
    a formatted text block that the language model can understand.

    Args:
        context (dict): Dictionary with retrieved_context_ids, retrieved_context,
                       and retrieved_context_ratings lists

    Returns:
        str: Formatted multi-line string with one product per line:
             "- ID: {asin}, rating: {rating}, description: {chunk}\n"

    Why this format:
        - Bullet points help LLM parse separate products
        - Including ID allows user to reference specific products
        - Rating helps LLM prioritize highly-rated items
        - Description provides the actual content for answering

    Example output:
        - ID: B09X12ABC, rating: 4.5, description: Wireless gaming headset...
        - ID: B09Y34DEF, rating: 4.2, description: USB microphone for streaming...

    Observability (Video 5):
        The @traceable decorator captures:
        - Input: structured context dict with lists
        - Output: formatted string ready for LLM prompt
        - run_type="prompt": Indicates this is prompt engineering/formatting
        - Enables debugging: "Is the context format causing poor answers?"
        - Can see exact text passed to LLM in next step
    """
    formatted_context = ""

    # Zip iterates through three lists in parallel (product by product)
    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@traceable(name="Build Prompt", run_type="prompt")  # LangSmith categorizes as prompt construction
def build_prompt(preprocessed_context, question):
    """
    Construct the final prompt sent to the language model.

    This implements prompt engineering for the RAG system. The prompt structure:
    1. System role: You are a shopping assistant
    2. Task: Answer questions about products
    3. Context: Retrieved product information
    4. Constraint: Only use provided context
    5. Question: User's actual query

    Args:
        preprocessed_context (str): Formatted string of retrieved products
        question (str): User's original question

    Returns:
        str: Complete prompt with system instructions, context, and question

    Why this structure:
        - Clear role definition improves response quality
        - "Only use provided context" prevents hallucination
        - "Refer to available products" makes responses more natural
        - Separating Context and Question helps LLM focus on both parts

    Prompt engineering best practices applied:
        - Explicit instructions reduce ambiguity
        - Examples could be added for few-shot learning
        - Triple-quoted string preserves formatting

    Observability (Video 5):
        The @traceable decorator captures:
        - Input: formatted context and user question
        - Output: complete prompt string sent to LLM
        - Critical for debugging: Can see exact prompt in LangSmith UI
        - Enables prompt iteration: Test different instructions/formats
        - Can analyze: "Is the prompt structure causing poor answers?"
    """
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.
- As an output you need to provide:

* The answer to the question based on the provided context.
* The list of the IDs of the chunks that were used to answer the question. Only return the ones that are used in the answer.
* Short description (1-2 sentences) of the item based on the description provided in the context.

- The short description should have the name of the item.
- The answer to the question should contain detailed information about the product and returned with detailed specification in bullet points.


Context:
{preprocessed_context}

Question:
{question}
"""

    return prompt


@traceable(
    name="Generate Answer",  # Display name in LangSmith trace UI
    run_type="llm",  # Categorizes this span as an LLM call
    metadata={"ls_model_name": "gpt-4.1-mini", "ls_provider": "openai"},  # Static metadata for filtering traces
)
def generate_answer(prompt):
    """
    Generate final answer using OpenAI's language model.

    This implements the "Generation" part of RAG. The LLM reads the prompt
    (which includes retrieved context) and generates a natural language answer.

    Args:
        prompt (str): Complete prompt with instructions, context, and question

    Returns:
        str: Natural language answer to the user's question

    Model configuration:
        - model: gpt-5-nano (smaller, faster model for simple Q&A)
        - messages: Single system message (no conversation history)
        - reasoning_effort: "minimal" (faster responses, less complex reasoning)

    Why these choices:
        - gpt-5-nano: Cost-effective for straightforward retrieval-based answers
        - System role: Provides all context upfront vs user/assistant back-and-forth
        - Minimal reasoning: Products speak for themselves, no complex logic needed

    Alternative approaches:
        - Could use streaming for real-time responses
        - Could add temperature parameter for creativity control
        - Could implement caching for repeated questions

    Observability (Video 5):
        The @traceable decorator automatically:
        - Creates an LLM span in the trace hierarchy
        - Captures prompt input and generated answer output
        - Records model name, provider, and execution time
        - Links to parent spans (build_prompt, rag_pipeline)

        Manual token tracking added:
        - get_current_run_tree() retrieves the active LangSmith run context
        - We manually add usage_metadata with detailed token breakdowns
        - input_tokens: Prompt tokens (system instructions + context + question)
        - output_tokens: Generated answer tokens
        - total_tokens: Sum of input + output (for cost calculation)
        - Cost tracking: gpt-5-nano pricing × total_tokens
        - LangSmith aggregates tokens across RAG pipeline (embedding + generation)

        Why manual tracking:
        - LangSmith doesn't auto-capture OpenAI SDK usage stats
        - We need token counts for accurate cost monitoring
        - Enables analysis: "Which queries are most expensive?"
    """
    client = instructor.from_openai(openai.OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=RAGGenerationResponse,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )

    # Manual instrumentation: Capture OpenAI API usage for cost tracking
    # This is the most expensive call in the pipeline (LLM generation costs more than embeddings)
    current_run = get_current_run_tree()

    if current_run:
        # Attach detailed token usage to the current trace span
        # This appears in LangSmith UI under "Metadata" for this LLM call
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,  # All prompt text (instructions + context + question)
            "output_tokens": raw_response.usage.completion_tokens,  # Generated answer length
            "total_tokens": raw_response.usage.total_tokens,  # Total tokens processed (input + output)
        }

    return response


@traceable(
    name="RAG Pipeline",  # Root span name in LangSmith trace tree
)
def rag_pipeline(question, top_k=5):
    """
    Complete RAG pipeline: Retrieve relevant products and generate an answer.

    This orchestrates the full RAG workflow from query to answer.
    Called by the FastAPI endpoint when a user submits a question.

    Args:
        question (str): User's question about products
        top_k (int): Number of products to retrieve. Default is 5

    Returns:
        dict: Structured response containing:
            - answer (str): Natural language answer generated by the LLM
            - question (str): Original user query (for reference)
            - retrieved_context_ids (list): Product ASINs that were retrieved
            - retrieved_context (list): Product descriptions used for answering
            - similarity_scores (list): Cosine similarity scores for retrieved products

        Changed in Video 5: Previously returned just a string (answer), now returns
        a structured dict with answer + metadata for observability and debugging.

    Pipeline stages:
        1. RETRIEVE: Query Qdrant for k similar products (semantic search)
        2. PREPROCESS: Format products into readable text
        3. AUGMENT: Build prompt with context and question
        4. GENERATE: LLM produces final answer
        5. PACKAGE: Return answer with retrieval metadata (NEW in Video 5)

    Why RAG vs pure LLM:
        - LLM alone: May hallucinate product details, outdated knowledge
        - RAG: Grounds responses in actual product data, always current
        - Benefit: Accurate, verifiable answers with product references

    Production considerations:
        - Connection pooling: Creating new Qdrant client per request is inefficient
        - Caching: Could cache embeddings for common queries
        - Error handling: No try/except blocks (TODO for production)
        - Monitoring: Should log retrieval quality and answer relevance

    Docker networking:
        - url="http://qdrant:6333": 'qdrant' is Docker Compose service name
        - Containers communicate by service name, not localhost

    Observability (Video 5):
        The @traceable decorator creates the ROOT span of the trace tree:
        - All other functions (get_embedding, retrieve_data, etc.) are child spans
        - LangSmith visualizes the full pipeline execution as a tree
        - Captures total execution time from query to final result
        - Enables end-to-end debugging: "Why did this query fail/succeed?"

        Trace hierarchy in LangSmith:
        RAG Pipeline (this function - ROOT)
        ├── Retrieve Data
        │   └── Get Embedding (query embedding)
        ├── Format Retrieved Context
        ├── Build Prompt
        └── Generate Answer

        Key observability benefits:
        - See which step is slowest (embedding, retrieval, or generation)
        - Analyze token usage across all LLM calls
        - Debug failures: Which step threw an error?
        - Compare traces: Why did query A succeed but query B fail?
        - Track costs: Total tokens across all operations

        Return value metadata (NEW in Video 5):
        - retrieved_context_ids: Enables "show me the products you used"
        - similarity_scores: Debugging retrieval quality (low scores = poor matches)
        - retrieved_context: Transparency into what information influenced answer
        - question: Echo back for validation and logging
    """
    # Initialize Qdrant client (connects to vector database)
    # Use qdrant service name for Docker Compose networking
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    # Step 1: Retrieve k most relevant products based on semantic similarity
    retrieved_context = retrieve_data(question, qdrant_client, top_k)

    # Step 2: Format retrieved data into readable text for the LLM
    preprocessed_context = process_context(retrieved_context)

    # Step 3: Build complete prompt with instructions, context, and question
    prompt = build_prompt(preprocessed_context, question)

    # Step 4: Generate natural language answer using LLM
    answer = generate_answer(prompt)

    # Step 5: Construct final result with answer and retrieved context metadata (NEW in Video 5)
    # Previously: Just returned `answer` string
    # Now: Return structured dict with answer + retrieval metadata
    # Why: Enables observability, debugging, and transparency
    #   - Frontend can display "Products used in this answer"
    #   - Debugging: "Why did this query return irrelevant products?" (check similarity_scores)
    #   - Analytics: Track which products are most frequently retrieved
    #   - Validation: Echo back question to ensure correct processing
    final_result = {
        "answer": answer.answer,  # Natural language response from LLM
        "references": answer.references,  # List of RAGUsedContext objects
        "question": question,  # Original user query (for validation/logging)
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],  # Product ASINs
        "retrieved_context": retrieved_context["retrieved_context"],  # Product descriptions
        "similarity_scores": retrieved_context["similarity_scores"],  # Cosine similarity (0-1)
    }

    return final_result


def rag_pipeline_wrapper(question: str, top_k: int = 5) -> dict:
    """
    Enriches RAG pipeline results with product metadata (images and prices).

    This wrapper function extends the core rag_pipeline by fetching additional
    product metadata from Qdrant for display purposes. It keeps the core RAG
    logic unchanged while adding presentation-layer enhancements.

    Video 3 Enhancement: Adding Rich Product Context
    ------------------------------------------------
    The basic RAG pipeline returns product IDs and descriptions, but for a
    rich frontend experience, we need images and pricing. This wrapper:
    1. Calls the standard rag_pipeline to get the answer + product IDs
    2. For each referenced product, queries Qdrant for full metadata
    3. Returns enriched data suitable for visual product cards in the UI

    Args:
        question (str): User's natural language query
        top_k (int): Number of products to retrieve (default: 5)

    Returns:
        dict: Structured response containing:
            - answer (str): Natural language answer from LLM
            - used_context (list[dict]): Product metadata for each referenced item
                Each item contains:
                - image_url (str | None): Product image URL from Qdrant
                - price (float | None): Product price from Qdrant
                - description (str): Product description from LLM references

    Why a Wrapper Pattern?
    ----------------------
    - Keeps core rag_pipeline focused on retrieval + generation
    - Separates data enrichment (presentation) from core logic
    - Allows testing/using rag_pipeline independently
    - Easy to add more metadata fields (ratings, reviews) later

    Implementation Details:
    -----------------------
    - Uses dummy vector for Qdrant query (we're filtering by ID, not similarity)
    - Queries Qdrant once per referenced product (could batch for optimization)
    - Handles nullable fields (some products may lack images/prices)
    - Docker networking: Uses "qdrant:6333" service name (not localhost)

    Example Response:
    ----------------
    {
        "answer": "The best wireless headphones are...",
        "used_context": [
            {
                "image_url": "https://amazon.com/image.jpg",
                "price": 39.99,
                "description": "TELSOR Wireless Earbuds..."
            },
            ...
        ]
    }

    Performance Considerations:
    --------------------------
    - N+1 query pattern: One Qdrant query per product reference
    - Could batch queries if performance becomes an issue
    - Typically only 2-3 products referenced, so overhead is minimal
    - Consider caching product metadata if same products frequently appear
    """
    # Initialize Qdrant client with Docker service name for inter-container communication
    # Note: Use "qdrant" (service name) not "localhost" when running in Docker Compose
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    # Step 1: Get the base RAG result (answer + product references)
    # This calls the full RAG pipeline: embed → retrieve → generate
    result = rag_pipeline(question, top_k)

    # Step 2: Initialize list to collect enriched product context
    used_context = []

    # Create a dummy embedding vector for Qdrant query
    # We're filtering by product ID (parent_asin), so the vector value doesn't matter
    # But Qdrant's query_points requires a query vector, so we use zeros
    # Shape: (1536,) matches OpenAI text-embedding-3-small dimensionality
    dummy_vector = np.zeros((1536,)).tolist()

    # Step 3: For each product referenced in the LLM's answer, fetch full metadata
    # Note: result["references"] contains only products the LLM actually used
    # (not all retrieved products - the LLM filters for relevance)
    for item in result.get("references", []):
        # Query Qdrant for this specific product using its ASIN (Amazon ID)
        # Filter ensures we get exactly the product we want (not similar products)
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-00",  # Collection created during preprocessing
            query=dummy_vector,  # Required by API but not used (we're filtering by ID)
            limit=1,  # We only want this specific product
            with_payload=True,  # Include all metadata fields (image, price, etc.)
            query_filter=Filter(must=[
                # Exact match filter: Find product with this specific parent_asin
                FieldCondition(
                    key="parent_asin",  # Field name in Qdrant payload
                    match=MatchValue(value=item.id)  # The ASIN from LLM references
                )
            ])
        ).points[0].payload  # Extract the payload dict from the first (only) result

        # Extract presentation metadata from Qdrant payload
        # These fields may be None if data is missing (handled by Optional[] in models)
        image_url = payload.get("image")  # Product image URL (Optional[str])
        price = payload.get("price")  # Product price (Optional[float])

        # Append enriched context to results
        # Combines Qdrant metadata (image, price) with LLM description
        used_context.append({
            "image_url": image_url,  # May be None
            "price": price,  # May be None
            "description": item.description  # From LLM's RAGUsedContext
        })

    # Step 4: Return enriched response for the API endpoint
    # Structure matches what RAGResponse expects in endpoints.py
    return {
        "answer": result["answer"],  # Natural language answer from LLM
        "used_context": used_context  # Enriched product metadata for frontend
    }