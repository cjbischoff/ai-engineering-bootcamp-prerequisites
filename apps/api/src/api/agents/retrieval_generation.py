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
"""

import openai
from qdrant_client import QdrantClient


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
    """
    response = openai.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


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
        retrieved_context.append(result.payload["description"])  # Combined title + features
        retrieved_context_ratings.append(result.payload["average_rating"])  # Customer rating
        similarity_scores.append(result.score)  # How similar to query (0-1)

    # Return structured data for downstream processing
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


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
             "- ID: {asin}, rating: {rating}, description: {description}\n"

    Why this format:
        - Bullet points help LLM parse separate products
        - Including ID allows user to reference specific products
        - Rating helps LLM prioritize highly-rated items
        - Description provides the actual content for answering

    Example output:
        - ID: B09X12ABC, rating: 4.5, description: Wireless gaming headset...
        - ID: B09Y34DEF, rating: 4.2, description: USB microphone for streaming...
    """
    formatted_context = ""

    # Zip iterates through three lists in parallel (product by product)
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


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
    """
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
    """
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "system", "content": prompt}],
        reasoning_effort="minimal",
    )

    return response.choices[0].message.content


def rag_pipeline(question, top_k=5):
    """
    Complete RAG pipeline: Retrieve relevant products and generate an answer.

    This orchestrates the full RAG workflow from query to answer.
    Called by the FastAPI endpoint when a user submits a question.

    Args:
        question (str): User's question about products
        top_k (int): Number of products to retrieve. Default is 5

    Returns:
        str: Natural language answer generated by the LLM

    Pipeline stages:
        1. RETRIEVE: Query Qdrant for k similar products (semantic search)
        2. PREPROCESS: Format products into readable text
        3. AUGMENT: Build prompt with context and question
        4. GENERATE: LLM produces final answer

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
    """
    # Initialize Qdrant client (connects to vector database)
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    # Step 1: Retrieve k most relevant products based on semantic similarity
    retrieved_context = retrieve_data(question, qdrant_client, top_k)

    # Step 2: Format retrieved data into readable text for the LLM
    preprocessed_context = process_context(retrieved_context)

    # Step 3: Build complete prompt with instructions, context, and question
    prompt = build_prompt(preprocessed_context, question)

    # Step 4: Generate natural language answer using LLM
    answer = generate_answer(prompt)

    return answer
