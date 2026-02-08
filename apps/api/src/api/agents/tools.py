"""
Retrieval tool for ReAct agent (Sprint 2 / Video 5-6).

get_formatted_context is the tool the agent calls to retrieve product data.
Hides vector search behind tool use: agent decides when/what to retrieve.
Uses hybrid search (dense + BM25, RRF fusion) - same as retrieval_generation.
"""
import openai
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Document



@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    return response.data[0].embedding


@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query, k=5):
    """Retrieve top-k products via hybrid search. Creates Qdrant client internally."""
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    # Qdrant query (hybrid search with RRF fusion)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(text=query, model="qdrant/bm25"),
                using="bm25",
                limit=20
            ),
        ],
        query=FusionQuery(fusion="rrf"),
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


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"]
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context
def get_formatted_context(query: str, top_k: int = 5) -> str:
    """
    Tool invoked by agent when it needs product context. Returns formatted string
    of top-k products (ID, rating, description). Agent uses this to answer questions.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    context = retrieve_data(query, k=top_k)
    formatted_context = process_context(context)
    return formatted_context
