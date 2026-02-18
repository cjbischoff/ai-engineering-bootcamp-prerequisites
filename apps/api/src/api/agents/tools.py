"""
Retrieval tools for ReAct agent (Sprint 2 / Video 5-6; Week 4 multiple tools).

get_formatted_context: product descriptions (hybrid search on Amazon-items-collection-01-hybrid-search).
get_formatted_reviews_context: customer reviews scoped by item IDs (Amazon-items-collection-01-reviews).
Hides vector search behind tool use: agent decides when/what to retrieve.
"""
import openai
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Document, Filter, FieldCondition, MatchAny

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    # LangSmith: record token usage on the current run for observability/cost tracking.
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding




# Item Description Retrieval Tool

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
        context["retrieved_context_ratings"],
        strict=True,
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

# Item Reviews Retrieval Tool (Week 4: second collection for review text)

@traceable(
    name="retrieve_reviews_data",
    run_type="retriever"
)
def retrieve_reviews_data(query, item_list, k=5):
    """Retrieve top-k reviews for given item IDs. Collection stores payload 'text' (review body), not 'description'."""
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    # Prefilter by parent_asin so we only search within reviews for the items the agent cares about.
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-reviews",
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=item_list
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    # Reviews collection payload uses "text" for review content (see notebook 02-Multiple-Tools).
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["text"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


# Reviews context has no ratings; use this formatter (not process_context which expects retrieved_context_ratings).
@traceable(
    name="format_retrieved_reviews_context",
    run_type="prompt"
)
def process_reviews_context(context):
    formatted_context = ""
    for id, chunk in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        strict=True,
    ):
        formatted_context += f"- ID: {id}, review: {chunk}\n"
    return formatted_context


def get_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str:
    """
    Get the top k reviews matching a query for a list of prefiltered items.
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """
    context = retrieve_reviews_data(query, item_list, k=top_k)
    # Must use process_reviews_context: retrieve_reviews_data returns no retrieved_context_ratings.
    formatted_context = process_reviews_context(context)
    return formatted_context
