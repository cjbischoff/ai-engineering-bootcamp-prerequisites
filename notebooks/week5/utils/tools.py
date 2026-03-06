"""
Retrieval and cart tools for Week 5 multi-agent shopping assistant (Sprint 4).

Product Q&A tools (used by product_qa_agent):
- get_formatted_items_context: product descriptions (hybrid search on Amazon-items-collection-01-hybrid-search).
- get_formatted_reviews_context: customer reviews scoped by item IDs (Amazon-items-collection-01-reviews).

Shopping cart tools (used by shopping_cart_agent):
- add_to_shopping_cart, get_shopping_cart, remove_from_cart: persist to tools_database.

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
def retrieve_items_data(query, k=5):
    """Retrieve top-k products via hybrid search. Creates Qdrant client internally."""
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://localhost:6333")

    # Qdrant hybrid search (Week 2 Video 5): dense + sparse (BM25) with RRF fusion.
    # Prefetch retrieves from both vectors; FusionQuery merges by rank (scale-independent).
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
def process_items_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
        strict=True,
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context


def get_formatted_items_context(query: str, top_k: int = 5) -> str:
    """
    Tool invoked by agent when it needs product context. Returns formatted string
    of top-k products (ID, rating, description). Agent uses this to answer questions.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    context = retrieve_items_data(query, k=top_k)
    formatted_context = process_items_context(context)
    return formatted_context

# Item Reviews Retrieval Tool (Week 4: second collection for review text)

@traceable(
    name="retrieve_reviews_data",
    run_type="retriever"
)
def retrieve_reviews_data(query, item_list, k=5):
    """Retrieve top-k reviews for given item IDs. Collection stores payload 'text' (review body), not 'description'."""
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://localhost:6333")

    # Prefilter by parent_asin: reviews collection stores (parent_asin, text); we only want
    # reviews for items the agent already retrieved (Week 4 two-stage: items first, then reviews).
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



"""
Add to Shopping Cart tool for Week 5 Shopping Cart Agent.

Learning: Agent tools that persist state require a database. This tool:
- Fetches product metadata (price, image) from Qdrant by parent_asin
- Uses tools_database (bootcamp spec) separate from langgraph_db (checkpointer)
- Upsert logic: INSERT new row or UPDATE quantity if (user_id, cart_id, product_id) exists
- Prefetch + FusionQuery: Qdrant hybrid-search pattern for exact product lookup by filter
"""

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery


def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:
    """Add a list of provided items to the shopping cart.

    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary
            with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.

    Returns:
        A list of the items added to the shopping cart.
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password",
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        for item in items:
            product_id = item["product_id"]
            quantity = item["quantity"]

            qdrant_client = QdrantClient(url="http://localhost:6333")

            # Qdrant lookup: Prefetch with filter by parent_asin (dummy_vector unused; filter does the work)
            # Why Prefetch: hybrid-search collection expects prefetch; filter narrows to single product
            dummy_vector = np.zeros(1536).tolist()
            results = qdrant_client.query_points(
                collection_name="Amazon-items-collection-01-hybrid-search",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id),
                                )
                            ]
                        ),
                        using="text-embedding-3-small",
                        limit=20,
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1,
            )
            # Guard: product_id must exist in Qdrant catalog; else IndexError on points[0]
            if not results.points:
                raise ValueError(f"Product {product_id} not found in catalog")
            payload = results.points[0].payload

            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = "USD"

            # Upsert: check if (user_id, cart_id, product_id) exists; UPDATE quantity or INSERT
            check_query = """
            SELECT id, quantity, price
            FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()

            if existing_item:
                # Update existing item
                new_quantity = existing_item["quantity"] + quantity
                update_query = """
                UPDATE shopping_carts.shopping_cart_items
                SET
                    quantity = %s,
                    price = %s,
                    currency = %s,
                    product_image_url = COALESCE(%s, product_image_url)
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                RETURNING id, quantity, price
                """
                cursor.execute(
                    update_query,
                    (
                        new_quantity,
                        price,
                        currency,
                        product_image_url,
                        user_id,
                        cart_id,
                        product_id,
                    ),
                )
            else:
                # Insert new item
                insert_query = """
                INSERT INTO shopping_carts.shopping_cart_items (
                    user_id, shopping_cart_id, product_id,
                    price, quantity, currency, product_image_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, quantity, price
                """
                cursor.execute(
                    insert_query,
                    (
                        user_id,
                        cart_id,
                        product_id,
                        price,
                        quantity,
                        currency,
                        product_image_url,
                    ),
                )

    conn.close()
    return f"Added {items} to the shopping cart."


def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    """Retrieve all items in a user's shopping cart.

    Learning: Read-only tool; returns list of dicts with product_id, price, quantity,
    total_price (price * quantity). RealDictCursor yields dict-like rows for easy serialization.
    Args:
        user_id: User ID
        cart_id: Cart identifier

    Returns:
        List of dictionaries containing cart items
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password",
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
        SELECT
            product_id, price, quantity,
            currency, product_image_url,
            (price * quantity) as total_price
        FROM shopping_carts.shopping_cart_items
        WHERE user_id = %s AND shopping_cart_id = %s
        ORDER BY added_at DESC
        """
        cursor.execute(query, (user_id, cart_id))
        # Convert RealDictRow to plain dict for JSON-serializable return (agent tool output)
        return [dict(row) for row in cursor.fetchall()]

    conn.close()


def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> bool:
    """Remove an item completely from the shopping cart.

    Learning: DELETE by (user_id, cart_id, product_id). rowcount > 0 indicates success;
    returns False if item wasn't in cart (idempotent for "remove" semantics).
    Args:
        user_id: User ID
        product_id: Product ID to remove
        cart_id: Cart identifier

    Returns:
        True if item was removed, False if item wasn't found
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password",
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
        DELETE FROM shopping_carts.shopping_cart_items
        WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
        """
        cursor.execute(query, (user_id, cart_id, product_id))
        return cursor.rowcount > 0

    conn.close()
