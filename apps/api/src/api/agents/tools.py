"""
Agent tools for the LangGraph multi-agent workflow (Sprint 2 / Video 5-6; Week 4-5).

Product QA agent tools (retrieval):
- get_formatted_items_context: Hybrid search on Amazon-items-collection-01-hybrid-search.
  Returns formatted product descriptions (ID, rating, description) for the agent to answer questions.
- get_formatted_reviews_context: Customer reviews from Amazon-items-collection-01-reviews,
  scoped by item IDs. Two-stage: agent retrieves items first, then reviews for those items.

Shopping cart agent tools (persistence in ``tools_database`` / schema ``shopping_carts``):
- add_to_shopping_cart: Upsert items; fetches price/image from Qdrant by parent_asin.
- get_shopping_cart: Returns cart items with total_price (price × quantity).
- remove_from_cart: Delete item by product_id.

Warehouse manager tools (same Postgres **host** as cart—``tools_database``, schema ``warehouses``):
- check_warehouse_availability: Read ``warehouses.inventory``; classify full vs partial fulfillment.
- reserve_warehouse_items: Transactionally increment ``reserved_quantity`` (``available_quantity``
  is generated as ``total_quantity - reserved_quantity``). See ``scripts/sql/warehouse_management.sql``.

**Docker networking:** Connections use ``host="postgres"``, ``port=5432`` (service on the compose
network). From the **host machine**, ``psql`` often uses ``localhost:5433`` mapped to that port.

**Logging:** INFO lines around tool entry/exit help correlate API logs with LangSmith spans.

All tools are ``@traceable`` for LangSmith. The LLM chooses when to invoke them (ReAct pattern).
"""
import logging

import openai
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Document, Filter, FieldCondition, MatchAny, MatchValue
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

# Standard library logger name -> appears as ``api.agents.tools`` in uvicorn/docker output.
logger = logging.getLogger(__name__)

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
    qdrant_client = QdrantClient(url="http://qdrant:6333")

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
    logger.info(
        "tool get_formatted_items_context top_k=%s query_len=%s",
        top_k,
        len(query),
    )
    context = retrieve_items_data(query, top_k)
    formatted_context = process_items_context(context)
    logger.info(
        "tool get_formatted_items_context done chunks=%s context_chars=%s",
        len(context.get("retrieved_context", [])),
        len(formatted_context),
    )
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
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory
    """
    logger.info(
        "tool get_formatted_reviews_context top_k=%s item_list_len=%s query_len=%s",
        top_k,
        len(item_list),
        len(query),
    )
    context = retrieve_reviews_data(query, item_list, top_k)
    formatted_context = process_reviews_context(context)
    logger.info(
        "tool get_formatted_reviews_context done chunks=%s context_chars=%s",
        len(context.get("retrieved_context", [])),
        len(formatted_context),
    )
    return formatted_context



def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:
    """Add a list of provided items to the shopping cart.

    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.

    Returns:
        A confirmation message listing the items added to the shopping cart.
    """
    logger.info(
        "tool add_to_shopping_cart n_items=%s user_id=%s cart_id=%s",
        len(items),
        user_id,
        cart_id,
    )
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password",
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        for item in items:
            product_id = item["product_id"]
            quantity = item["quantity"]

            qdrant_client = QdrantClient(url="http://qdrant:6333")

            # Qdrant lookup: Prefetch with filter by parent_asin (dummy_vector unused; filter does the work)
            # Why Prefetch: hybrid-search collection expects prefetch; filter narrows to single product
            dummy_vector = np.zeros(1536).tolist()
            payload = qdrant_client.query_points(
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
            ).points[0].payload

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
    msg = f"Added {items} to the shopping cart."
    logger.info("tool add_to_shopping_cart done msg_chars=%s", len(msg))
    return msg


def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    """Retrieve all items in a user's shopping cart.

    Args:
        user_id: User ID.
        cart_id: Cart identifier.

    Returns:
        List of dictionaries containing cart items.
    """
    logger.info("tool get_shopping_cart user_id=%s cart_id=%s", user_id, cart_id)
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
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
        rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    logger.info("tool get_shopping_cart done rows=%s", len(rows))
    return rows


def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> bool:
    """Remove an item completely from the shopping cart.

    Args:
        product_id: Product ID to remove.
        user_id: User ID.
        cart_id: Cart identifier.

    Returns:
        True if item was removed, False if item wasn't found.
    """
    logger.info(
        "tool remove_from_cart product_id=%s user_id=%s cart_id=%s",
        product_id,
        user_id,
        cart_id,
    )
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
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
        removed = cursor.rowcount > 0
    conn.close()
    logger.info("tool remove_from_cart done removed=%s", removed)
    return removed


def check_warehouse_availability(items: list[dict]) -> dict:
    """Check availability of items across warehouses, including partial fulfillment options.

    Queries warehouses.inventory per warehouse and per item. Categorizes warehouses as
    full (can fulfill all items) or partial (some stock). Tracks unavailable items by
    summing available_quantity across all warehouses.

    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """
    logger.info("tool check_warehouse_availability n_items=%s", len(items))
    # Same DB as shopping cart (warehouses schema); from api container use service postgres:5432 (5433 is host-only)
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password",
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": []
            }

            # Check each warehouse for availability
            warehouse_query = """
                SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
                FROM warehouses.inventory
            """
            cursor.execute(warehouse_query)
            warehouses = cursor.fetchall()

            for warehouse in warehouses:
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details = {
                    "warehouse_id": warehouse['warehouse_id'],
                    "warehouse_name": warehouse['warehouse_name'],
                    "warehouse_location": warehouse['warehouse_location'],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False
                }

                for item in items:
                    product_id = item['product_id']
                    requested_quantity = item['quantity']

                    # Check availability in this warehouse
                    availability_query = """
                        SELECT product_id, total_quantity, reserved_quantity, available_quantity
                        FROM warehouses.inventory
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(availability_query, (warehouse['warehouse_id'], product_id))
                    inventory = cursor.fetchone()

                    available_qty = inventory['available_quantity'] if inventory else 0

                    item_detail = {
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "available": available_qty,
                        "can_fulfill_completely": available_qty >= requested_quantity,
                        "can_fulfill_partially": available_qty > 0 and available_qty < requested_quantity
                    }
                    warehouse_details["items"].append(item_detail)

                    # Track if warehouse can fulfill this item completely
                    if available_qty < requested_quantity:
                        warehouse_can_fulfill_all = False

                    # Track if warehouse has any availability for any item
                    if available_qty > 0:
                        has_any_availability = True

                # Categorize warehouse
                if warehouse_can_fulfill_all:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })
                elif has_any_availability:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })

                result["details"].append(warehouse_details)

            # Check if any items cannot be fulfilled from any warehouse
            for item in items:
                product_id = item['product_id']
                requested_quantity = item['quantity']

                # Get total available quantity across all warehouses
                total_available_query = """
                    SELECT product_id, SUM(available_quantity) as total_available
                    FROM warehouses.inventory
                    WHERE product_id = %s
                    GROUP BY product_id
                """
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()

                total_available_qty = total_available['total_available'] if total_available else 0

                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append({
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "total_available_across_warehouses": total_available_qty,
                        "shortage": requested_quantity - total_available_qty
                    })

            result["can_fulfill_completely"] = (
                len(result["warehouses_full_fulfillment"]) > 0
                and len(result["unavailable_items"]) == 0
            )

            logger.info(
                "tool check_warehouse_availability done can_fulfill_completely=%s "
                "full_warehouses=%s partial_warehouses=%s unavailable_items=%s",
                result["can_fulfill_completely"],
                len(result["warehouses_full_fulfillment"]),
                len(result["warehouses_partial_fulfillment"]),
                len(result["unavailable_items"]),
            )
            return result

    finally:
        conn.close()


def reserve_warehouse_items(reservations: list[dict]) -> dict:
    """Reserve items from multiple warehouses in a single transaction.

    Uses SELECT ... FOR UPDATE to lock inventory rows, then increments reserved_quantity.
    available_quantity is a GENERATED column (total - reserved), so we only update reserved.
    Commits only if all reservations succeed; otherwise rolls back.

    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
            - warehouse_id: The warehouse to reserve from
            - product_id: The product to reserve
            - quantity: The quantity to reserve

    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """
    logger.info("tool reserve_warehouse_items n_reservations=%s", len(reservations))
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password",
    )
    conn.autocommit = False  # Manual commit/rollback for atomic all-or-nothing behavior

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "success": False,
                "reserved_items": [],
                "failed_items": []
            }

            for reservation in reservations:
                warehouse_id = reservation['warehouse_id']
                product_id = reservation['product_id']
                quantity = reservation['quantity']

                # FOR UPDATE locks the row until commit/rollback; prevents concurrent over-reservation
                check_query = """
                    SELECT warehouse_id, product_id, warehouse_name, warehouse_location,
                           total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    FOR UPDATE
                """
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()

                if inventory and inventory['available_quantity'] >= quantity:
                    # Update inventory to reserve the items
                    update_query = """
                        UPDATE warehouses.inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))
                    result["reserved_items"].append({
                        "product_id": product_id,
                        "quantity": quantity,
                        "warehouse_id": warehouse_id,
                        "warehouse_name": inventory['warehouse_name'],
                        "warehouse_location": inventory['warehouse_location']
                    })
                else:
                    result["failed_items"].append({
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "requested": quantity,
                        "available": inventory['available_quantity'] if inventory else 0,
                        "reason": "insufficient_stock" if inventory else "not_in_warehouse"
                    })

            # Only commit if all items were successfully reserved
            if len(result["failed_items"]) == 0:
                conn.commit()
                result["success"] = True
            else:
                conn.rollback()

        logger.info(
            "tool reserve_warehouse_items done success=%s n_reserved=%s n_failed=%s",
            result["success"],
            len(result["reserved_items"]),
            len(result["failed_items"]),
        )
        return result

    except Exception as e:
        logger.warning(
            "tool reserve_warehouse_items failed host=%s port=%s db=%s err=%s",
            "postgres",
            5432,
            "tools_database",
            e,
            exc_info=True,
        )
        conn.rollback()
        raise e
    finally:
        conn.close()
