# Week 5: Shopping Cart Agent Tools

Agent tools that persist state to PostgreSQL (tools_database). Builds on Week 4 multi-turn patterns.

## What's in This Week

- **01-Shopping-Cart-Agent-Tools.ipynb**: Three tools—add, get, remove—for a shopping cart backed by `shopping_carts.shopping_cart_items` in `tools_database`.
- **02-Shopping-Cart-Agent.ipynb**: Multi-agent workflow with intent router (product_qa | shopping_cart | other), product_qa_agent (retrieval tools), and shopping_cart_agent (cart tools). Uses PostgresSaver for state persistence.

## Key Concepts

### Database Separation
- **tools_database**: Shopping cart (Week 5). Run `make setup-shopping-cart` before the notebook.
- **langgraph_db**: LangGraph checkpointer (Week 4). Do not mix shopping cart data here.

### Tools
1. **add_to_shopping_cart(items, user_id, cart_id)**: Upsert items; fetches price/image from Qdrant by `parent_asin`.
2. **get_shopping_cart(user_id, cart_id)**: Returns cart items with `total_price` (price × quantity).
3. **remove_from_cart(product_id, user_id, cart_id)**: DELETE by product; returns True if removed.

### Qdrant Lookup
Uses Prefetch + Filter on `Amazon-items-collection-01-hybrid-search` to fetch product metadata by `parent_asin`. Guard: raises `ValueError` if product not in catalog.

## Prerequisites

```bash
make setup-shopping-cart   # Create tools_database + schema
make smoke-test-shopping-cart  # Verify schema/table/columns
```

## Files

| File | Purpose |
|------|---------|
| `01-Shopping-Cart-Agent-Tools.ipynb` | Notebook with add/get/remove tools |
| `02-Shopping-Cart-Agent.ipynb` | Multi-agent workflow: intent router + product_qa + shopping_cart |
| `utils/utils.py` | format_ai_message, get_tool_descriptions (shared with Week 4) |
| `utils/tools.py` | get_formatted_items_context, get_formatted_reviews_context, get_shopping_cart, add_to_shopping_cart, remove_from_cart |

## Related

- Schema: `scripts/sql/shopping_cart_table.sql`
- Smoke test: `scripts/smoke_test_shopping_cart.py`
