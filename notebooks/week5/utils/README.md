# Week 5 Utils

Shared utilities for Week 5 multi-agent notebooks (Sprint 4).

## Files

| File | Purpose |
|------|---------|
| `tools.py` | Product retrieval, shopping cart, and warehouse tools used by product_qa_agent, shopping_cart_agent, warehouse_manager_agent |
| `utils.py` | format_ai_message, get_tool_descriptions (shared with Week 4) |

## tools.py

**Product Q&A tools** (product_qa_agent):
- `get_formatted_items_context`: Hybrid search on Amazon catalog; returns formatted product descriptions.
- `get_formatted_reviews_context`: Reviews scoped by item IDs; two-stage retrieval (items first, then reviews).

**Shopping cart tools** (shopping_cart_agent):
- `add_to_shopping_cart`: Upsert to tools_database; fetches price/image from Qdrant.
- `get_shopping_cart`: Read cart with total_price.
- `remove_from_cart`: DELETE by product.

**Warehouse tools** (warehouse_manager_agent):
- `check_warehouse_availability`: Query warehouses.inventory; returns full/partial fulfillment.
- `reserve_warehouse_items`: Transactional reservation with FOR UPDATE; all-or-nothing.

## utils.py

- `format_ai_message`: Converts AgentResponse to AIMessage for LangGraph.
- `get_tool_descriptions`: Extracts tool metadata from function source for agent prompts.

## Path Setup

Notebooks add `notebooks/week5` to `sys.path` so `from utils.tools import ...` works from project root or `notebooks/week5`.
