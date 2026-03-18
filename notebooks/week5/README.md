# Week 5: Shopping Cart Agent Tools

Agent tools that persist state to PostgreSQL (tools_database). Builds on Week 4 multi-turn patterns.

## What's in This Week

- **01-Shopping-Cart-Agent-Tools.ipynb**: Three tools—add, get, remove—for a shopping cart backed by `shopping_carts.shopping_cart_items` in `tools_database`.
- **02-Shopping-Cart-Agent.ipynb**: Multi-agent workflow with intent router (product_qa | shopping_cart | other), product_qa_agent (retrieval tools), and shopping_cart_agent (cart tools). Uses PostgresSaver for state persistence.
- **03-Coordinator-Agent.ipynb**: Coordinator-based multi-agent workflow. Entry point is coordinator_agent (not intent_router); routes to product_qa_agent or shopping_cart_agent and loops back until done.
- **04-Warehouse-Agent-Database.ipynb**: Prepares synthetic warehouse inventory for the warehouse agent. Fetches product IDs from Qdrant, generates availability per warehouse, and bulk-inserts into `warehouses.inventory`. Run `scripts/sql/warehouse_management.sql` first.

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
| `03-Coordinator-Agent.ipynb` | Coordinator-based workflow: coordinator_agent routes to product_qa/shopping_cart |
| `04-Warehouse-Agent-Database.ipynb` | Synthetic inventory generation and bulk insert into warehouses.inventory |
| `utils/utils.py` | format_ai_message, get_tool_descriptions (shared with Week 4) |
| `utils/tools.py` | get_formatted_items_context, get_formatted_reviews_context, get_shopping_cart, add_to_shopping_cart, remove_from_cart |

## Troubleshooting

### LangSmith: "keys must be str, int, float, bool or None, not function"

**Affected:** `03-Coordinator-Agent.ipynb` only. `02-Shopping-Cart-Agent.ipynb` does not show this error.

**Cause:** The error occurs in `LangChainTracer.on_chain_end` when the tracer serializes the graph state for LangSmith. It is triggered by the **coordinator_agent** node. 02-Shopping-Cart-Agent does not use coordinator_agent (it uses intent_router as entry point), so the tracer never hits the problematic path. In 03-Coordinator-Agent, coordinator_agent is the entry point and runs repeatedly; when its state (with `CoordinatorAgentProperties` and `Annotated[List, add]` reducers) is serialized, a function (the `add` reducer) ends up in a structure that JSON cannot serialize.

**Workarounds:**
1. Disable LangSmith tracing for the run: `LANGSMITH_TRACING=false` (or equivalent).
2. Upgrade LangSmith/LangChain packages in case a fix exists in newer versions.
3. Report the issue upstream if it persists after upgrading.

### coordinator_agent_edge: AttributeError

**Error:** `'State' object has no attribute 'coordinator_agent_edge'` or `'CoordinatorAgentProperties' object has no attribute 'final_answer'`.

**Cause:** Wrong state field name or missing field on `CoordinatorAgentProperties`.

**Fixes:**
- Use `state.coordinator_agent` (not `state.coordinator_agent_edge`) for iteration/plan/next_agent. `coordinator_agent_edge` is the edge function name, not a state field.
- Add `final_answer: bool = False` to `CoordinatorAgentProperties` if the coordinator returns it.

### Empty `result["answer"]` when using `stream()`

**Symptom:** `print(result["answer"])` prints nothing after streaming.

**Cause:** `stream(stream_mode=["values"])` yields state after each node. If the stream stops early (e.g. after product_qa before tools run, or if the tool node fails), the last chunk has `answer: ""`.

**Fix:** Use `graph.invoke()` when you need the final answer. It runs the full graph to completion and returns the final state with `answer` populated.

## Related

- Schema: `scripts/sql/shopping_cart_table.sql`
- Warehouse schema: `scripts/sql/warehouse_management.sql` (warehouses.inventory)
- Smoke test: `scripts/smoke_test_shopping_cart.py`
