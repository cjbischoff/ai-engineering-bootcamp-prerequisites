# SQL Scripts

Schema and setup scripts for PostgreSQL databases.

## Files

| File | Purpose |
|------|---------|
| `shopping_cart_table.sql` | Creates `shopping_carts` schema and `shopping_cart_items` table (Week 5) |

## Usage

**Shopping cart (tools_database):**
```bash
make setup-shopping-cart
# Or manually:
docker compose exec -T postgres psql -U langgraph_user -d tools_database < scripts/sql/shopping_cart_table.sql
```

## shopping_cart_table.sql

Creates:
- Schema: `shopping_carts`
- Table: `shopping_cart_items` with columns id, user_id, shopping_cart_id, product_id, price, quantity, currency, product_image_url, added_at, updated_at
- Constraints: positive_price, positive_quantity, unique (user_id, shopping_cart_id, product_id)
- Trigger: `updated_at` auto-update on UPDATE
- Indexes: user_cart, user_id, product_id

Run against `tools_database` (bootcamp spec).
