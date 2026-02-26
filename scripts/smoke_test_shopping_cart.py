#!/usr/bin/env python3
"""
Smoke test for shopping cart database (Week 5 Shopping Cart Agent).

Learning: Run before week5 notebook to ensure tools_database has correct schema.
Uses information_schema to verify schema, table, and columns exist—catches
UndefinedTable errors before add_to_shopping_cart runs. tools_database is
separate from langgraph_db (LangGraph checkpointer) per bootcamp spec.

Verifies:
- Can connect to Postgres (tools_database)
- shopping_carts schema exists
- shopping_cart_items table exists
- Required columns exist (id, user_id, shopping_cart_id, product_id, price, quantity,
  currency, product_image_url, added_at, updated_at)

Connection: localhost:5433, tools_database, langgraph_user (matches bootcamp spec and notebook).

Usage:
    make smoke-test-shopping-cart
    uv run scripts/smoke_test_shopping_cart.py
"""

import sys

try:
    import psycopg
except ImportError:
    print("❌ Missing psycopg. Run: uv sync (psycopg[binary] in dev dependency-group)")
    sys.exit(1)

CONN_STRING = "postgresql://langgraph_user:langgraph_password@localhost:5433/tools_database"

REQUIRED_COLUMNS = frozenset({
    "id",
    "user_id",
    "shopping_cart_id",
    "product_id",
    "price",
    "quantity",
    "currency",
    "product_image_url",
    "added_at",
    "updated_at",
})

# ANSI colors (match health_check.py / smoke_test_postgres.py)
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_failure(text: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def run_smoke_test() -> bool:
    """Verify shopping cart schema, table, and columns. Return True if all passed.

    Checks: SELECT 1, schema existence, table existence, required columns.
    On failure, prints hint to run make setup-shopping-cart.
    """
    print_header("🧪 Smoke Test: Shopping Cart Database (Week 5)")
    print_info(f"Connection: {CONN_STRING.split('@')[1] if '@' in CONN_STRING else 'localhost:5433/tools_database'}")

    all_passed = True

    try:
        with psycopg.connect(CONN_STRING) as conn:
            with conn.cursor() as cur:
                # 1. Database responds
                cur.execute("SELECT 1")
                if cur.fetchone()[0] != 1:
                    print_failure("Database did not respond to SELECT 1")
                    all_passed = False
                else:
                    print_success("Database responding (SELECT 1)")

                # 2. Schema exists
                cur.execute(
                    "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                    ("shopping_carts",),
                )
                if cur.fetchone() is None:
                    print_failure("Schema 'shopping_carts' does not exist")
                    all_passed = False
                else:
                    print_success("Schema 'shopping_carts' exists")

                # 3. Table exists
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    ("shopping_carts", "shopping_cart_items"),
                )
                if cur.fetchone() is None:
                    print_failure("Table 'shopping_carts.shopping_cart_items' does not exist")
                    all_passed = False
                else:
                    print_success("Table 'shopping_carts.shopping_cart_items' exists")

                # 4. Required columns exist
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    ("shopping_carts", "shopping_cart_items"),
                )
                actual_columns = frozenset(row[0] for row in cur.fetchall())
                missing = REQUIRED_COLUMNS - actual_columns
                extra = actual_columns - REQUIRED_COLUMNS

                if missing:
                    print_failure(f"Missing columns: {', '.join(sorted(missing))}")
                    all_passed = False
                else:
                    print_success(f"All required columns exist ({len(REQUIRED_COLUMNS)} columns)")

                if extra and not missing:
                    print_info(f"Additional columns (OK): {', '.join(sorted(extra))}")

    except psycopg.OperationalError as e:
        print_failure(f"Cannot connect to Postgres: {e}")
        all_passed = False
    except Exception as e:
        print_failure(f"Error: {e}")
        all_passed = False

    print()
    if all_passed:
        print_success("✅ Shopping cart smoke test PASSED")
    else:
        print_failure("❌ Shopping cart smoke test FAILED")
        print_info("Run: docker compose exec -T postgres psql -U langgraph_user -d tools_database < scripts/sql/shopping_cart_table.sql")

    return all_passed


def main() -> None:
    passed = run_smoke_test()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
