#!/usr/bin/env python3
"""
Smoke test for PostgreSQL (LangGraph checkpointer database).

Verifies:
- Can connect to Postgres with the same connection string used by the app/notebook
- Database responds to a simple query (SELECT 1)
- Optionally reports whether LangGraph checkpoint tables exist (informational)

Connection: localhost:5433, langgraph_db, langgraph_user (matches docker-compose and week4 notebook).

Usage:
    make smoke-test-postgres
    uv run scripts/smoke_test_postgres.py
"""

import sys

try:
    import psycopg
except ImportError:
    print("❌ Missing psycopg. Run: uv sync (psycopg[binary] in dev dependency-group)")
    sys.exit(1)

# Same connection string as app/notebook (host sees port 5433 via docker-compose)
CONN_STRING = "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"

# ANSI colors (match health_check.py / smoke_test.py)
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
    print(f"{Colors.BLUE}i{Colors.RESET} {text}")


def check_checkpoints_table(conn) -> bool:
    """Return True if LangGraph checkpoints table exists."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            ("public", "checkpoints"),
        )
        return cur.fetchone() is not None


def run_smoke_test() -> bool:
    """Connect, run SELECT 1, optionally report checkpoint table. Return True if smoke test passed."""
    print_header("🧪 Smoke Test: PostgreSQL (LangGraph checkpointer)")
    print_info(f"Connection: {CONN_STRING.split('@')[1] if '@' in CONN_STRING else 'localhost:5433/langgraph_db'}")

    all_passed = True

    # 1. Connect
    try:
        with psycopg.connect(CONN_STRING) as conn:
            print_success("Connected to Postgres")

            # 2. Simple query
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
                if row and row[0] == 1:
                    print_success("Database responds (SELECT 1)")
                else:
                    print_failure("Unexpected result from SELECT 1")
                    all_passed = False

            # 3. Checkpoint table + row count (informational: 0 rows is OK before first agent turn)
            if check_checkpoints_table(conn):
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM checkpoints")
                        n = cur.fetchone()[0]
                    print_success(f"LangGraph checkpoints table: {n:,} row(s)")
                    if n == 0:
                        print_info("No checkpoints yet (expected before first /agent/ conversation with saver)")
                except Exception as ex:
                    print_info(f"Could not COUNT checkpoints: {ex}")
            else:
                print_info("Table `checkpoints` not found — run PostgresSaver.setup() once if using LangGraph persistence")
    except psycopg.OperationalError as e:
        print_failure(f"Cannot connect to Postgres: {e}")
        all_passed = False
    except Exception as e:
        print_failure(f"Error: {e}")
        all_passed = False

    print()
    if all_passed:
        print_success("Postgres smoke test PASSED")
    else:
        print_failure("Postgres smoke test FAILED")
        print_info("Next: `docker compose ps postgres` and `docker compose logs postgres`")

    return all_passed


def main() -> None:
    passed = run_smoke_test()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
