#!/usr/bin/env python3
"""
Health Check Script for AI Engineering Bootcamp Application

Verifies that all infrastructure components are running and properly configured:
- Docker containers (api, streamlit-app, qdrant, postgres, items_mcp_server, reviews_mcp_server)
- Network ports (8000, 8501, 6333, 6334, 5433, 8001, 8002)
- Qdrant: hybrid collection used by the API agent/RAG (required), reviews collection (warning)
- Postgres connection (LangGraph checkpointer)
- FastAPI: OpenAPI schema (proves app loaded, not just an open port)

Usage:
    make health              # Full output with details
    make health-silent       # Only show failures
    uv run scripts/health_check.py          # Direct invocation
    uv run scripts/health_check.py --silent # Silent mode
    uv run scripts/health_check.py --strict # Fail if psycopg missing (Postgres not verified)
"""

import argparse
import socket
import subprocess
import sys

try:
    import requests
    from qdrant_client import QdrantClient
except ImportError:
    print("❌ Missing dependencies. Run: uv sync")
    sys.exit(1)

try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a bold header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")


def print_success(text: str):
    """Print success message with checkmark."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_failure(text: str):
    """Print failure message with X."""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    """Non-fatal informational line."""
    print(f"{Colors.BLUE}i{Colors.RESET} {text}")


def check_docker_containers() -> tuple[bool, str]:
    """
    Check if all required Docker containers are running.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse container status
        import json
        containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line]

        # MCP servers (items, reviews) expose tools for agent; required for 04-MCP notebook
        required_services = {
            "api",
            "streamlit-app",
            "qdrant",
            "postgres",
            "items_mcp_server",
            "reviews_mcp_server",
        }
        running_services = {
            container["Service"]
            for container in containers
            if container.get("State") == "running"
        }

        missing_services = required_services - running_services

        if missing_services:
            return False, f"Missing containers: {', '.join(missing_services)}"

        return True, f"All containers running: {', '.join(running_services)}"

    except subprocess.CalledProcessError:
        return False, "Docker Compose not running or not available"
    except Exception as e:
        return False, f"Error checking containers: {e!s}"


def check_port(port: int, service_name: str) -> tuple[bool, str]:
    """
    Check if a port is listening.

    Args:
        port: Port number to check
        service_name: Name of the service for error messages

    Returns:
        Tuple of (success: bool, message: str)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)

    try:
        result = sock.connect_ex(('localhost', port))
        sock.close()

        if result == 0:
            return True, f"{service_name} listening on port {port}"
        else:
            return False, f"{service_name} not listening on port {port}"
    except Exception as e:
        return False, f"Error checking port {port}: {e!s}"


# Collections used by the running API (see apps/api agents/tools.py, retrieval_generation.py)
HYBRID_COLLECTION = "Amazon-items-collection-01-hybrid-search"
REVIEWS_COLLECTION = "Amazon-items-collection-01-reviews"
LEGACY_COLLECTION = "Amazon-items-collection-00"


def check_qdrant_hybrid_collection() -> tuple[bool, str]:
    """Required for agent + RAG retrieval; fail if missing or empty."""
    try:
        client = QdrantClient(url="http://localhost:6333")
        names = [c.name for c in client.get_collections().collections]
        if HYBRID_COLLECTION not in names:
            return (
                False,
                f"'{HYBRID_COLLECTION}' not found (agent/RAG need this). "
                f"Found: {', '.join(sorted(names)) or '(none)'}",
            )
        info = client.get_collection(HYBRID_COLLECTION)
        count = info.points_count
        if count == 0:
            return False, f"'{HYBRID_COLLECTION}' exists but has 0 points — re-run indexing / hybrid notebook"
        return True, f"{HYBRID_COLLECTION}: {count:,} points"
    except Exception as e:
        return False, f"Qdrant hybrid check failed: {e}"


def check_qdrant_reviews_collection() -> tuple[bool, str]:
    """Warning-only: used for review-aware tools / MCP paths."""
    try:
        client = QdrantClient(url="http://localhost:6333")
        names = [c.name for c in client.get_collections().collections]
        if REVIEWS_COLLECTION not in names:
            return False, f"'{REVIEWS_COLLECTION}' not found (review tools may fail)"
        info = client.get_collection(REVIEWS_COLLECTION)
        count = info.points_count
        if count == 0:
            return False, f"'{REVIEWS_COLLECTION}' exists but empty"
        return True, f"{REVIEWS_COLLECTION}: {count:,} points"
    except Exception as e:
        return False, f"Qdrant reviews check failed: {e}"


def check_qdrant_legacy_collection() -> tuple[bool, str]:
    """Informational: older notebooks use collection-00; not required for current API hybrid path."""
    try:
        client = QdrantClient(url="http://localhost:6333")
        names = [c.name for c in client.get_collections().collections]
        if LEGACY_COLLECTION not in names:
            return True, f"{LEGACY_COLLECTION} absent (OK unless you use Week 1 dense-only notebooks)"
        info = client.get_collection(LEGACY_COLLECTION)
        return True, f"{LEGACY_COLLECTION}: {info.points_count:,} points (legacy / notebooks)"
    except Exception as e:
        return True, f"Legacy collection check skipped: {e}"


# Connection string for Postgres (LangGraph checkpointer); host uses port 5433 per docker-compose
_POSTGRES_CONN_STRING = "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"


def check_postgres_connection() -> tuple[bool, str, bool]:
    """
    Check if Postgres (LangGraph checkpointer) is reachable and responds.

    Returns:
        Tuple of (success: bool, message: str, skipped: bool)
    """
    if not HAS_PSYCOPG:
        return True, "Skipped — install psycopg[binary] (dev deps) to verify LangGraph DB", True
    try:
        with psycopg.connect(_POSTGRES_CONN_STRING) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            if cur.fetchone()[0] == 1:
                return True, "Postgres responding (SELECT 1)", False
        return False, "Postgres returned unexpected result", False
    except psycopg.OperationalError as e:
        return False, f"Postgres not reachable: {e}", False
    except Exception as e:
        return False, f"Error checking Postgres: {e}", False


def check_mcp_server(base_url: str, name: str) -> tuple[bool, str]:
    """
    Check if an MCP server (FastMCP) is reachable via HTTP.

    Learning: MCP servers use HTTP transport. GET on root may return 404/405 because
    the MCP protocol uses POST/SSE for tool listing and invocation. Any response
    (200, 404, 405) indicates the server process is running and reachable.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code in (200, 404, 405):
            return True, f"{name} responding (HTTP {response.status_code})"
        return False, f"{name} returned HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {name} (connection refused)"
    except requests.exceptions.Timeout:
        return False, f"{name} health check timed out"
    except Exception as e:
        return False, f"Error checking {name}: {e!s}"


def check_fastapi_openapi() -> tuple[bool, str]:
    """
    Confirm FastAPI app is actually serving (OpenAPI JSON), not just a bound port.

    /health is not registered in this project; /openapi.json is always present on FastAPI.
    """
    try:
        response = requests.get("http://localhost:8000/openapi.json", timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                paths = len(data.get("paths", {}))
                return True, f"FastAPI OpenAPI OK ({paths} path(s) registered)"
            except Exception:
                return True, "FastAPI OpenAPI OK (non-JSON body — unexpected)"
        if response.status_code == 404:
            return (
                False,
                "GET /openapi.json returned 404 — port open but this may not be the FastAPI app",
            )
        return False, f"GET /openapi.json returned HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API (connection refused)"
    except requests.exceptions.Timeout:
        return False, "API OpenAPI check timed out"
    except Exception as e:
        return False, f"Error checking API OpenAPI: {e}"


def print_next_steps(failures: list[str], warnings: list[str], postgres_skipped: bool) -> None:
    """Actionable hints after a run (full mode only)."""
    if not failures and not warnings and not postgres_skipped:
        return
    print_header("Suggested next steps")
    if failures:
        for line in failures:
            print(f"  {Colors.RED}•{Colors.RESET} {line}")
    if warnings:
        for line in warnings:
            print(f"  {Colors.YELLOW}•{Colors.RESET} {line}")
    if postgres_skipped:
        print(
            f"  {Colors.YELLOW}•{Colors.RESET} Postgres: run `uv sync` (includes psycopg) then re-run health"
        )
    print(
        f"  {Colors.BLUE}•{Colors.RESET} Logs: `docker compose logs -f api` "
        f"or `docker compose ps`"
    )
    if any("Qdrant" in f or "hybrid" in f.lower() or "collection" in f.lower() for f in failures):
        print(
            f"  {Colors.BLUE}•{Colors.RESET} Qdrant data: run hybrid indexing notebook "
            f"(collection `{HYBRID_COLLECTION}`)"
        )


def main():
    """Run all health checks."""
    parser = argparse.ArgumentParser(description="Health check for application infrastructure")
    parser.add_argument("--silent", action="store_true", help="Only show failures")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if Postgres cannot be checked (psycopg not installed)",
    )
    args = parser.parse_args()

    silent = args.silent
    next_step_failures: list[str] = []
    next_step_warnings: list[str] = []
    postgres_skipped = False

    if not silent:
        print_header("Infrastructure Health Check")

    all_passed = True

    # Check 1: Docker containers
    success, message = check_docker_containers()
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Docker Containers: {message}")
    if not success:
        next_step_failures.append("Start stack: `make run-docker-compose` (with `op` if using 1Password)")

    # Check 2: API port
    success, message = check_port(8000, "API")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"API Port: {message}")
    if not success:
        next_step_failures.append("Ensure `api` container is running and port 8000 is published")

    # Check 3: Streamlit port
    success, message = check_port(8501, "Streamlit")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Streamlit Port: {message}")

    # Check 4: Qdrant REST port
    success, message = check_port(6333, "Qdrant")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Qdrant Port: {message}")

    # Check 4b: Qdrant gRPC (optional; retrieval uses REST)
    success_grpc, msg_grpc = check_port(6334, "Qdrant gRPC")
    if not silent:
        (print_success if success_grpc else print_warning)(f"Qdrant gRPC Port: {msg_grpc}")

    # Check 5: Qdrant hybrid collection (required for current API)
    success, message = check_qdrant_hybrid_collection()
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Qdrant Hybrid (agent/RAG): {message}")
    if not success:
        next_step_failures.append(
            f"Create/load `{HYBRID_COLLECTION}` (Week 2 hybrid search notebook / preprocessing pipeline)"
        )

    # Check 5b: Reviews collection (warning only)
    rev_ok, rev_msg = check_qdrant_reviews_collection()
    if rev_ok:
        if not silent:
            print_success(f"Qdrant Reviews: {rev_msg}")
    else:
        if not silent:
            print_warning(f"Qdrant Reviews: {rev_msg}")
        next_step_warnings.append(f"Optional: index `{REVIEWS_COLLECTION}` for review-aware tools")

    # Check 5c: Legacy collection (informational)
    _leg_ok, leg_msg = check_qdrant_legacy_collection()
    if not silent:
        print_info(f"Qdrant Legacy: {leg_msg}")

    # Check 6: Postgres connection (LangGraph checkpointer)
    success, message, skipped = check_postgres_connection()
    if skipped:
        postgres_skipped = True
        if args.strict:
            all_passed = False
            if not silent:
                print_failure(f"Postgres: {message} (--strict: treating as failure)")
            next_step_failures.append("Install psycopg: `uv sync` then re-run health")
        else:
            if not silent:
                print_warning(f"Postgres: {message}")
    else:
        all_passed = all_passed and success
        if not silent or not success:
            (print_success if success else print_failure)(f"Postgres: {message}")
        if not success:
            next_step_failures.append(
                "Check `postgres` container and `langgraph_db` on localhost:5433 "
                "(see docker-compose.yml)"
            )

    # Check 7: FastAPI OpenAPI (app actually loaded)
    success, message = check_fastapi_openapi()
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"FastAPI App: {message}")
    if not success:
        next_step_failures.append("Inspect API logs: `docker compose logs -f api`")

    # Check 8-11: MCP servers
    success, message = check_port(8001, "items_mcp_server")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"items_mcp_server Port: {message}")
    if not success:
        next_step_failures.append("Start MCP services from root docker-compose (items_mcp_server)")

    success, message = check_port(8002, "reviews_mcp_server")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"reviews_mcp_server Port: {message}")

    success, message = check_mcp_server("http://localhost:8001", "items_mcp_server")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"items_mcp_server HTTP: {message}")

    success, message = check_mcp_server("http://localhost:8002", "reviews_mcp_server")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"reviews_mcp_server HTTP: {message}")

    # Summary
    if not silent:
        print()
        if all_passed and not next_step_warnings and not postgres_skipped:
            print_success("All required checks passed. Infrastructure looks ready.")
        elif all_passed:
            print_success("Required checks passed (see warnings above).")
        else:
            print_failure("One or more required checks failed.")
        if not silent and (next_step_failures or next_step_warnings or postgres_skipped):
            print_next_steps(next_step_failures, next_step_warnings, postgres_skipped)

    # Silent mode: still print compact next steps on failure
    if silent and not all_passed and (next_step_failures or next_step_warnings):
        for line in next_step_failures:
            print_failure(line)
        for line in next_step_warnings:
            print_warning(line)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
