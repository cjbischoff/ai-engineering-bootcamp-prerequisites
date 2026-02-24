#!/usr/bin/env python3
"""
Smoke test for MCP servers (items_mcp_server, reviews_mcp_server).

Learning: MCP servers run as separate HTTP services. This script verifies they are up before
running the 04-MCP notebook or any agent that depends on them. Mirrors smoke_test.py pattern
for the main RAG API.

Verifies:
- Ports 8001 and 8002 are listening (items and reviews MCP servers)
- Each server responds to HTTP GET; accepts 200, 404, or 405 (MCP endpoints may only accept POST/SSE)

Usage:
    make smoke-test-mcp
    uv run scripts/smoke_test_mcp.py
"""
import socket
import sys
from typing import Tuple

try:
    import requests
except ImportError:
    print("❌ Missing requests. Run: uv sync")
    sys.exit(1)

# Host ports: docker-compose maps 8001->items_mcp_server:8000, 8002->reviews_mcp_server:8000
ITEMS_MCP_PORT = 8001
REVIEWS_MCP_PORT = 8002
HTTP_TIMEOUT = 5

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
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def check_port(port: int, service_name: str) -> Tuple[bool, str]:
    """Return (success, message) for port listening on localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:
            return True, f"{service_name} listening on port {port}"
        return False, f"{service_name} not listening on port {port}"
    except Exception as e:
        return False, f"Error checking port {port}: {e}"


def check_mcp_http(base_url: str, name: str) -> Tuple[bool, str]:
    """GET base_url; accept 200, 404, 405 as server up. Return (success, message).
    MCP endpoints often only accept POST/SSE; 404 or 405 on GET still means the server is running."""
    try:
        response = requests.get(f"{base_url}/", timeout=HTTP_TIMEOUT)
        if response.status_code in (200, 404, 405):
            return True, f"{name} responding (HTTP {response.status_code})"
        return False, f"{name} returned HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {name} (connection refused)"
    except requests.exceptions.Timeout:
        return False, f"{name} request timed out"
    except Exception as e:
        return False, f"Error checking {name}: {e}"


def run_smoke_test() -> bool:
    """Run MCP server smoke tests. Return True if all passed."""
    print_header("🧪 Smoke Test: MCP Servers (items_mcp_server, reviews_mcp_server)")
    print_info(f"Ports: items={ITEMS_MCP_PORT}, reviews={REVIEWS_MCP_PORT}")

    all_passed = True

    # items_mcp_server: port then HTTP
    success, msg = check_port(ITEMS_MCP_PORT, "items_mcp_server")
    all_passed = all_passed and success
    (print_success if success else print_failure)(msg)

    success, msg = check_mcp_http("http://localhost:8001", "items_mcp_server")
    all_passed = all_passed and success
    (print_success if success else print_failure)(msg)

    # reviews_mcp_server: port then HTTP
    success, msg = check_port(REVIEWS_MCP_PORT, "reviews_mcp_server")
    all_passed = all_passed and success
    (print_success if success else print_failure)(msg)

    success, msg = check_mcp_http("http://localhost:8002", "reviews_mcp_server")
    all_passed = all_passed and success
    (print_success if success else print_failure)(msg)

    print()
    if all_passed:
        print_success("✅ MCP smoke test PASSED")
    else:
        print_failure("❌ MCP smoke test FAILED")

    return all_passed


def main() -> None:
    passed = run_smoke_test()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
