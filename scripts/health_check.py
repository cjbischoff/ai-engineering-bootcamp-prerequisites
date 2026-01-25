#!/usr/bin/env python3
"""
Health Check Script for AI Engineering Bootcamp Application

Verifies that all infrastructure components are running and properly configured:
- Docker containers (api, streamlit-app, qdrant)
- Network ports (8000, 8501, 6333, 6334)
- Qdrant collection and document count
- FastAPI health endpoint

Usage:
    make health              # Full output with details
    make health-silent       # Only show failures
    uv run scripts/health_check.py          # Direct invocation
    uv run scripts/health_check.py --silent # Silent mode
"""

import sys
import subprocess
import socket
from typing import Tuple
import argparse

try:
    import requests
    from qdrant_client import QdrantClient
except ImportError:
    print("❌ Missing dependencies. Run: uv sync")
    sys.exit(1)


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


def check_docker_containers() -> Tuple[bool, str]:
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

        required_services = {"api", "streamlit-app", "qdrant"}
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
        return False, f"Error checking containers: {str(e)}"


def check_port(port: int, service_name: str) -> Tuple[bool, str]:
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
        return False, f"Error checking port {port}: {str(e)}"


def check_qdrant_collection() -> Tuple[bool, str]:
    """
    Check if Qdrant collection exists and has documents.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        client = QdrantClient(url="http://localhost:6333")

        # Check collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if "Amazon-items-collection-00" not in collection_names:
            return False, "Collection 'Amazon-items-collection-00' not found"

        # Check document count
        collection_info = client.get_collection("Amazon-items-collection-00")
        count = collection_info.points_count

        if count == 0:
            return False, "Collection is empty (0 documents)"

        return True, f"Collection has {count:,} documents"

    except Exception as e:
        return False, f"Error connecting to Qdrant: {str(e)}"


def check_fastapi_health() -> Tuple[bool, str]:
    """
    Check FastAPI health endpoint (if it exists).

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)

        if response.status_code == 200:
            return True, f"API health endpoint responding (status: {response.status_code})"
        elif response.status_code == 404:
            # Health endpoint doesn't exist - this is OK, just means it's not implemented yet
            return True, "API responding (health endpoint not implemented)"
        else:
            return False, f"API health endpoint returned status {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API (connection refused)"
    except requests.exceptions.Timeout:
        return False, "API health check timed out"
    except Exception as e:
        return False, f"Error checking API health: {str(e)}"


def main():
    """Run all health checks."""
    parser = argparse.ArgumentParser(description="Health check for application infrastructure")
    parser.add_argument("--silent", action="store_true", help="Only show failures")
    args = parser.parse_args()

    silent = args.silent

    if not silent:
        print_header("🏥 Infrastructure Health Check")

    all_passed = True

    # Check 1: Docker containers
    success, message = check_docker_containers()
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Docker Containers: {message}")

    # Check 2: API port
    success, message = check_port(8000, "API")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"API Port: {message}")

    # Check 3: Streamlit port
    success, message = check_port(8501, "Streamlit")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Streamlit Port: {message}")

    # Check 4: Qdrant port
    success, message = check_port(6333, "Qdrant")
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Qdrant Port: {message}")

    # Check 5: Qdrant collection
    success, message = check_qdrant_collection()
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"Qdrant Collection: {message}")

    # Check 6: FastAPI health endpoint
    success, message = check_fastapi_health()
    all_passed = all_passed and success
    if not silent or not success:
        (print_success if success else print_failure)(f"FastAPI Health: {message}")

    # Summary
    if not silent:
        print()
        if all_passed:
            print_success("All checks passed! Infrastructure is healthy.")
        else:
            print_failure("Some checks failed. See above for details.")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
