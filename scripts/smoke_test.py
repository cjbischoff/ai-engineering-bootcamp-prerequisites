#!/usr/bin/env python3
"""
Smoke Test Script for AI Engineering Bootcamp RAG Pipeline

Runs end-to-end test of the RAG pipeline to verify:
- API endpoint responds correctly
- Response structure matches Pydantic models
- Product context includes required fields
- Response time is acceptable

Usage:
    make smoke-test          # Run with summary output
    make smoke-test-verbose  # Show full JSON response
    uv run scripts/smoke_test.py            # Direct invocation
    uv run scripts/smoke_test.py --verbose  # Verbose mode
"""

import sys
import time
import json
import argparse
from typing import Tuple, Dict, Any

try:
    import requests
except ImportError:
    print("❌ Missing dependencies. Run: uv sync")
    sys.exit(1)


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
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


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ{Colors.RESET} {text}")


def print_json(data: Dict[Any, Any]):
    """Pretty-print JSON data."""
    print(json.dumps(data, indent=2))


def validate_response_structure(response_data: Dict[Any, Any]) -> Tuple[bool, str]:
    """
    Validate that response matches RAGResponse Pydantic model structure.

    Args:
        response_data: Parsed JSON response from API

    Returns:
        Tuple of (valid: bool, message: str)
    """
    # Check top-level fields
    required_fields = {"request_id", "answer", "used_context"}
    missing_fields = required_fields - set(response_data.keys())

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    # Check types
    if not isinstance(response_data["request_id"], str):
        return False, "request_id must be a string"

    if not isinstance(response_data["answer"], str):
        return False, "answer must be a string"

    if not isinstance(response_data["used_context"], list):
        return False, "used_context must be a list"

    # Validate each context item
    for idx, item in enumerate(response_data["used_context"]):
        if not isinstance(item, dict):
            return False, f"used_context[{idx}] must be a dict"

        # Check for description (required)
        if "description" not in item:
            return False, f"used_context[{idx}] missing 'description' field"

        # image_url and price are optional, but if present, check types
        if "image_url" in item and item["image_url"] is not None:
            if not isinstance(item["image_url"], str):
                return False, f"used_context[{idx}].image_url must be string or null"

        if "price" in item and item["price"] is not None:
            if not isinstance(item["price"], (int, float)):
                return False, f"used_context[{idx}].price must be number or null"

    return True, f"Valid structure with {len(response_data['used_context'])} products"


def run_smoke_test(query: str, verbose: bool = False) -> bool:
    """
    Run a single smoke test query against the RAG endpoint.

    Args:
        query: Test query to send
        verbose: Whether to print full response JSON

    Returns:
        bool: True if test passed, False otherwise
    """
    print_header(f"🧪 Smoke Test: RAG Pipeline")
    print_info(f"Query: {query}")

    all_passed = True

    # Test 1: API responds
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/rag/",
            json={"query": query},
            timeout=30
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            print_success(f"API responded with status 200 in {elapsed:.2f}s")
        else:
            print_failure(f"API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print_failure("Cannot connect to API (is it running?)")
        return False
    except requests.exceptions.Timeout:
        print_failure("Request timed out (> 30 seconds)")
        return False
    except Exception as e:
        print_failure(f"Error making request: {str(e)}")
        return False

    # Test 2: Response is valid JSON
    try:
        response_data = response.json()
        print_success("Response is valid JSON")
    except Exception as e:
        print_failure(f"Response is not valid JSON: {str(e)}")
        return False

    # Test 3: Response structure matches Pydantic model
    valid, message = validate_response_structure(response_data)
    if valid:
        print_success(f"Response structure valid: {message}")
    else:
        print_failure(f"Response structure invalid: {message}")
        all_passed = False

    # Test 4: Response time acceptable
    # Note: First query may be slower due to model initialization (cold start)
    # We use 20s threshold to account for embedding + retrieval + LLM generation
    if elapsed < 20.0:
        print_success(f"Response time acceptable: {elapsed:.2f}s < 20.0s")
    else:
        print_failure(f"Response time slow: {elapsed:.2f}s >= 20.0s")
        all_passed = False

    # Test 5: Answer is non-empty
    if len(response_data.get("answer", "")) > 0:
        print_success(f"Answer generated ({len(response_data['answer'])} chars)")
    else:
        print_failure("Answer is empty")
        all_passed = False

    # Test 6: At least one product in context
    context_count = len(response_data.get("used_context", []))
    if context_count > 0:
        print_success(f"Products in context: {context_count}")
    else:
        print_failure("No products in used_context")
        all_passed = False

    # Print response details
    if verbose:
        print_header("📄 Full Response")
        print_json(response_data)
    else:
        print_header("📄 Response Summary")
        print(f"Request ID: {response_data.get('request_id', 'N/A')}")
        print(f"Answer: {response_data.get('answer', '')[:150]}...")
        print(f"Products: {context_count}")

        if context_count > 0:
            print("\nSample Product:")
            sample = response_data["used_context"][0]
            print(f"  Description: {sample.get('description', 'N/A')[:80]}...")
            print(f"  Price: ${sample.get('price', 'N/A')}")
            print(f"  Image: {'Available' if sample.get('image_url') else 'Not available'}")

    # Summary
    print()
    if all_passed:
        print_success("✅ Smoke test PASSED - RAG pipeline is working correctly")
    else:
        print_failure("❌ Smoke test FAILED - See errors above")

    return all_passed


def main():
    """Run smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test for RAG pipeline")
    parser.add_argument(
        "--query",
        default="best wireless headphones under $100",
        help="Query to test (default: 'best wireless headphones under $100')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full JSON response"
    )
    args = parser.parse_args()

    success = run_smoke_test(args.query, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
