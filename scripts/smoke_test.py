#!/usr/bin/env python3
"""
Smoke Test Script for AI Engineering Bootcamp RAG Pipeline

Runs end-to-end test of the agent pipeline to verify:
- POST /agent/ responds with SSE (200)
- final_answer JSON contains answer, used_context (product-shaped dicts)
- Answer non-empty and retrieval returned at least one context item
- End-to-end latency (first request to final_answer) within a reasonable bound

STREAMING SUPPORT (Week 4 / Sprint 3):
--------------------------------------
The /agent/ endpoint returns text/event-stream (SSE) instead of JSON. Events:
- Plain text: Status updates ("Analysing the question...", "Planning...", etc.)
- JSON type=final_answer: Contains answer, used_context, trace_id
- JSON type=error: Contains error message

This script consumes the stream, parses SSE "data: " lines, and validates
the final_answer payload. request_id comes from X-Request-ID response header
(set by RequestIDMiddleware).

Usage:
    make smoke-test          # Run with summary output
    make smoke-test-verbose  # Show full JSON response
    uv run scripts/smoke_test.py            # Direct invocation
    uv run scripts/smoke_test.py --verbose  # Verbose mode
"""

import argparse
import json
import sys
import time
from typing import Any

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
    print(f"{Colors.CYAN}i{Colors.RESET} {text}")


def print_next_steps() -> None:
    """When smoke fails, suggest what to run next."""
    print_header("Suggested next steps")
    print(f"  {Colors.CYAN}•{Colors.RESET} `make health` — containers, Qdrant hybrid collection, API OpenAPI")
    print(f"  {Colors.CYAN}•{Colors.RESET} `docker compose logs -f api` — stack traces, LiteLLM/OpenAI errors")
    print(f"  {Colors.CYAN}•{Colors.RESET} `op signin` then `make run-docker-compose` if using 1Password for API keys")
    print(f"  {Colors.CYAN}•{Colors.RESET} `uv run scripts/test_agent.py --query \"...\"` — reproduce without Streamlit")


def print_json(data: dict[Any, Any]):
    """Pretty-print JSON data."""
    print(json.dumps(data, indent=2))


def validate_response_structure(response_data: dict[Any, Any]) -> tuple[bool, str]:
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
        if "image_url" in item and item["image_url"] is not None and not isinstance(
            item["image_url"], str
        ):
            return False, f"used_context[{idx}].image_url must be string or null"

        if "price" in item and item["price"] is not None and not isinstance(
            item["price"], (int, float)
        ):
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
    print_header("Smoke Test: Agent pipeline (POST /agent/)")
    print_info(f"Query: {query}")

    all_passed = True

    # Test 1: API responds (streaming SSE); elapsed = until final_answer (not just TTFB)
    start_time = time.time()
    try:
        # thread_id required by API (LangGraph checkpointing); fixed ID for reproducible smoke run.
        response = requests.post(
            "http://localhost:8000/agent/",
            json={"query": query, "thread_id": "smoke-test"},
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=120,
        )

        if response.status_code != 200:
            print_failure(f"API returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            print_next_steps()
            return False

    except requests.exceptions.ConnectionError:
        print_failure("Cannot connect to API (is it running?)")
        print_next_steps()
        return False
    except requests.exceptions.Timeout:
        print_failure("Request timed out (> 120 seconds)")
        print_next_steps()
        return False
    except Exception as e:
        print_failure(f"Error making request: {e!s}")
        print_next_steps()
        return False

    # Test 2: Consume SSE stream and extract final_answer
    response_data = None
    elapsed = 0.0
    try:
        request_id = response.headers.get("X-Request-ID", "unknown")
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = line[6:].strip()
                if not data:
                    continue
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict) and parsed.get("type") == "final_answer":
                        payload = parsed.get("data", {})
                        response_data = {
                            "request_id": request_id,
                            "trace_id": payload.get("trace_id") or "",
                            "answer": payload.get("answer", ""),
                            "used_context": payload.get("used_context", []),
                        }
                        elapsed = time.time() - start_time
                        break
                    if isinstance(parsed, dict) and parsed.get("type") == "error":
                        err_msg = parsed.get("data", {}).get("message", "Unknown error")
                        print_failure(f"Stream error: {err_msg}")
                        print_next_steps()
                        return False
                except json.JSONDecodeError:
                    pass  # Plain text status (e.g. "Analysing the question...")
        if response_data is None:
            elapsed = time.time() - start_time
            print_failure("No final_answer event in stream")
            print_next_steps()
            return False
        print_success(f"SSE final_answer received (wall time {elapsed:.2f}s, X-Request-ID={request_id})")
    except Exception as e:
        print_failure(f"Error parsing SSE stream: {e!s}")
        print_next_steps()
        return False

    # Test 3: Response structure matches Pydantic model
    valid, message = validate_response_structure(response_data)
    if valid:
        print_success(f"Response structure valid: {message}")
    else:
        print_failure(f"Response structure invalid: {message}")
        all_passed = False

    # Test 4: Response time (full stream, multi-agent + retrieval + LLM)
    # Cold start and coordinator loops can exceed 20s; treat as soft/strict bands.
    if elapsed < 45.0:
        print_success(f"Latency OK: {elapsed:.2f}s (< 45s target for full agent run)")
    elif elapsed < 90.0:
        print_info(f"Latency elevated: {elapsed:.2f}s (acceptable; watch for regressions)")
    else:
        print_failure(f"Latency high: {elapsed:.2f}s (>= 90s) — check API logs / provider slowness")
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
        print_header("Full Response")
        print_json(response_data)
    else:
        print_header("Response Summary")
        print(f"Request ID: {response_data.get('request_id', 'N/A')}")
        tid = response_data.get("trace_id") or "(none)"
        print(f"Trace ID: {tid}")
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
        print_success("Smoke test PASSED — agent returned answer + product context")
    else:
        print_failure("Smoke test FAILED — see errors above")
        print_next_steps()

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
