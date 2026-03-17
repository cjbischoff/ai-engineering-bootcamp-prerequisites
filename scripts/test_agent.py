#!/usr/bin/env python3
"""
Test script for the coordinator agent API (POST /agent/).

Calls the agent endpoint, consumes the SSE stream, and prints:
- Status updates (e.g. "Analysing the question...")
- Final answer and metadata (answer, used_context, trace_id, shopping_cart)
- Validation: Suggestions tab (used_context) and Main page (answer)
- Any errors

Useful for debugging without the Streamlit UI. Run with API and services up:
    make health                    # Verify services
    uv run scripts/test_agent.py   # Default query
    uv run scripts/test_agent.py "best wireless headphones under $100"
    uv run scripts/test_agent.py --query "add earphones to my cart" --thread-id test-123
"""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("❌ Missing requests. Run: uv sync")
    sys.exit(1)

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Test agent API (POST /agent/)")
    parser.add_argument(
        "--query",
        default="What are the best wireless headphones?",
        help="Query to send",
    )
    parser.add_argument(
        "--thread-id",
        default="test-agent-script",
        help="Thread ID for conversation continuity",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/agent/",
        help="Agent API URL (use trailing slash to avoid 307)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if suggestions or main page validation fails",
    )
    args = parser.parse_args()

    print(f"Query: {args.query}")
    print(f"Thread ID: {args.thread_id}")
    print(f"URL: {args.url}")
    print("-" * 60)

    try:
        response = requests.post(
            args.url,
            json={"query": args.query, "thread_id": args.thread_id},
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=args.timeout,
        )
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect. Is the API running? (make run-docker-compose)")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        sys.exit(1)

    if response.status_code != 200:
        print(f"❌ Status {response.status_code}: {response.text[:500]}")
        sys.exit(1)

    print(f"✓ Status {response.status_code}\n")

    final_answer = None
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data:
            continue
        try:
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                continue
            if parsed.get("type") == "final_answer":
                final_answer = parsed.get("data", {})
                break
            if parsed.get("type") == "error":
                err = parsed.get("data", {}).get("message", "Unknown error")
                print(f"❌ Error: {err}")
                sys.exit(1)
        except json.JSONDecodeError:
            # Plain text status
            if data:
                print(f"  {data}")

    if not final_answer:
        print("❌ No final_answer in stream")
        sys.exit(1)

    answer = final_answer.get("answer", "")
    used_context = final_answer.get("used_context", [])
    trace_id = final_answer.get("trace_id", "")
    shopping_cart = final_answer.get("shopping_cart", [])

    # Validation: map to UI (Suggestions tab = used_context, Main page = answer)
    suggestions_ok = len(used_context) > 0
    main_page_ok = bool(answer and answer.strip())

    print(f"\n{BOLD}--- VALIDATION (UI mapping) ---{RESET}")
    print(
        f"{GREEN}✓{RESET} Suggestions tab: {len(used_context)} items"
        if suggestions_ok
        else f"{RED}✗{RESET} Suggestions tab: 0 items (expected product cards)"
    )
    print(
        f"{GREEN}✓{RESET} Main page: {len(answer)} chars"
        if main_page_ok
        else f"{RED}✗{RESET} Main page: empty (expected answer text)"
    )

    print(f"\n{BOLD}--- RESULT ---{RESET}")
    print(f"Answer ({len(answer)} chars):")
    print(answer[:500] + "..." if answer and len(answer) > 500 else (answer or "(empty)"))
    print(f"\nTrace ID: {trace_id or '(empty)'}")
    print(f"Used context: {len(used_context)} items")
    print(f"Shopping cart: {len(shopping_cart)} items")
    if used_context:
        print("\nUsed context sample (Suggestions tab):")
        for i, item in enumerate(used_context[:3]):
            desc = item.get("description", "")[:80]
            print(f"  [{i}] {desc}...")
    if shopping_cart:
        print("\nShopping cart:")
        for i, item in enumerate(shopping_cart[:5]):
            print(f"  [{i}] qty={item.get('quantity')} price={item.get('price')} {item.get('currency', '')}")
    print("-" * 60)

    if args.strict and (not suggestions_ok or not main_page_ok):
        print(f"{RED}Validation failed (--strict). Exit 1.{RESET}")
        sys.exit(1)
    print("✓ Done")


if __name__ == "__main__":
    main()
