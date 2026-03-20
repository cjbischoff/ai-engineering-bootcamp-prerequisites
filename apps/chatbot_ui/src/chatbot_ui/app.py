"""
Streamlit Chatbot UI - AI Engineering Bootcamp

PURPOSE:
========
This is the frontend application for the AI chatbot, built with Streamlit.
It provides a chat interface for users to interact with the RAG (Retrieval-Augmented
Generation) pipeline and displays product recommendations with enriched metadata.

ARCHITECTURE:
=============
- User types a question in the chat input
- Frontend sends POST to FastAPI /agent/ with query and thread_id (session_id)
- Backend runs LangGraph coordinator -> product_qa_agent | shopping_cart_agent (Week 5)
- Backend streams SSE: status updates ("Planning...", "Looking for items...") and final_answer JSON
- Frontend displays:
  1. LLM's answer in the chat
  2. Product cards in Suggestions tab; Shopping Cart tab shows cart items (Week 5)

VIDEO PROGRESSION:
==================
- Video 1-2: Basic chat interface (just text responses)
- Video 3: API returns enriched product metadata (images, prices)
- Video 4: Sidebar displays visual product cards (THIS FILE'S ENHANCEMENT)

STREAMLIT CONCEPTS:
===================
- st.session_state: Persists data across reruns (like browser session storage)
- st.rerun(): Triggers a full page refresh to show new data
- st.sidebar: Content appears in left sidebar instead of main area
- st.chat_message(): Styled chat bubbles (user vs assistant)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import requests  # HTTP library for making API calls to our FastAPI backend
import streamlit as st  # Streamlit framework for building the web UI

from chatbot_ui.core.config import config  # Configuration (API_URL from .env)
import uuid
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Educational: This must be the first Streamlit command (before any st.* calls)
# It configures the overall page layout and browser tab appearance

st.set_page_config(
    page_title="Chatbot UI",  # Text shown in browser tab
    layout="wide",  # Use full browser width (vs "centered" which limits width)
    initial_sidebar_state="expanded",  # Sidebar visible by default (vs "collapsed")
)
# Why "expanded"? We're showing product recommendations in the sidebar (Video 4)
# so we want it visible immediately, not requiring users to click to open it

def get_session_id():
    """Stable ID per browser session so the backend can persist multi-turn state (LangGraph thread_id)."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit thumbs and/or comment to API POST /submit_feedback/ for LangSmith (Week 4)."""
    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None

    # trace_id from last RAG response (stored below); API/LangSmith need it to attach feedback to the run.
    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "thread_id": get_session_id(),
        "feedback_source_type": "api"
    }

    logger.info(
        "Submitting feedback: trace_id=%s (present=%s), feedback_score=%s, has_text=%s",
        st.session_state.trace_id,
        st.session_state.trace_id is not None and bool(st.session_state.trace_id),
        feedback_data["feedback_score"],
        bool(feedback_text and feedback_text.strip()),
    )

    status, response = api_call("post", f"{config.API_URL}/submit_feedback/", json=feedback_data)
    return status, response




# =============================================================================
# API COMMUNICATION
# =============================================================================

def api_call(method, url, **kwargs):
    """
    Make HTTP requests to the backend API with comprehensive error handling.

    HOW IT WORKS:
    1. Makes HTTP request using requests library
    2. Attempts to parse JSON response
    3. Returns (success: bool, data: dict) tuple
    4. Shows user-friendly error popups if something fails

    WHY THIS WRAPPER:
    - Centralizes error handling (don't repeat try/except everywhere)
    - Provides consistent error messages to users
    - Returns predictable tuple format for easy handling
    - Uses Streamlit session state for error popups

    Args:
        method (str): HTTP method ('get', 'post', 'put', 'delete')
        url (str): Full URL to call (e.g., "http://api:8000/agent")
        **kwargs: Additional arguments passed to requests (json, headers, timeout, etc.)

    Returns:
        Tuple[bool, dict]: (success, response_data)
        - success: True if request succeeded (2xx status), False otherwise
        - response_data: Parsed JSON from response or error dict

    Example:
        success, data = api_call("post", f"{config.API_URL}/agent", json={"query": "..."})
        if success:
            answer = data["answer"]
            products = data["used_context"]
    """
    def _show_error_popup(message):
        """
        Display error message to user via Streamlit session state.

        Why session state?
        - Allows showing error in a dedicated UI element (not inline)
        - Persists across reruns until dismissed
        - Could be rendered in a st.error() box at top of page

        Currently just stores in state - would need corresponding UI element to display.
        """
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        # Make HTTP request using dynamic method selection
        # getattr(requests, method) gets requests.post, requests.get, etc.
        # This is more flexible than hardcoding "if method == 'post': requests.post(...)"
        response = getattr(requests, method)(url, **kwargs)

        # Try to parse JSON response
        # Educational: Most modern APIs return JSON, but we should handle exceptions
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            # Server returned non-JSON response (HTML error page, plain text, etc.)
            # This could happen if:
            # - Server is returning 500 error with HTML error page
            # - URL is wrong and hit a different service
            # - Response is malformed
            response_data = {"message": "Invalid response format from server"}

        # Check if request was successful (status code 2xx)
        # response.ok is True for status codes 200-299
        if response.ok:
            return True, response_data

        # Request failed (4xx client error, 5xx server error)
        # Return the error response from server (might contain useful error message)
        return False, response_data

    except requests.exceptions.ConnectionError:
        # Cannot connect to server at all
        # Causes:
        # - API container isn't running
        # - Wrong URL/port
        # - Network issues
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}

    except requests.exceptions.Timeout:
        # Request took too long to complete
        # This means server is responding but taking too long to process
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}

    except Exception as e:
        # Catch-all for unexpected errors
        # Could be: SSL errors, DNS resolution, malformed URLs, etc.
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}



def api_call_stream(method, url, **kwargs):
    """
    Make streaming HTTP request to /agent/ endpoint (SSE).

    Returns either:
    - Iterator of bytes/str (response.iter_lines()) on success
    - (False, {"message": "..."}) on error (connection, timeout, etc.)

    WHY check isinstance(stream_result, tuple)? On error we return (False, dict)
    instead of an iterator. Caller must check before iterating to avoid
    'bool' object has no attribute 'decode' when looping over error tuple.
    """
    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner"""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, stream=True, **kwargs)
        return response.iter_lines()
    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}



# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Educational: Streamlit reruns the entire script on every interaction
# Session state persists data across these reruns (like React state or Vue data)

# Initialize chat message history if it doesn't exist
# Why check "if not in session_state"?
# - First page load: messages don't exist, create them
# - Subsequent reruns: messages already exist, don't reset them
if "messages" not in st.session_state:
    st.session_state.messages = [
        # Start with a friendly greeting from the assistant
        # "role" matches OpenAI chat API format (system/user/assistant)
        {"role": "assistant", "content": "Hello! How can i assist you today?"}
    ]
# Educational: This list grows as conversation continues
# Each user message and assistant response gets appended
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]


# =============================================================================
# VIDEO 4 FEATURE: PRODUCT SUGGESTIONS SIDEBAR
# =============================================================================
# Initialize used_context in session state (stores product metadata)
# This was added in Video 3 when API started returning enriched product data

if "used_context" not in st.session_state:
    # used_context stores the product cards to display in sidebar
    # Structure: list of dicts with {image_url, price, description}
    # Example: [
    #   {"image_url": "https://...", "price": 39.99, "description": "TELSOR Earbuds..."},
    #   {"image_url": "https://...", "price": 29.99, "description": "Sony WH-1000XM4..."}
    # ]
    st.session_state.used_context = []
# Why empty list initially?
# - No conversation yet means no products to show
# - Gets populated after first API call (see line 90 below)

if "shopping_cart" not in st.session_state:
    st.session_state.shopping_cart = []

# Initialize feedback states (simplified)
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False

if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None








# Create sidebar content
# Educational: "with st.sidebar:" context manager puts all content in the left sidebar
# This is cleaner than calling st.sidebar.write(), st.sidebar.image(), etc. repeatedly
with st.sidebar:
    # =============================================================================
    # SIDEBAR TABS
    # =============================================================================
    # Educational: st.tabs() creates clickable tabs (like browser tabs)
    # Returns a tuple of tab objects, one per tab label
    # Unpacks the tuple: (tab1, tab2) -> suggestions_tab, shopping_cart_tab

    # Create tabs in the sidebar
    # Why tabs? Future enhancement could add more tabs:
    # - "🔍 Suggestions" (current products)
    # - "📊 History" (past conversations)
    # - "⚙️ Settings" (model selection, temperature, etc.)
    suggestions_tab, shopping_cart_tab = st.tabs(["🔍 Suggestions","🛒 Shopping Cart"])

    # Educational: st.tabs() returns a tuple; unpack to get individual tab objects

    # =============================================================================
    # SUGGESTIONS TAB CONTENT
    # =============================================================================
    # Display product cards with images, prices, and descriptions
    # This is the visual grounding that shows users "what products did the LLM use?"

    with suggestions_tab:
        # Check if we have any products to display
        # Educational: "if list:" checks if list is non-empty (truthy)
        # Empty list [] is falsy, non-empty list is truthy
        if st.session_state.used_context:
            # We have products! Loop through and create a card for each one

            # Educational: enumerate() gives us both index and item
            # idx is useful for error messages or unique keys
            # item is the product dict: {image_url, price, description}
            for idx, item in enumerate(st.session_state.used_context):

                # Display product description as a caption (smaller text, gray color)
                # Educational: .get('description', 'No description') is safe dictionary access
                # - If 'description' key exists: returns its value
                # - If 'description' key missing: returns 'No description' (default)
                # - Prevents KeyError that would crash the app
                st.caption(item.get('description', 'No description'))
                # Why caption? It's styled for secondary text (vs st.write() for main text)

                # =============================================================================
                # CONDITIONAL IMAGE DISPLAY
                # =============================================================================
                # Only show image if image_url exists AND is not None
                # Why two checks?
                # - 'image_url' in item: checks if key exists in dictionary
                # - item['image_url']: checks if value is truthy (not None, not empty string)
                #
                # This is necessary because API returns Optional[str] for image_url
                # Some products don't have images in Qdrant database
                if 'image_url' in item and item['image_url']:
                    # Display the product image
                    # width=250 keeps images consistent size (prevents layout jumping)
                    st.image(item["image_url"], width=250)
                    # Educational: st.image() can take:
                    # - URL string (what we're using)
                    # - File path ("/path/to/image.png")
                    # - PIL Image object (for dynamic image generation)
                    # - NumPy array (for scientific computing)
                # If no image available, just skip it (graceful degradation)
                # Could add: else: st.write("No image available")

                # Display product price as a caption
                # Why caption instead of st.write()?
                # - Consistent styling with description above
                # - Keeps UI compact (captions are smaller/lighter)
                st.caption(f"Price: {item['price']} USD")
                # Educational: f-string formatting embeds the price value
                # Alternative: st.caption("Price: " + str(item['price']) + " USD")
                # But f-strings are more readable and Pythonic

                # Add a visual separator between products
                # st.divider() creates a horizontal line (like <hr> in HTML)
                st.divider()
                # Why dividers? Makes it clear where one product card ends and next begins
                # Without dividers, cards would blend together visually

        else:
            # No products in used_context yet (initial page load, no conversation)
            # Show helpful message so sidebar doesn't look broken/empty

            # st.info() creates a blue information box (vs st.error() red, st.warning() yellow)
            st.info("No suggestions yet")
            # Educational: This tells users:
            # 1. The sidebar is working (it's not a bug that it's empty)
            # 2. Products will appear here after asking a question
            #
            # Alternative UI patterns:
            # - Show sample products: st.write("Ask about headphones, laptops, cameras...")
            # - Show tips: st.write("💡 Try asking: 'best wireless headphones under $100'")
            # - Hide sidebar completely: if not used_context: st.sidebar.hide()


    # =============================================================================
    # SHOPPING CART TAB CONTENT
    # =============================================================================
    with shopping_cart_tab:
        if st.session_state.shopping_cart:
            for idx, item in enumerate(st.session_state.shopping_cart):
                st.caption(item.get('description', 'No description'))
                if 'product_image_url' in item:
                    st.image(item["product_image_url"], width=250)
                st.caption(f"Price: {item['price']} {item['currency']}")
                st.caption(f"Quantity: {item['quantity']}")
                st.caption(f"Total price: {item['total_price']} {item['currency']}")
                st.divider()
        else:
            st.info("Your cart is empty")


for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
    is_latest_assistant = (
        message["role"] == "assistant"
        and idx == len(st.session_state.messages) - 1
        and idx > 0
    )

    if is_latest_assistant:
        # Use Streamlit's built-in feedback component
        feedback_key = f"feedback_{len(st.session_state.messages)}"
        feedback_result = st.feedback("thumbs", key=feedback_key)

        # Handle feedback selection
        if feedback_result is not None:
            feedback_type = "positive" if feedback_result == 1 else "negative"

            # Only submit if this is a new/different feedback
            if st.session_state.latest_feedback != feedback_type:
                with st.spinner("Submitting feedback..."):
                    status, response = submit_feedback(feedback_type=feedback_type)
                # Always set latest_feedback so we don't retry on every rerun (avoids request storm on 422)
                st.session_state.latest_feedback = feedback_type
                if status:
                    st.session_state.feedback_submission_status = "success"
                    st.session_state.show_feedback_box = (feedback_type == "negative")
                else:
                    st.session_state.feedback_submission_status = "error"
                    st.error("Failed to submit feedback. Please try again.")
                st.rerun()

        # Show feedback status message
        if st.session_state.latest_feedback and st.session_state.feedback_submission_status == "success":
            if st.session_state.latest_feedback == "positive":
                st.success("🎉 Thank you for your positive feedback!")
            elif st.session_state.latest_feedback == "negative" and not st.session_state.show_feedback_box:
                st.success("🙏 Thank you for your feedback!")
        elif st.session_state.feedback_submission_status == "error":
            st.error("❌ Failed to submit feedback. Please try again.")

        # Show feedback text box if thumbs down was pressed
        if st.session_state.show_feedback_box:
            st.markdown("**Want to tell us more? (Optional)**")
            st.caption("Your negative feedback has already been recorded. You can optionally provide additional details below.")

            feedback_text = st.text_area(
                "Additional feedback (optional)",
                key=f"feedback_text_{len(st.session_state.messages)}",
                placeholder="Please describe what was wrong with this response...",
                height=100
            )

            col_send, col_spacer, col_close = st.columns([3, 5, 2])
            with col_send:
                if st.button("Send Additional Details", key=f"send_additional_{len(st.session_state.messages)}"):
                    if feedback_text.strip():
                        with st.spinner("Submitting additional feedback..."):
                            status, response = submit_feedback(feedback_text=feedback_text)
                        if status:
                            st.success("✅ Thank you! Your additional feedback has been recorded.")
                            st.session_state.show_feedback_box = False
                        else:
                            st.error("❌ Failed to submit additional feedback. Please try again.")
                    else:
                        st.warning("Please enter some feedback text before submitting.")
                    st.rerun()

            with col_close:
                if st.button("Close", key=f"close_feedback_{len(st.session_state.messages)}"):
                    st.session_state.show_feedback_box = False
                    st.rerun()





# =============================================================================
# CHAT INPUT AND MESSAGE HANDLING
# =============================================================================
# This is the main interaction loop where users type questions and get answers

# Educational: := is the "walrus operator" (Python 3.8+)
# It assigns AND checks in one line:
#   prompt := st.chat_input(...)  # Assign user input to prompt variable
#   if prompt:                     # Check if prompt is truthy (not empty)
#
# Traditional way (two lines):
#   prompt = st.chat_input(...)
#   if prompt:
#
# Why walrus operator? More concise, commonly seen in Streamlit examples

if prompt := st.chat_input("Hello! How can I assist you today?"):
    # User submitted a message! Process it:

    # 1. Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display user's message immediately (gives instant feedback)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Call backend API and display assistant's response
    with st.chat_message("assistant"):
        session_id = get_session_id()  # Same ID across turns so backend can load/save conversation state

        status_placeholder = st.empty()
        message_placeholder = st.empty()
        stream_result = api_call_stream(
            "post",
            f"{config.API_URL}/agent",
            json={"query": prompt, "thread_id": session_id},
            headers={"Accept": "text/event-stream"},
        )
        # On error, api_call_stream returns (False, {"message": "..."}) instead of an iterator
        if isinstance(stream_result, tuple):
            _status, err_data = stream_result
            err_msg = err_data.get("message", "Request failed")
            logger.warning("Stream request failed: %s", err_msg)
            status_placeholder.error(err_msg)
            answer = err_msg
            st.session_state.used_context = []
        else:
            answer = None
            for line in stream_result:
                line_text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
                if line_text.startswith("data: "):
                    data = line_text[6:]
                    try:
                        output = json.loads(data)

                        if output.get("type") == "error":
                            answer = output.get("data", {}).get("message", "An error occurred")
                            logger.warning("Stream error from backend: %s", answer)
                            status_placeholder.error(answer)
                            break
                        if output.get("type") == "final_answer":
                            answer = output["data"].get("answer", "")
                            used_context = output["data"].get("used_context", [])
                            trace_id = output["data"].get("trace_id", "")
                            shopping_cart = output["data"].get("shopping_cart", [])

                            st.session_state.used_context = used_context
                            st.session_state.trace_id = trace_id
                            st.session_state.shopping_cart = shopping_cart

                            st.session_state.latest_feedback = None
                            st.session_state.show_feedback_box = False
                            st.session_state.feedback_submission_status = None

                            status_placeholder.empty()
                            message_placeholder.markdown(answer if answer else "_No response received._")
                            logger.info("Received final_answer: trace_id=%s, used_context_len=%d, answer_len=%d", trace_id, len(used_context), len(answer or ""))
                            break

                    except json.JSONDecodeError:
                        # Plain text status (not JSON): "Analysing the question...", "Planning...", etc.
                        # Show in status_placeholder for progressive feedback (Week 4 streaming UX)
                        status_text = data.strip()
                        if status_text:
                            logger.info("Stream status: %s", status_text)
                            status_placeholder.markdown(f"*{status_text}*")




        # Add assistant's response to chat history (single append; do not append inside final_answer block)
        if answer is None:
            answer = "No response received."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        logger.info("Appended assistant message to history (total=%d)", len(st.session_state.messages))
        # Educational: Now both user's question AND assistant's answer are in history
        # Next rerun will display them in the chat (see line 51-53)

        # =============================================================================
        # STREAMLIT RERUN
        # =============================================================================
        # Force Streamlit to rerun the entire script to show updated UI
        # Why needed?
        # - We updated session state (messages, used_context)
        # - Streamlit doesn't automatically detect these changes
        # - st.rerun() triggers a fresh top-to-bottom execution
        # - Causes sidebar to re-render with new products (line 66-74)
        st.rerun()
        # Educational: Without this:
        # - Answer appears in chat (because we're inside chat_message context)
        # - But sidebar wouldn't update until next user interaction
        # - Creates confusing UX (sidebar shows old products)
        #
        # With st.rerun():
        # - Sidebar updates immediately to show current products
        # - Chat history properly includes new messages
        # - Full UI is in sync with session state

# =============================================================================
# END OF SCRIPT
# =============================================================================
# Educational: Key Streamlit patterns demonstrated:
# 1. Session state for persistence (messages, used_context)
# 2. Conditional rendering (if used_context, if image_url)
# 3. Context managers (with st.sidebar, with st.chat_message)
# 4. API communication with error handling
# 5. Walrus operator for concise input handling
# 6. Manual rerun to update UI after state changes
#
# Future enhancements:
# - Add loading spinner during API call: with st.spinner("Thinking..."):
# - Add error handling UI: if not status: st.error(output["message"])
# - Add message editing: st.button("Edit") next to each message
# - Add conversation export: st.download_button("Download Chat")
# - Add streaming responses: for chunk in stream: st.write(chunk)
