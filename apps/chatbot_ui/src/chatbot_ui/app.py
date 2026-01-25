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
- Frontend sends POST request to FastAPI backend (/rag endpoint)
- Backend runs RAG pipeline (retrieve products, generate answer, enrich with metadata)
- Frontend displays:
  1. LLM's answer in the chat
  2. Product cards in the sidebar (Video 4 enhancement)

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
        url (str): Full URL to call (e.g., "http://api:8000/rag")
        **kwargs: Additional arguments passed to requests (json, headers, timeout, etc.)

    Returns:
        Tuple[bool, dict]: (success, response_data)
        - success: True if request succeeded (2xx status), False otherwise
        - response_data: Parsed JSON from response or error dict

    Example:
        success, data = api_call("post", f"{config.API_URL}/rag", json={"query": "..."})
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
# RENDER CHAT HISTORY
# =============================================================================
# Display all previous messages in the chat interface
# This happens on EVERY rerun, so users see the full conversation history

for message in st.session_state.messages:
    # st.chat_message() creates a styled chat bubble
    # role="user" -> bubble on right side with user avatar
    # role="assistant" -> bubble on left side with bot avatar
    with st.chat_message(message["role"]):
        # st.markdown() supports Markdown formatting (bold, italics, links, etc.)
        # This allows rich text in responses (e.g., "**Product**: XYZ")
        st.markdown(message["content"])
# Educational: The "with" context manager ensures content goes inside the chat bubble
# Without it, content would appear outside the styled bubble


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


# Create sidebar content
# Educational: "with st.sidebar:" context manager puts all content in the left sidebar
# This is cleaner than calling st.sidebar.write(), st.sidebar.image(), etc. repeatedly
with st.sidebar:
    # =============================================================================
    # SIDEBAR TABS
    # =============================================================================
    # Educational: st.tabs() creates clickable tabs (like browser tabs)
    # Returns a tuple of tab objects, one per tab label
    # The comma after suggestions_tab unpacks the single-item tuple: (tab,) -> tab

    # Create tabs in the sidebar
    # Why tabs? Future enhancement could add more tabs:
    # - "🔍 Suggestions" (current products)
    # - "📊 History" (past conversations)
    # - "⚙️ Settings" (model selection, temperature, etc.)
    suggestions_tab, = st.tabs(["🔍 Suggestions"])
    # Educational: The trailing comma is REQUIRED for single-item tuple unpacking
    # Without it: suggestions_tab = st.tabs([...]) would be a tuple, not a tab object
    # With it: suggestions_tab, = st.tabs([...]) unpacks to just the tab

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
        # Make POST request to RAG endpoint
        # Educational: We're sending JSON: {"query": "user's question"}
        # API returns JSON: {"request_id": "...", "answer": "...", "used_context": [...]}
        status, output = api_call(
            "post",  # HTTP POST method
            f"{config.API_URL}/rag",  # URL from config (e.g., "http://api:8000/rag")
            json={"query": prompt}  # Request body (automatically serialized to JSON)
        )
        # Educational: requests library automatically:
        # - Sets Content-Type: application/json header
        # - Converts dict to JSON string
        # - Sends in request body

        # Extract data from API response
        # Educational: We assume API call succeeded (output has expected structure)
        # Production code should check: if status: ... else: show error
        answer = output["answer"]  # LLM's natural language response
        used_context = output["used_context"]  # List of product dicts (Video 3 feature)

        # =============================================================================
        # VIDEO 4: UPDATE SIDEBAR WITH NEW PRODUCTS
        # =============================================================================
        # Store products in session state so sidebar can display them
        # Why here? We want sidebar to update immediately after getting API response
        st.session_state.used_context = used_context
        # Educational: This triggers sidebar to show products on next rerun (line 66-74)
        # Sidebar code checks "if st.session_state.used_context:" and displays cards

        # Display the LLM's answer in the chat
        st.write(answer)
        # Educational: st.write() is the Swiss Army knife of Streamlit
        # It automatically formats based on data type:
        # - String: displays as text
        # - DataFrame: displays as table
        # - Dict: displays as JSON
        # - Plot: displays as chart

        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
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
