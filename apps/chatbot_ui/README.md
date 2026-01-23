# Streamlit Frontend UI

Interactive chatbot interface built with Streamlit, providing a user-friendly web UI for conversing with the RAG-powered product Q&A system.

## Overview

This Streamlit application serves as the frontend for the RAG chatbot system. It provides a clean, responsive chat interface that communicates with the FastAPI backend to deliver natural language answers about products.

## Architecture

```
apps/chatbot_ui/
├── src/chatbot_ui/
│   ├── app.py                 # Main Streamlit application
│   └── core/
│       └── config.py          # Environment configuration
├── Dockerfile                 # Container image definition
└── pyproject.toml             # Python dependencies
```

## Key Components

### Application (`app.py`)

**Core Functionality**:
1. **API Communication** (`api_call` helper function)
   - Generic HTTP client wrapper around `requests` library
   - Error handling with user-friendly popup messages
   - Returns tuple: `(success: bool, data: dict)`

2. **Session State Management**
   - Maintains conversation history in `st.session_state.messages`
   - Format: `[{"role": "user"|"assistant", "content": str}]`
   - Persists across Streamlit reruns (form submissions)

3. **Chat Interface**
   - Message rendering with `st.chat_message(role)`
   - User input via `st.chat_input()`
   - Scrollable conversation history

4. **Error Handling**
   - Connection errors: Network issues
   - Timeout errors: API response delays
   - Popup notifications for user feedback

**Streamlit Pattern**: Imperative rendering - script runs top-to-bottom on every interaction.

### Configuration (`core/config.py`)

- **Pattern**: `pydantic-settings` with `.env` file loading
- **Environment Variables**:
  - `API_URL`: Backend endpoint (default: `http://api:8000`)

**Docker vs Local**:
- Docker: `http://api:8000` (Docker Compose service name)
- Local: `http://localhost:8000` (localhost)

## User Flow

1. **Initial Load**
   - Streamlit initializes session state
   - Default greeting message: "Hello! How can I assist you today?"

2. **User Asks Question**
   - User types in chat input
   - Message added to session state
   - Displayed in chat interface

3. **API Request**
   - POST to `/rag/` endpoint: `{"query": user_message}`
   - Loading spinner shown during request

4. **Response Handling**
   - **Success**: Display assistant's answer in chat
   - **Error**: Show popup notification with error message

5. **Conversation Continues**
   - New question repeats flow
   - Full history maintained in session state

## Running Locally

### Docker (Recommended)
```bash
# From project root
make run-docker-compose
# or manually:
docker compose up --build
```

**Service URL**: http://localhost:8501

### Local Development (Without Docker)
```bash
cd apps/chatbot_ui

# Install dependencies
uv sync

# Run Streamlit app
uv run streamlit run src/chatbot_ui/app.py
```

**Note**: Requires FastAPI backend running. Update `API_URL` in `.env` to point to backend.

## Dependencies

**Core Framework**:
- `streamlit` - Web UI framework
- `requests` - HTTP client for API calls
- `pydantic-settings` - Configuration management

See [pyproject.toml](pyproject.toml) for complete dependency list.

## Docker Integration

**Dockerfile Highlights**:
- Base image: `python:3.12-slim`
- Package manager: `uv` for fast dependency resolution
- Working directory: `/app`
- Port: 8501 (Streamlit default)
- Hot reload: Volume mounts in `docker-compose.yml`

**Volume Mounts** (from project root):
```yaml
volumes:
  - ./apps/chatbot_ui/src:/app/apps/chatbot_ui/src  # Hot reload
```

**Environment Variables**: Loaded from `.env` via `env_file` in `docker-compose.yml`

## Development Workflow

1. **Add Dependency**: `uv add --package chatbot-ui <package-name>`
2. **Modify Code**: Changes auto-reload (Streamlit watches files)
3. **Test Locally**: Open http://localhost:8501 in browser
4. **Check API Connection**: Verify backend is running and `API_URL` is correct

## Key Design Patterns

### 1. Session State for Conversation History
- Streamlit reruns script on every interaction
- `st.session_state` persists data across reruns
- Pattern: Check if key exists before initializing

```python
if "messages" not in st.session_state:
    st.session_state.messages = [initial_message]
```

### 2. Error Handling with Popups
- User-friendly messages instead of stack traces
- Specific error types: Connection, Timeout, General
- Non-blocking: User can continue chatting after error

### 3. Generic API Client
- `api_call(method, url, **kwargs)` wraps `requests`
- Reusable for GET, POST, PUT, DELETE
- Centralized error handling

### 4. Streamlit Chat Components
- `st.chat_message(role)`: Styled message container
- `st.chat_input()`: Bottom-anchored input field
- `st.markdown()`: Rich text rendering (supports code blocks, links)

### 5. Docker Networking
- In Docker: Services communicate via service names (`http://api:8000`)
- Localhost refers to container itself, not host machine
- `API_URL` must match deployment environment

## Environment Variables

Create `.env` file (see `env.example`):

```env
API_URL=http://api:8000  # Docker Compose
# or
API_URL=http://localhost:8000  # Local development
```

## Troubleshooting

**Connection Errors**:
- Check `API_URL` matches running backend
- In Docker: Use `http://api:8000` (service name)
- Locally: Use `http://localhost:8000`
- Verify backend is running: `curl http://localhost:8000/health`

**UI Not Loading**:
- Check Streamlit logs: `docker compose logs chatbot-ui`
- Verify port 8501 not in use: `lsof -i :8501`
- Restart container: `docker compose restart chatbot-ui`

**Session State Issues**:
- Streamlit Cloud: Session state cleared on inactivity
- Local: Persistent until page refresh
- Debug: Print `st.session_state` to inspect contents

**API Response Errors**:
- Check backend logs: `docker compose logs api`
- Verify request format matches API expectations
- Use browser DevTools Network tab to inspect requests

## Known Limitations

**Not Implemented (Intentional MVP Scope)**:
- Authentication: No user login
- Conversation persistence: History lost on page refresh
- Multi-turn context: Each question independent (no conversation memory)
- Rate limiting: No client-side throttling
- Message editing: Can't edit previous messages
- Conversation export: Can't save/download chat history
- Typing indicators: No visual feedback while API processes
- Error retry: Manual refresh required after errors

**When to Add**:
- Authentication: When deploying publicly
- Persistence: When users need to return to conversations
- Multi-turn: When context across questions is important
- Export: When users want to save conversations

## UI Enhancements (Future)

**Potential Improvements**:
1. Display retrieved products as cards (images, ratings, links)
2. Show similarity scores to indicate confidence
3. Add "thinking" animation during API calls
4. Enable message reactions (helpful/not helpful)
5. Syntax highlighting for code in responses
6. Dark mode toggle
7. Conversation branching (explore different answers)
8. Share conversation via URL

## Related Documentation

- **Project Root**: [../../README.md](../../README.md) - Overall architecture
- **Backend API**: [../api/README.md](../api/README.md) - FastAPI service
- **Notebooks**: [../../notebooks/week1/README.md](../../notebooks/week1/README.md) - Data preprocessing

## Streamlit Resources

- **Documentation**: https://docs.streamlit.io
- **Chat Components**: https://docs.streamlit.io/develop/api-reference/chat
- **Session State**: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- **Deployment**: https://docs.streamlit.io/deploy

## Production Considerations

Before deploying to production:
1. Add authentication (Streamlit Auth, OAuth)
2. Implement rate limiting on client side
3. Add conversation persistence (database)
4. Enable HTTPS/TLS
5. Set up monitoring (error tracking, usage analytics)
6. Add input sanitization
7. Implement session timeouts
8. Configure CORS properly
9. Add comprehensive error messages
10. Set up CI/CD pipeline with UI testing
