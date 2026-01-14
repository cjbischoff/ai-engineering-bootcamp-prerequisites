# AI Engineering Bootcamp Prerequisites

This repository contains prerequisite materials and a complete AI chatbot application stack for the AI Engineering Bootcamp, featuring a FastAPI backend and Streamlit frontend.

## Features

- **FastAPI Backend**: Multi-provider LLM API service supporting OpenAI, Groq, and Google GenAI
- **Streamlit Frontend**: Interactive chatbot UI with provider selection
- **Docker Support**: Containerized deployment with Docker Compose
- **Workspace Architecture**: Modular monorepo structure with `uv` package manager
- **Jupyter Notebooks**: Interactive tutorials for learning LLM APIs

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker and Docker Compose
- API Keys for:
  - OpenAI (optional, but quota may be exceeded)
  - Groq (recommended)
  - Google GenAI (recommended)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-engineering-bootcamp-prerequisites_me
```

### 2. Configure Environment Variables

```bash
cp env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
API_URL=http://api:8000
```

**⚠️ Important:** Never commit your `.env` file with real API keys!

### 3. Install Dependencies

```bash
uv sync
```

### 4. Run with Docker Compose

```bash
make run-docker-compose
```

Or manually:
```bash
uv sync
docker compose up --build
```

### 5. Access the Applications

- **Chatbot UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Project Structure

```
.
├── apps/
│   ├── api/                        # FastAPI Backend
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── src/api/
│   │       ├── app.py              # Main FastAPI application
│   │       └── core/
│   │           └── config.py       # Configuration management
│   │
│   └── chatbot_ui/                 # Streamlit Frontend
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── src/chatbot_ui/
│           ├── app.py              # Streamlit UI application
│           └── core/
│               └── config.py       # Configuration management
│
├── notebooks/
│   └── week0/
│       └── 01-llm-apis.ipynb       # LLM API tutorials
│
├── docker-compose.yml              # Multi-service orchestration
├── Makefile                        # Common commands
├── pyproject.toml                  # Root workspace configuration
└── .env                            # Environment variables (not tracked)
```

## API Endpoints

### FastAPI Backend (`http://localhost:8000`)

#### `GET /`
Welcome endpoint

**Response:**
```json
{
  "message": "Welcome to the AI Chat API",
  "status": "running"
}
```

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy"
}
```

#### `POST /chat`
Chat with AI providers

**Request Body:**
```json
{
  "provider": "Groq",
  "model_name": "llama-3.3-70b-versatile",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}
```

**Response:**
```json
{
  "message": "Hello! How can I help you today?"
}
```

**Supported Providers:**
- `OpenAI`: Models like `gpt-4o-mini`, `o1-mini`
- `Groq`: Models like `llama-3.3-70b-versatile`
- `Google`: Models like `gemini-2.0-flash-exp`

## Usage

### Using the Chatbot UI

1. Open http://localhost:8501 in your browser
2. Select a provider (OpenAI, Groq, or Google) from the sidebar
3. Choose a model from the dropdown
4. Type your message and press Enter
5. View the AI's response in the chat interface

### Using the API Directly

**With curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "Groq",
    "model_name": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**With Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "provider": "Groq",
        "model_name": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json())
```

### Running Jupyter Notebooks

```bash
uv run jupyter notebook notebooks/
```

Or with the activated virtual environment:
```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

### Cleaning Notebook Outputs

Before committing notebooks, clean their outputs:
```bash
make clean-notebook-outputs
```

## Development

### Workspace Structure

This project uses a `uv` workspace with multiple packages:

- **Root workspace**: Common dependencies and workspace configuration
- **apps/api**: FastAPI backend service
- **apps/chatbot_ui**: Streamlit frontend service

### Adding Dependencies

**To root workspace:**
```bash
uv add <package-name>
```

**To specific app:**
```bash
uv add --package api <package-name>
uv add --package chatbot-ui <package-name>
```

### Local Development (without Docker)

**Run API:**
```bash
cd apps/api
uv run uvicorn api.app:app --reload --port 8000
```

**Run Chatbot UI:**
```bash
cd apps/chatbot_ui
uv run streamlit run src/chatbot_ui/app.py
```

### Docker Commands

**Build services:**
```bash
docker compose build
```

**Run in detached mode:**
```bash
docker compose up -d
```

**View logs:**
```bash
docker compose logs -f
```

**Stop services:**
```bash
docker compose down
```

**Rebuild and restart:**
```bash
docker compose up --build --force-recreate
```

## Dependencies

### Root Workspace
- `openai>=2.15.0` - OpenAI API client
- `google-genai>=1.57.0` - Google Generative AI client
- `groq>=1.0.0` - Groq API client
- `streamlit>=1.52.2` - Streamlit web framework
- `pydantic>=2.12.5` - Data validation
- `jupyter>=1.1.1` - Jupyter notebook support
- `python-dotenv>=1.2.1` - Environment variable management

### API Service
- `fastapi>=0.128.0` - FastAPI framework
- `uvicorn>=0.40.0` - ASGI server

### Chatbot UI Service
- `streamlit>=1.52.2` - Streamlit framework
- `requests>=2.32.0` - HTTP client

## Makefile Commands

```bash
make run-docker-compose       # Sync dependencies and run Docker Compose
make clean-notebook-outputs   # Clear Jupyter notebook outputs
```

## Security Notes

### Environment Variables
- The `.env` file is gitignored to prevent accidental exposure of API keys
- Use `env.example` as a template
- Never commit real API keys to version control

### API Key Rotation
- Rotate your API keys immediately if they are exposed
- Monitor your API usage for unusual activity
- Use different keys for development and production

### Docker Security
- Services run as non-root users
- Environment variables are passed securely via `.env` file
- No secrets are baked into Docker images

## Troubleshooting

### Port Already in Use
If ports 8000 or 8501 are already in use:
```bash
# Find process using the port
lsof -i :8000
lsof -i :8501

# Kill the process or change ports in docker-compose.yml
```

### API Connection Errors
- Ensure the API service is running: `docker compose ps`
- Check API logs: `docker compose logs api`
- Verify `API_URL` in `.env` is set to `http://api:8000`

### Missing Dependencies
```bash
# Reinstall all dependencies
uv sync --reinstall
```

### Docker Build Issues
```bash
# Clean Docker cache and rebuild
docker compose down
docker system prune -f
docker compose up --build
```

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Clean notebook outputs: `make clean-notebook-outputs`
4. Commit your changes
5. Push and create a pull request

## License

This project is for educational purposes as part of the AI Engineering Bootcamp.

## Support

For questions or issues, please open an issue in the repository or contact the bootcamp instructors.
