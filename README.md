# AI Engineering Bootcamp Prerequisites

This repository contains prerequisite materials and exercises for the AI Engineering Bootcamp.

## Setup

### Prerequisites
- Python 3.12 or higher
- uv (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-engineering-bootcamp-prerequisites_me
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp env.example .env
```

4. Add your API keys to `.env`:
- `OPENAI_KEY`: Your OpenAI API key
- `GOOGLE_API_KEY`: Your Google AI API key
- `GROQ_API_KEY`: Your Groq API key

**Important:** Never commit your `.env` file with real API keys!

## Project Structure

```
.
├── notebooks/
│   └── week0/
│       └── 01-llm-apis.ipynb  # Introduction to LLM APIs
├── main.py                     # Main application entry point
├── pyproject.toml              # Project dependencies
└── .env                        # Environment variables (not tracked)
```

## Usage

### Running Notebooks

Open Jupyter notebooks in the `notebooks/` directory:
```bash
jupyter notebook notebooks/
```

### Running the Main Script

```bash
uv run python main.py
```

## Dependencies

- `openai`: OpenAI API client
- `google-genai`: Google Generative AI client
- `groq`: Groq API client
- `python-dotenv`: Environment variable management
- `ipykernel`: Jupyter kernel support

## Security Notes

- The `.env` file is gitignored to prevent accidental exposure of API keys
- Never share your API keys publicly
- Rotate your API keys immediately if they are exposed
