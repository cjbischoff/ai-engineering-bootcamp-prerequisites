# Week 0: LLM API Fundamentals

Jupyter notebooks exploring basic LLM API interactions with OpenAI, Groq, and Google GenAI. This week establishes foundational understanding of how to call LLM APIs programmatically.

## Overview

Week 0 focuses on getting comfortable with LLM APIs before building more complex applications. You'll learn how to authenticate, make requests, and handle responses from multiple LLM providers.

## Notebooks

### 01-llm-apis.ipynb

**Purpose**: Introduction to calling LLM APIs from three major providers

**What You'll Learn**:
- Setting up API clients (OpenAI, Groq, Google)
- Environment variable management with `python-dotenv`
- Making basic chat completion requests
- Understanding request/response structure
- Comparing API interfaces across providers

**Key Concepts**:
1. **API Authentication**: API keys via environment variables
2. **Chat Format**: Role-based messages (system, user, assistant)
3. **Model Selection**: Different models per provider
4. **Provider Differences**: API structure varies between providers

**Example Request Pattern**:
```python
# OpenAI
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

# Groq
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello"}]
)

# Google GenAI
response = genai_client.generate_content(
    contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
    model="gemini-2.0-flash-exp"
)
```

**Why Different Providers**:
- **OpenAI**: Industry standard, most features
- **Groq**: Fast inference, cost-effective
- **Google**: Gemini models, different API pattern

**Prerequisites**:
- API keys from all three providers
- `.env` file with keys (see `env.example`)

**Running the Notebook**:
```bash
# From project root
uv run jupyter notebook notebooks/week0/01-llm-apis.ipynb
```

**Common Issues**:
1. **Missing API Keys**: Ensure `.env` has all keys
2. **ImportError**: Run `uv sync` to install dependencies
3. **Rate Limits**: Free tier limits may apply

## Environment Setup

Create `.env` file in project root:
```env
OPENAI_KEY=sk-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AI...
```

See [../../env.example](../../env.example) for complete template.

## Dependencies

All dependencies managed via `uv` workspace:
- `openai` - OpenAI Python SDK
- `groq` - Groq Python SDK
- `google-generativeai` - Google GenAI SDK
- `python-dotenv` - Environment variable loading
- `jupyter` - Notebook environment

Install with:
```bash
uv sync
```

## Key Takeaways

1. **Consistent Pattern Across Providers**:
   - Initialize client with API key
   - Create messages array
   - Call completion method
   - Extract response text

2. **Provider-Specific Differences**:
   - OpenAI/Groq: Similar interface (both use `chat.completions.create`)
   - Google: Different structure (`generate_content`, `contents` instead of `messages`)

3. **API Key Security**:
   - Never commit `.env` to git
   - Use environment variables, not hardcoded strings
   - `.gitignore` already excludes `.env`

4. **Model Names**:
   - OpenAI: `gpt-4o-mini`, `o1-mini`
   - Groq: `llama-3.3-70b-versatile`
   - Google: `gemini-2.0-flash-exp`

## Next Steps

After completing Week 0, you'll move to Week 1 where you'll:
- Build a RAG (Retrieval-Augmented Generation) system
- Work with vector databases (Qdrant)
- Create embeddings for semantic search
- Combine retrieval with LLM generation

See [../week1/README.md](../week1/README.md) for Week 1 content.

## Troubleshooting

**Jupyter Not Starting**:
```bash
# Verify Jupyter installed
uv run jupyter --version

# Launch from project root
uv run jupyter notebook
```

**API Authentication Errors**:
- Verify `.env` exists in project root
- Check API keys are valid (no extra spaces)
- Ensure `load_dotenv()` called in notebook

**Module Import Errors**:
- Run `uv sync` to install dependencies
- Restart Jupyter kernel after installing packages

## Related Documentation

- **Week 1**: [../week1/README.md](../week1/README.md) - RAG implementation
- **Project Root**: [../../README.md](../../README.md) - Overall architecture
- **API Backend**: [../../apps/api/README.md](../../apps/api/README.md) - Production API

## External Resources

- **OpenAI Docs**: https://platform.openai.com/docs/api-reference
- **Groq Docs**: https://console.groq.com/docs
- **Google GenAI Docs**: https://ai.google.dev/docs
- **Jupyter Docs**: https://docs.jupyter.org/
