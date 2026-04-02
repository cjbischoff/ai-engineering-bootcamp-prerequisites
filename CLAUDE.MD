# Project Claude Code Documentation

**Project:** AI Engineering Bootcamp Prerequisites
**Last Updated:** 2026-04-01
**Location:** `/Users/christopher/Development/_me/ai-engineering-bootcamp-prerequisites_me/CLAUDE.MD`

> **Global Config:** See `~/.claude/CLAUDE.md` for Claude Code installation, plugins, and MCP servers.

---

## 🧭 Documentation Routing Guide

**IMPORTANT:** This file contains essential project context. For detailed technical documentation, **you MUST read the corresponding `.docs/` file** based on the domain of the user's request.

### When to Read Sub-Documents

| User Request Domain | Read This File | Triggers |
|---------------------|---------------|----------|
| **Git operations, branching, commits, PRs** | [`.docs/git-workflow.md`](.docs/git-workflow.md) | "create branch", "commit", "PR", "merge", "push", "conventional commits", "sprint workflow", "CodeRabbit", "pre-commit", "pre-merge" |
| **Session startup, debugging, Docker issues** | [`.docs/claude-best-practices.md`](.docs/claude-best-practices.md) | "session startup", "debugging", "docker logs", "container not running", "health check", "smoke test", "import error", "connection refused", "hot reload" |
| **Vector search, Qdrant, dual vectors, RRF** | [`.docs/hybrid-search.md`](.docs/hybrid-search.md) | "hybrid search", "dense vector", "sparse vector", "BM25", "prefetch", "RRF", "fusion", "named vectors", "Qdrant collection", "embedding search" |
| **Reranking, Cohere, cross-encoders** | [`.docs/reranking.md`](.docs/reranking.md) | "reranking", "Cohere", "cross-encoder", "bi-encoder", "two-stage retrieval", "precision improvement", "rerank API" |
| **Prompts, YAML configs, Jinja2 templates** | [`.docs/prompt-management.md`](.docs/prompt-management.md) | "prompt", "YAML", "Jinja2", "template", "prompt versioning", "semantic versioning", "prompt config", "build_prompt", "LangSmith" |

### Routing Rules

**ALWAYS** follow these rules when helping with tasks:

1. **Read Before Acting**: If the user's request matches ANY trigger in the table above, **immediately read the corresponding `.docs/` file** using the Read tool before providing advice or making changes.

2. **Multiple Domains**: If the request spans multiple domains (e.g., "commit my hybrid search changes"), read ALL relevant sub-documents.

3. **Uncertainty**: If unsure whether a sub-document is relevant, **err on the side of reading it**. The files are concise and focused.

4. **Essential Context First**: Always read this main CLAUDE.md file first for project context, then read relevant sub-documents for detailed guidance.

5. **Examples**:
   - User: "How do I commit my changes?" → Read `.docs/git-workflow.md`
   - User: "The API container isn't starting" → Read `.docs/claude-best-practices.md`
   - User: "Explain how hybrid search works" → Read `.docs/hybrid-search.md`
   - User: "Update the RAG prompt" → Read `.docs/prompt-management.md`
   - User: "Commit my reranking implementation" → Read `.docs/git-workflow.md` AND `.docs/reranking.md`

---

## 📋 Project Overview

AI chatbot application stack with FastAPI backend and Streamlit frontend, featuring multi-provider LLM support (OpenAI, Groq, Google GenAI), Qdrant vector database, and RAG capabilities.

**Tech Stack:**
- **Backend:** FastAPI (Python 3.12+)
- **Frontend:** Streamlit
- **Package Manager:** uv (workspace architecture)
- **Vector DB:** Qdrant
- **Containerization:** Docker Compose
- **LLM Providers:** OpenAI, Groq, Google GenAI

---

## 🏗️ Architecture

```
ai-engineering-bootcamp-prerequisites_me/
├── apps/
│   ├── api/           # FastAPI backend service
│   └── chatbot_ui/    # Streamlit frontend
├── scripts/           # Test and debug utilities
│   ├── health_check.py    # Infrastructure health verification
│   └── smoke_test.py      # End-to-end RAG pipeline testing
├── data/              # Datasets and data files
├── notebooks/         # Jupyter notebooks for tutorials
├── qdrant_storage/    # Qdrant vector database storage
├── documentation/     # Project documentation
├── .venv/             # Python virtual environment
├── pyproject.toml     # uv workspace configuration
├── docker-compose.yml # Container orchestration
└── Makefile           # Common development commands
```

---

## 🚀 Common Commands

### Development Workflow

**Note:** All commands that need API keys should be prefixed with `op run --env-file=".env" --`

| Command | Purpose |
|---------|---------|
| `op run --env-file=".env" -- make run-docker-compose` | Sync dependencies and start all services |
| `op run --env-file=".env" -- make health` | Verify infrastructure health (containers, ports, collections) |
| `op run --env-file=".env" -- make health-silent` | Health check (only show failures) |
| `op run --env-file=".env" -- make smoke-test` | End-to-end **agent** test (`POST /agent/`, SSE `final_answer`) |
| `op run --env-file=".env" -- make smoke-test-verbose` | Same with full JSON payload |
| `make clean-notebook-outputs` | Clean Jupyter notebook outputs before commit (no secrets needed) |
| `op run --env-file=".env" -- make run-evals-retriever` | Run RAGAS evaluation metrics |

**Makefile + Docker:** `make run-docker-compose` runs `uv sync` and `docker compose up` under `op run --env-file="$(CURDIR)/.env" --` when the `op` CLI is on your `PATH`. Compose merges `env_file: .env` with `environment: ${VAR}` blocks so containers get **resolved** secrets from the host (see `docker-compose.yml` header comments). For host-only scripts (`make health`), `op` is optional unless your `.env` holds literal `op://` references and no plaintext fallback.

**Health vs smoke:** `make health` checks containers, **Qdrant `Amazon-items-collection-01-hybrid-search`**, `GET /openapi.json`, Postgres (if `psycopg` installed), and MCP ports. Use `make health-strict` to fail when Postgres cannot be checked. `make smoke-test` validates a real coordinator path through the API.

### Environment Setup

**This project uses 1Password for secure secret management.**

```bash
# 1. Install 1Password CLI (if not already installed)
brew install --cask 1password-cli

# 2. Authenticate with 1Password
op signin

# 3. The .env file already contains 1Password secret references
# No need to manually copy API keys!

# 4. Install dependencies
make install

# 5. Run commands with 1Password injection
op run --env-file=".env" -- make run-docker-compose
op run --env-file=".env" -- make health
op run --env-file=".env" -- make smoke-test
```

**Alternative: Manual API Keys (Not Recommended)**

If you prefer not to use 1Password:
```bash
cp env.example .env
# Edit .env with your actual API keys
# ⚠️  WARNING: Never commit .env with real keys!
```

### API Keys Required

All API keys are stored in 1Password Personal vault:

| Environment Variable | 1Password Item | Description |
|---------------------|----------------|-------------|
| `OPENAI_API_KEY` | `api_openai` | OpenAI API (embeddings & LLM) |
| `GOOGLE_API_KEY` | `api_google_fullaccess` | Google GenAI (alternative LLM) |
| `GROQ_API_KEY` | `api_groq_fullaccess` | Groq (fast inference) |
| `LANGSMITH_API_KEY` | `api_langsmith_fullaccess` | LangSmith (observability) |
| `COHERE_API_KEY` | `api_cohere_fullaccess` | Cohere (reranking) |
| `HF_TOKEN` | `api_huggingface_read` | HuggingFace (model downloads) |

---

## 🌿 Git Branching Strategy (AI Engineering Bootcamp)

**Quick Reference:**
- Branches: `sprint/1`, `sprint/2`, `sprint/3`
- One sprint = All videos (6-9 commits)
- Always sign commits (`-S` flag)
- Current Sprint: **Sprint 5** (In Progress)

**Essential Workflow:**
```bash
# Start sprint
git checkout -b sprint/5

# Per video: clean → comment → document → commit → push
make clean-notebook-outputs
# Add educational comments & documentation
git add .
git commit -S -m "feat(sprint5): complete video N - description"
git push origin sprint/5

# Merge: Pre-merge steps → PR → CodeRabbit → merge
gh pr create --base main --head sprint/5 --title "Sprint 5: Title"
gh pr merge sprint/5 --merge
```

**📘 Full Guide:** See [`.docs/git-workflow.md`](.docs/git-workflow.md) for complete workflow, commit conventions, and branch management rules


## 📐 Project Conventions

### Code Organization

- **Workspace Structure:** Uses `uv` workspace with `apps/` directory for modular applications
- **Backend:** FastAPI app in `apps/api/`
- **Frontend:** Streamlit app in `apps/client/`
- **Shared Code:** Cross-app utilities and models in workspace packages

### Python Conventions

- **Python Version:** 3.12+ (defined in `.python-version`)
- **Package Manager:** `uv` (not pip/poetry/pipenv)
- **Virtual Environment:** `.venv/` directory (managed by uv)
- **Dependencies:** Defined in `pyproject.toml` with workspace configuration

### Docker Conventions

- **Compose File:** `docker-compose.yml` defines all services
- **Environment Variables:** Loaded from `.env` file
- **API URL:** `http://api:8000` for container-to-container communication
- **Volumes:** Qdrant storage persisted in `./qdrant_storage`

### Security Rules

- ✅ **Use 1Password for secrets** - All API keys stored in 1Password Personal vault
- ✅ **1Password CLI injection** - Run commands with `op run --env-file=".env" --`
- ✅ **`.env` contains references only** - Format: `op://Personal/api_vendor_fullaccess/credential`
- ⚠️ **NEVER commit actual API keys** — `.env` is **gitignored**; keep it local even when it holds only `op://` references (those still reveal vault/item names).
- ⚠️ **Use `env.example`** - Template for required environment variables
- ⚠️ **API Keys in 1Password only** - Never hardcode in source files

**1Password Naming Convention:**
- Vault: `Personal`
- Item Type: `API Credential`
- Item Name: `api_vendor_permissions` (e.g., `api_openai`, `api_google_fullaccess`)
- Field: `credential`

---

## 🔧 Development Patterns

### Adding New LLM Provider

1. **Add API key to 1Password:**
   ```bash
   op item create \
     --category="API Credential" \
     --vault=Personal \
     --title="api_provider_fullaccess" \
     "credential[password]=your-api-key-here"
   ```

2. **Update `.env` with 1Password reference:**
   ```bash
   PROVIDER_API_KEY=op://Personal/api_provider_fullaccess/credential
   ```

3. **Update backend provider configuration** in `apps/api/`
4. **Add provider-specific client initialization**
5. **Update frontend provider selection UI** in `apps/client/`
6. **Add new API key to `env.example`** with documentation
7. **Document provider setup** in README.md

### Working with Vector Database

- **Qdrant Location:** Running in Docker container
- **Storage:** Persisted in `./qdrant_storage/`
- **Access:** Via Qdrant client in backend
- **Notebooks:** Example usage in `notebooks/`

### Notebook Development

- **Location:** `notebooks/` directory
- **Purpose:** Interactive tutorials, dataset exploration, RAG preprocessing
- **Topics:** LLM APIs, embeddings, vector search
- **Run From:** Project root with virtual environment active

### Prompt Configuration Management (Week 2 / Video 7)

**Pattern**: Externalize prompts to YAML files with Jinja2 templates for version control and easier collaboration.

**File Structure**:
```
apps/api/src/api/agents/
├── utils/
│   └── prompt_management.py       # Loading utilities
└── prompts/
    └── retrieval_generation.yaml  # Prompt configuration
```

**YAML Template Format**:
```yaml
metadata:
  name: Retrieval Generation Prompt
  version: 1.0.0
  description: Retrieval Generation Prompt for RAG Pipeline
  author: Your Name

prompts:
  retrieval_generation: |
    You are a shopping assistant...

    Context:
    {{ preprocessed_context }}

    Question:
    {{ question }}
```

**Usage in Code**:
```python
from api.agents.utils.prompt_management import prompt_template_config

def build_prompt(preprocessed_context, question):
    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )
    return template.render(
        preprocessed_context=preprocessed_context,
        question=question
    )
```

**Benefits**:
- ✅ **Separation of Concerns**: Prompts in YAML, logic in Python
- ✅ **Version Control**: Semantic versioning (1.0.0 → 1.1.0)
- ✅ **Collaboration**: Non-engineers can edit YAML files
- ✅ **Hot Reload**: YAML changes picked up by FastAPI without deployment
- ✅ **A/B Testing**: Load different prompts at runtime
- ✅ **Reduced LOC**: 60-line prompt function → 8 lines

**Jinja2 Variable Syntax**:
- `{{ variable_name }}` - Variable substitution
- `{% if condition %}...{% endif %}` - Conditionals (advanced)
- `{% for item in items %}...{% endfor %}` - Loops (advanced)

**File Paths**:
- **Local**: Relative to project root (e.g., `apps/api/src/...`)
- **Docker**: Same path works due to volume mount (`./apps/api/src:/app/apps/api/src`)

**Testing Prompts**:
```bash
# Smoke test validates end-to-end with template loading
make smoke-test

# Unit test for template loading
def test_prompt_template():
    template = prompt_template_config(yaml_file, key)
    prompt = template.render(preprocessed_context="...", question="...")
    assert "expected content" in prompt
```

**Versioning Best Practices**:
- **Patch** (1.0.0 → 1.0.1): Typo fixes, grammar corrections
- **Minor** (1.0.0 → 1.1.0): New instructions, improved clarity
- **Major** (1.0.0 → 2.0.0): Different output format, breaking changes

**Git Workflow**:
```bash
# 1. Edit YAML file
vim apps/api/src/api/agents/prompts/retrieval_generation.yaml

# 2. Update version in metadata (1.0.0 → 1.1.0)

# 3. Test changes
make smoke-test

# 4. Commit with descriptive message (signed)
git commit -S -m "feat(prompts): add rating emphasis to RAG prompt (v1.1.0)"

# 5. FastAPI hot reload picks up changes automatically
```

**Common Pitfalls**:
- ❌ **Wrong path**: Use `apps/api/src/...` (from project root), not `api/...`
- ❌ **Missing variables**: All template variables must be in `.render()`
- ❌ **YAML syntax**: Use `|` for multiline, check indentation
- ❌ **F-string syntax**: Use `{{ var }}` (Jinja2), not `{var}` (f-string)

**Future Enhancements**:
- LangSmith registry integration for cloud-based prompt management
- Template caching with `@lru_cache` for performance
- Multiple prompt variants (verbose, concise, reasoning)
- Conditional logic with Jinja2 (`{% if %}` blocks)

---

## 🎯 Claude Code Workflow

### Starting Work on This Project

```bash
# 1. Check git status and branch
git status && git branch

# 2. Ensure environment is set up
make install

# 3. Start services if needed
make up

# 4. Work on feature branch
git checkout -b feature/your-feature-name
```

### Before Committing

1. **Run Tests:** `make test`
2. **Check Linting:** Ensure code follows project conventions
3. **Verify Environment:** Never commit `.env` file
4. **Review Changes:** `git diff` before staging

### Project-Specific Reminders

- **Use uv, not pip:** All dependency management via `uv`
- **Docker for services:** Qdrant runs in container, not locally
- **Workspace structure:** Apps are separate packages in workspace
- **API keys required:** Most features need at least one LLM provider configured

---

## 🔍 Claude Code Best Practices

**Session Startup (ALWAYS):**
```bash
git status && git branch        # Check branch
make run-docker-compose         # Start services (foreground for logs)
make health                     # Verify infrastructure (new terminal)
```

**Quick Reference:**
- Monitor logs: `docker compose logs -f`
- Use service names (`http://qdrant:6333`), NOT `localhost`
- After dependencies: `uv lock` → `docker compose build api`
- Test scripts: `make health`, `make smoke-test`

**Common Errors:**
| Error | Solution |
|-------|----------|
| ModuleNotFoundError | Add to pyproject.toml, `uv lock`, rebuild |
| ConnectionRefusedError | Use service name, not localhost |
| ValidationError | Check Optional[] for nullable fields |

**📘 Full Guide:** See [`.docs/claude-best-practices.md`](.docs/claude-best-practices.md) for debugging, testing workflow, dependency management, and pitfalls


---

## 🔍 Hybrid Search Implementation (Week 2 / Video 5)

**Quick Reference:**
- Collection: `Amazon-items-collection-01-hybrid-search`
- Method: Dense (text-embedding-3-small) + Sparse (BM25) with RRF fusion
- Latency: ~115ms | Memory: ~7.4KB per product
- Recall improvement: Dense 70% → Hybrid 90%

**Key Concepts:**
- **Named Vectors**: Single collection stores dense + sparse vectors
- **Prefetch**: Retrieve top-20 from each method independently
- **RRF Fusion**: Rank-based merging (no manual normalization)
- **Document Wrapper**: `Document(text=desc, model="qdrant/bm25")` auto-computes BM25

**When to Use:**
- ✅ Queries with product codes, model numbers, technical terms
- ✅ Need exact keyword matching + semantic understanding
- ❌ All natural language queries (dense-only simpler)

**📘 Full Guide:** See [`.docs/hybrid-search.md`](.docs/hybrid-search.md) for complete implementation, patterns, pitfalls, and examples


---

## 🔍 Reranking with Cross-Encoders (Week 2 / Video 6)

**Quick Reference:**
- Provider: Cohere Rerank API (`rerank-v4.0-pro`)
- Two-stage: Hybrid search (115ms) → Rerank (500ms) = 616ms total
- Cost: $2.20 per 1K queries vs $0.20 for hybrid-only (10x increase)
- Precision improvement: Hybrid 70% → Hybrid+Rerank 95% (+25%)

**Architecture:**
1. **Stage 1**: Hybrid search retrieves top-20 candidates
2. **Stage 2**: Cross-encoder reranks for top-5 precision

**When to Use:**
- ✅ Precision critical (support, legal, medical)
- ✅ Budget allows ($2/1K queries)
- ❌ Need sub-200ms response times
- ❌ Cost-sensitive (<$0.50/1K queries)

**📘 Full Guide:** See [`.docs/reranking.md`](.docs/reranking.md) for bi-encoder vs cross-encoder, implementation patterns, and cost optimization


---

## 🔧 Prompt Configuration Management (Week 2 / Video 7)

**Quick Reference:**
- Refactor hardcoded prompts → YAML files with Jinja2 templates
- Files: `apps/api/src/api/agents/prompts/retrieval_generation.yaml`
- Reduction: 60-line function → 8-line function (-87%)

**YAML Format:**
```yaml
metadata:
  name: Retrieval Generation Prompt
  version: 1.0.0  # Semantic versioning

prompts:
  retrieval_generation: |
    You are a shopping assistant...
    Context: {{ preprocessed_context }}
    Question: {{ question }}
```

**Benefits:**
- ✅ Separation of concerns (prompts in YAML, logic in Python)
- ✅ Non-engineers can edit YAML files
- ✅ FastAPI hot reload (no redeploy needed)
- ✅ Semantic versioning (PATCH/MINOR/MAJOR)

**Git Workflow:**
```bash
vim apps/api/src/api/agents/prompts/retrieval_generation.yaml
# Update metadata version
make smoke-test
git commit -S -m "feat(prompts): add rating emphasis (v1.1.0)"
```

**📘 Full Guide:** See [`.docs/prompt-management.md`](.docs/prompt-management.md) for Jinja2 syntax, versioning, pitfalls, and LangSmith integration

---


---

## 📚 Related Documentation

### Project Documentation (.docs/)

**⚠️ IMPORTANT:** These files contain critical implementation details. **Use the Routing Guide at the top of this file** to determine which documents to read based on the user's request domain.

- [`.docs/git-workflow.md`](.docs/git-workflow.md) - Sprint branching, commits, PRs, CodeRabbit workflow
- [`.docs/claude-best-practices.md`](.docs/claude-best-practices.md) - Session startup, debugging, Docker, testing
- [`.docs/hybrid-search.md`](.docs/hybrid-search.md) - Dual vector architecture, RRF fusion, Qdrant
- [`.docs/reranking.md`](.docs/reranking.md) - Cross-encoder reranking, Cohere API
- [`.docs/prompt-management.md`](.docs/prompt-management.md) - YAML configs, Jinja2 templates, versioning

**Remember:** Don't just reference these files—**actively read them** when the user's request matches the domain triggers.

### Other Resources
- **Global Claude Config:** `~/.claude/CLAUDE.md`
- **Project README:** `./README.md`
- **Environment Template:** `./env.example`
- **Makefile:** `./Makefile` (common commands)
- **API Docs:** FastAPI auto-docs at `http://localhost:8000/docs` when running

---

## 🔄 Maintenance

**Update This File When:**
- Adding new services or components
- Changing development workflow
- Adding new conventions or patterns
- Discovering project-specific gotchas
- Updating dependencies or tech stack

**Keep Fresh:**
- Remove outdated patterns
- Update commands when Makefile changes
- Document architectural decisions
- Note common issues and solutions

---

**Last Review:** 2026-04-01
**Next Review:** After major architectural changes or monthly maintenance
