# Project Claude Code Documentation

**Project:** AI Engineering Bootcamp Prerequisites
**Last Updated:** 2026-01-24
**Location:** `/Users/christopher/Development/_me/ai-engineering-bootcamp-prerequisites_me/CLAUDE.MD`

> **Global Config:** See `~/.claude/CLAUDE.md` for Claude Code installation, plugins, and MCP servers.

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

| Command | Purpose |
|---------|---------|
| `make run-docker-compose` | Sync dependencies and start all services |
| `make health` | Verify infrastructure health (containers, ports, collections) |
| `make health-silent` | Health check (only show failures) |
| `make smoke-test` | Run end-to-end RAG pipeline test |
| `make smoke-test-verbose` | Smoke test with full JSON response |
| `make clean-notebook-outputs` | Clean Jupyter notebook outputs before commit |
| `make run-evals-retriever` | Run RAGAS evaluation metrics |

### Environment Setup

```bash
# Initial setup
cp env.example .env
# Edit .env with your API keys

# Install dependencies
make install

# Start services
make up
```

### API Keys Required

- `OPENAI_KEY` - OpenAI API (optional, quota may be exceeded)
- `GOOGLE_API_KEY` - Google GenAI (recommended)
- `GROQ_API_KEY` - Groq API (recommended)

---

## 🌿 Git Branching Strategy (AI Engineering Bootcamp)

### Sprint-Based Development

This project follows a **sprint-based branching strategy** for bootcamp capstone submissions.

**Branch Structure:**
- Branch naming: `sprint/1`, `sprint/2`, `sprint/3`
- One sprint = All videos in that sprint (typically 6-9 videos)
- Each video gets its own commit

**Sprint Lifecycle:**
1. Create sprint branch from `main`
2. Complete videos, commit and push each one
3. Create Pull Request via GitHub CLI
4. CodeRabbit reviews PR
5. Merge PR via GitHub CLI
6. Sprint branch **remains in GitHub permanently** (checkpoint for reviewers)

---

### Commit Message Convention

**Format:** Conventional Commits (never reference Claude/Cursor)

**Video completion commits:**
```bash
feat(sprint2): complete video 1 - agent basics
feat(sprint2): complete video 3 - langraph implementation
feat(sprint2): complete video 7 - multi-agent systems
```

**Other commits during sprint:**
```bash
fix(sprint2): correct validation error in agent pipeline
refactor(sprint2): optimize agent orchestration logic
docs(sprint2): add agent architecture documentation
test(sprint2): add unit tests for agent tools
```

**Conventional commit types:**
- `feat` - New feature or video completion
- `fix` - Bug fix or correction
- `refactor` - Code restructuring (no functionality change)
- `docs` - Documentation only
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

---

### Complete CLI Workflow

#### Starting a Sprint

```bash
# 1. Ensure you're on main with latest changes
git checkout main
git pull origin main

# 2. Create sprint branch
git checkout -b sprint/2
```

#### During Sprint (Per Video)

```bash
# 1. Complete video work
# ... make your changes ...

# 2. Stage changes
git add .

# 3. Commit with conventional format
git commit -m "feat(sprint2): complete video 3 - langraph implementation"

# 4. Push to GitHub (backup + visibility)
git push origin sprint/2

# Repeat for each video (6-9 times)
```

#### Completing Sprint

```bash
# 1. Create Pull Request via GitHub CLI
gh pr create \
  --base main \
  --head sprint/2 \
  --title "Sprint 2: Agents & Agentic Systems" \
  --body "Completed all videos for Sprint 2. Ready for review."

# 2. Check PR status (wait for CodeRabbit review)
gh pr status

# 3. View CodeRabbit feedback
gh pr view sprint/2

# 4. After approval, merge via CLI
gh pr merge sprint/2 --merge

# 5. Update local main
git checkout main
git pull origin main

# sprint/2 branch stays in GitHub permanently for review
```

---

### Branch Management Rules

**✅ DO:**
- Create all sprint branches from `main`
- Push after each video commit (backup protection)
- Use conventional commit format
- Include hotfixes in sprint branch (document in commit message)
- Keep sprint branches in GitHub permanently
- Merge only via Pull Request

**❌ DON'T:**
- Commit directly to `main`
- Delete sprint branches after merge
- Reference Claude/Cursor in commit messages
- Merge without PR review
- Create sprint branches from other sprint branches

---

### Current Sprint

**Sprint 2: Agents & Agentic Systems**
- Branch: `sprint/2`
- Videos: Agent basics through multi-agent orchestration
- Status: In progress

---

### GitHub CLI Setup

**Install GitHub CLI (if not already):**
```bash
# macOS
brew install gh

# Verify installation
gh --version

# Authenticate
gh auth login
```

**Useful commands:**
```bash
gh pr status              # Check PR status
gh pr view sprint/2       # View specific PR
gh pr list                # List all PRs
gh pr checks sprint/2     # View CI/review checks
```

---

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

- ⚠️ **NEVER commit `.env` file** - Contains API keys
- ⚠️ **Use `env.example`** - Template for required environment variables
- ⚠️ **API Keys in `.env` only** - Never hardcode in source files

---

## 🔧 Development Patterns

### Adding New LLM Provider

1. Update backend provider configuration in `apps/api/`
2. Add provider-specific client initialization
3. Update frontend provider selection UI in `apps/client/`
4. Add new API key to `env.example` and `.env`
5. Document provider setup in README.md

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

# 4. Commit with descriptive message
git commit -m "feat(prompts): add rating emphasis to RAG prompt (v1.1.0)"

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

### Session Startup Workflow

**ALWAYS start each Claude Code session with:**

```bash
# 1. Check git branch and status
git status && git branch

# 2. Start Docker Compose services in foreground
make run-docker-compose
# OR for background with logs accessible:
docker compose up -d && docker compose logs -f

# 3. Verify infrastructure health (in new terminal)
make health
```

**Why this matters:**
- **Live Debugging**: Watch API logs in real-time as you make code changes
- **Hot Reload Visibility**: See when FastAPI reloads after file changes
- **Error Detection**: Catch runtime errors immediately (import errors, validation errors, etc.)
- **Request Flow**: Trace API requests from client through middleware to pipeline
- **Performance Monitoring**: Observe response times and identify bottlenecks
- **Health Verification**: Confirm all services are running before starting development

### Debugging Docker-Based Applications

**When debugging issues in this project:**

1. **Monitor Logs Continuously**
   ```bash
   # Watch all services
   docker compose logs -f

   # Watch specific service
   docker compose logs -f api
   docker compose logs -f qdrant
   ```

2. **Check Container Status**
   ```bash
   docker compose ps
   # Should show: api (running), client (running), qdrant (running)
   ```

3. **Verify Service Networking**
   - **Service Names**: Use `http://qdrant:6333`, NOT `http://localhost:6333`
   - **Why**: Localhost in container = container itself, not other services
   - **Test**: `docker compose exec api ping qdrant` should succeed

4. **Rebuild After Dependency Changes**
   ```bash
   # When pyproject.toml changes (new dependencies added)
   uv lock                    # Update uv.lock file
   docker compose build api   # Rebuild API container with new deps
   docker compose up -d       # Restart services
   ```

5. **Common Error Patterns**
   - **ModuleNotFoundError**: Missing dependency in `pyproject.toml` → run `uv lock` + rebuild
   - **ConnectionRefusedError to localhost**: Using localhost instead of service name
   - **ValidationError**: Pydantic model mismatch → check Optional fields for nullable data
   - **KeyError in response**: Missing field in return dict → verify function return structure

### Test Scripts for Verification

The project includes Python-based test scripts for infrastructure verification and end-to-end testing. These scripts use `uv run` and integrate with the Makefile for easy invocation.

#### Health Check Script ([scripts/health_check.py](scripts/health_check.py))

**Purpose**: Verify infrastructure is ready before development

**Usage**:
```bash
make health              # Full output with colored checkmarks
make health-silent       # Only show failures (for CI/scripts)
```

**What it checks**:
- ✓ Docker containers running (api, streamlit-app, qdrant)
- ✓ Ports listening (8000, 8501, 6333, 6334)
- ✓ Qdrant collection exists and has documents
- ✓ API is responding

**When to use**:
- **Session startup**: ALWAYS run after `make run-docker-compose`
- **After service restarts**: Verify everything came back up correctly
- **When debugging**: Quickly identify which component is failing
- **Before making changes**: Ensure starting from a healthy state

**Exit codes**: Returns 0 if all checks pass, 1 if any fail (useful for CI/scripts)

#### Smoke Test Script ([scripts/smoke_test.py](scripts/smoke_test.py))

**Purpose**: End-to-end validation of the RAG pipeline

**Usage**:
```bash
make smoke-test          # Summary output with test results
make smoke-test-verbose  # Full JSON response included
```

**What it tests**:
- ✓ RAG API endpoint responds with status 200
- ✓ Response is valid JSON
- ✓ Response structure matches Pydantic models (RAGResponse schema)
- ✓ Response time is acceptable (< 20 seconds for cold start)
- ✓ LLM answer is generated (non-empty)
- ✓ Product context includes enriched metadata (images, prices, descriptions)

**When to use**:
- **After RAG changes**: Modified retrieval_generation.py, models.py, or endpoints.py
- **Before committing**: Verify your changes didn't break the pipeline
- **After dependency updates**: Ensure new package versions are compatible
- **When debugging quality issues**: Verify response structure and content

**Test query**: "best wireless headphones under $100" (can be customized with `--query` flag)

**Performance note**: First query may take 10-15 seconds due to:
- OpenAI embedding model initialization
- Qdrant client connection
- LLM cold start

#### Recommended Testing Workflow

```bash
# 1. Start session and verify health
make run-docker-compose  # Terminal 1: Watch logs
make health              # Terminal 2: Verify infrastructure

# 2. Make your code changes
# ... edit files ...

# 3. Test changes (hot reload should pick them up)
make smoke-test          # Verify end-to-end functionality

# 4. If tests pass, commit
git add .
git commit -m "Your descriptive commit message"
```

#### Script Implementation Details

- **Language**: Python 3.12+ (uses `uv run` for execution)
- **Dependencies**: Uses existing project dependencies (requests, qdrant-client)
- **Output**: ANSI colored terminal output (green ✓, red ✗, yellow ⚠)
- **Integration**: Makefile targets auto-run `uv sync` before script execution
- **Exit codes**: 0=success, 1=failure (suitable for CI/CD pipelines)

### Code Development Best Practices

**Before Making Changes:**

1. **Read Files First**: ALWAYS read files before editing
   ```python
   # Bad: Edit without reading
   Edit(file_path="...", old_string="...", new_string="...")

   # Good: Read, understand, then edit
   Read(file_path="...")
   # ... analyze structure ...
   Edit(file_path="...", ...)
   ```

2. **Check Imports**: Verify import paths match project structure
   ```python
   # Bad: apps.api.src.api.models (includes src)
   from apps.api.src.api.models import RAGResponse

   # Good: api.api.models (src is implicit in PYTHONPATH)
   from api.api.models import RAGResponse
   ```

3. **Test in Increments**: Make small changes, test, iterate
   - Change one function → watch logs → verify behavior
   - Don't make multiple large changes without testing

**After Making Changes:**

1. **Watch for Hot Reload**
   ```
   INFO:     Watching for file changes...
   INFO:     Application startup complete.
   ```

2. **Test with curl or Frontend**
   ```bash
   # Test API endpoint
   curl -X POST http://localhost:8000/rag/ \
     -H "Content-Type: application/json" \
     -d '{"query": "best wireless headphones"}'
   ```

3. **Verify Response Structure**
   - Check for required fields (request_id, answer, used_context)
   - Validate nested objects match Pydantic models
   - Confirm nullable fields handle None gracefully

### Dependency Management

**Adding New Dependencies:**

1. Add to `apps/api/pyproject.toml`:
   ```toml
   dependencies = [
       "instructor>=1.0.0",  # Example: new dependency
       ...
   ]
   ```

2. Update lock file:
   ```bash
   uv lock
   ```

3. Rebuild Docker image:
   ```bash
   docker compose build api
   docker compose up -d
   ```

4. **DO NOT** skip `uv lock` - Docker uses frozen lockfile for reproducibility

### Common Pitfalls and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Import Errors** | `ModuleNotFoundError` in logs | Missing dependency → add to `pyproject.toml`, run `uv lock`, rebuild |
| **Pydantic Validation** | `ValidationError: field - Input should be...` | Use `Optional[]` for nullable fields, check `.get()` for dict access |
| **Instructor Errors** | `KeyError` on expected fields | Add `response_model=YourModel` to `create_with_completion()` |
| **Qdrant Connection** | `ConnectionRefusedError [Errno 111]` | Use service name `http://qdrant:6333`, not localhost |
| **Hot Reload Not Working** | Changes don't appear | Check volume mount in docker-compose.yml, restart container |
| **Syntax Errors** | `SyntaxError` on startup | Check import statements (`from X import Y`, not `import X import Y`) |

### File Structure Awareness

**When navigating codebase:**

- **API Code**: `apps/api/src/api/` (not `apps/api/api/`)
  - `app.py` - FastAPI app initialization
  - `api/endpoints.py` - Route handlers
  - `api/models.py` - Pydantic schemas
  - `api/middleware.py` - Custom middleware
  - `agents/retrieval_generation.py` - RAG pipeline

- **Import Paths**: Use `from api.X import Y` (src is in PYTHONPATH)
- **Volume Mounts**: Only `src/` is mounted → changes outside `src/` need rebuild

### Testing Strategy

1. **Infrastructure Health Checks**: Verify services are running
   - **Tool**: `make health` ([scripts/health_check.py](scripts/health_check.py))
   - **When**: Session startup, after restarts, before making changes
   - **Checks**: Docker containers, ports, Qdrant collection, API connectivity
   - **Fast**: < 5 seconds, no LLM calls

2. **Smoke Testing**: End-to-end RAG pipeline validation
   - **Tool**: `make smoke-test` ([scripts/smoke_test.py](scripts/smoke_test.py))
   - **When**: After code changes, before commits, after dependency updates
   - **Tests**: API response, JSON structure, Pydantic models, response time, product enrichment
   - **Real query**: Uses actual LLM and Qdrant (10-15 seconds)

3. **Unit Testing**: Test individual functions in isolation
   - Mock Qdrant client for RAG pipeline tests
   - Verify Pydantic model validation edge cases
   - Test helper functions without external dependencies

4. **Integration Testing**: Test API endpoints end-to-end
   - Ensure Docker services are running (`make health` first)
   - Use real Qdrant instance (test collection)
   - Verify response structure matches OpenAPI schema

5. **Manual Testing**: Use curl or Streamlit frontend
   - Check logs for errors and performance
   - Verify enriched responses include images/prices
   - Test with queries that might return partial data (missing images)

### Documentation Hygiene

**Keep these updated:**

- `README.md` - After each major feature/sprint
- `CLAUDE.md` - When discovering new patterns or gotchas
- Code comments - Explain WHY not WHAT (especially for non-obvious decisions)
- OpenAPI docs - Pydantic Field descriptions auto-generate docs

**Update triggers:**

- New dependencies added
- Architectural patterns change
- Common errors discovered and solved
- Docker configuration modified

---

## 🔍 Hybrid Search Implementation (Week 2 / Video 5)

### Overview

**Hybrid search** combines dense (semantic) and sparse (keyword/BM25) retrieval for more robust search quality.

**Location:** `notebooks/week2/03-Hybrid-Search.ipynb`

**Collection:** `Amazon-items-collection-01-hybrid-search`

### Key Concepts for AI Assistants

#### 1. Dual Vector Architecture

**Named Vectors in Qdrant:**
- Single collection can store multiple vector types per point
- Each vector has its own index and search strategy
- Payload metadata is shared across all vectors

**Configuration Pattern:**
```python
vectors_config={
    "text-embedding-3-small": VectorParams(size=1536, distance=Distance.COSINE)
},
sparse_vectors_config={
    "bm25": SparseVectorParams(modifier=models.Modifier.IDF)
}
```

#### 2. Prefetch Mechanism

**What It Does:**
- Retrieves top-N candidates from EACH search method independently
- Runs searches in parallel (or could be parallelized)
- Provides broader candidate pool for fusion algorithm

**Pattern:**
```python
prefetch=[
    Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
    Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20)
]
```

**Key Parameter: limit**
- Set higher than final result count (e.g., 20 vs 5)
- More candidates = better fusion quality, but slower
- Sweet spot: 3-5x the final limit (e.g., limit=20 for top_k=5)

#### 3. RRF (Reciprocal Rank Fusion)

**Algorithm:**
- Merges ranked lists using rank positions (not raw scores)
- Formula: `RRF_score = Σ (1 / (k + rank_i))` where k=60
- Scale-independent: No manual normalization needed

**Usage:**
```python
query=FusionQuery(fusion="rrf")
```

**Why RRF:**
- Dense scores (~0.85) and sparse scores (~127.3) can't be directly combined
- Rank-based approach avoids normalization problems
- Products ranked highly in BOTH methods score best
- Research-proven standard (TREC competitions)

#### 4. Document Wrapper for BM25

**Pattern:**
```python
vector={
    "text-embedding-3-small": embedding,  # Pre-computed dense vector
    "bm25": Document(text=description, model="qdrant/bm25")  # Auto BM25
}
```

**What Document Wrapper Does:**
- Qdrant automatically computes BM25 sparse vector from text
- Handles tokenization, TF (term frequency), IDF (inverse document frequency)
- IDF weights update dynamically as collection grows
- No manual BM25 implementation needed

**Alternative (Manual BM25) - Avoid:**
```python
# Complex: requires manual tokenization, TF-IDF calculation
bm25_vector = {"usb": 2.1, "cable": 1.8, "type": 1.2}
```

#### 5. Point Structure with Dual Vectors

```python
PointStruct(
    id=i,
    vector={
        "text-embedding-3-small": embedding,  # Dense: 1536 floats
        "bm25": Document(text=description, ...)  # Sparse: automatic
    },
    payload=data
)
```

**Key Insights:**
- Vector is a **dictionary** of named vectors (not single vector)
- Each named vector uses its own index type (HNSW for dense, inverted for sparse)
- Payload stores complete product metadata (no second query needed)

### Common Patterns

#### Hybrid Retrieval Function

```python
def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
            Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20)
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )

    # Extract and return results
    return results
```

**Query Flow:**
1. Generate query embedding (OpenAI API ~100ms)
2. Dense prefetch (HNSW index <10ms)
3. Sparse prefetch (inverted index + BM25 <5ms)
4. RRF fusion (<1ms)
5. Return top-k results

**Total latency: ~115ms** (OpenAI API is bottleneck)

### When to Use Hybrid vs Dense-Only

**Use Hybrid Search When:**
- Queries include product codes, model numbers, technical terms
- Need exact keyword matching alongside semantic understanding
- Handling diverse query types (keywords + descriptions)
- Recall is critical (hybrid has ~20% higher recall than dense-only)

**Use Dense-Only When:**
- All queries are natural language descriptions
- No product codes or technical terms in queries
- Simplicity is preferred over marginal quality gain
- Latency is extremely critical (hybrid adds ~15ms)

**Use Sparse-Only When:**
- Working with structured data (IDs, codes, exact matches)
- No semantic understanding needed
- Lowest latency required (<5ms retrieval)

### Performance Characteristics

**Memory per Product:**
- Dense vector: 1536 floats × 4 bytes = 6KB
- Sparse vector: ~100 terms × 8 bytes = 800 bytes
- Payload: ~500 bytes
- **Total: ~7.4 KB per product**

**Scaling:**
- 1,000 products: ~9 MB (fits in RAM easily)
- 1,000,000 products: ~9 GB (requires decent server)

**Query Performance:**
- Dense search: O(log N) with HNSW
- Sparse search: O(T × log N) where T = query terms
- Fusion: O(K1 + K2) where K = prefetch limits
- **Scales to millions of products**

### Common Pitfalls

#### Pitfall 1: Forgetting Document Wrapper

**Wrong:**
```python
vector={"bm25": description}  # String, not Document
```

**Right:**
```python
vector={"bm25": Document(text=description, model="qdrant/bm25")}
```

#### Pitfall 2: Prefetch Limit Too Low

**Problem:** Prefetch limit=5, final limit=5 → No room for fusion to improve ranking
**Solution:** Use prefetch limit 3-5x higher than final limit (e.g., 20 vs 5)

#### Pitfall 3: Mixing Score-Based and Rank-Based Fusion

**Problem:** Trying to add dense scores + sparse scores directly
**Solution:** Always use RRF for hybrid search (rank-based, scale-independent)

#### Pitfall 4: Not Using Named Vectors

**Wrong:**
```python
# Trying to store two separate vectors
vector=embedding  # Only stores dense vector
```

**Right:**
```python
# Named vectors dictionary
vector={
    "dense": embedding,
    "sparse": Document(...)
}
```

### Testing Hybrid Search

**Comparison Queries:**
1. **Product Code**: "B0C142QS8X" (should rank exact match #1)
2. **Semantic**: "waterproof headphones" (should find "water-resistant")
3. **Hybrid**: "Sony WH-1000XM4 wireless" (model + feature)

**Quality Metrics:**
- **Recall@K**: % of relevant products in top-K
- **Precision@K**: % of top-K that are relevant
- **MRR**: Position of first relevant result

**Expected Improvement:**
- Dense-only: Recall@5 ~70%
- Hybrid: Recall@5 ~90% (significant gain)

### Integration with RAG Pipeline

**Drop-in Replacement:**
- Same function interface as Week 1 `retrieve_data()`
- Returns same data structure
- Can swap into existing RAG pipeline without code changes
- Only change: collection name to hybrid search collection

**Next Steps:**
- Update FastAPI endpoint to use hybrid collection
- A/B test hybrid vs dense-only
- Measure impact on RAG answer quality (RAGAS metrics)

### Cost Analysis

**Embedding Costs (1000 products):**
- OpenAI text-embedding-3-small: $0.020 / 1M tokens
- Average description: ~200 tokens
- Total: 200K tokens × $0.020 / 1M = **$0.004** (less than 1 cent)

**Query Costs:**
- Per query: ~10 tokens × $0.020 / 1M = **$0.0000002** (negligible)
- 1 million queries: **$0.20**

**Infrastructure:**
- Self-hosted Qdrant (Docker): Free
- Qdrant Cloud: $25/month (1M vectors)

**Total Monthly (10K queries):** $0-$25

### Key Learnings for AI Assistants

1. **Named Vectors Are Fundamental**: Qdrant's named vector support enables hybrid search
2. **Prefetch Is Not Optional**: Can't do hybrid search without prefetch mechanism
3. **RRF Is Simple Yet Powerful**: No manual tuning, works across score ranges
4. **Document Wrapper Simplifies BM25**: Let Qdrant handle sparse vector computation
5. **Hybrid Adds Minimal Latency**: ~15ms extra for significant quality improvement
6. **Memory Overhead Is Reasonable**: ~1KB sparse vector per 6KB dense vector
7. **Drop-In Replacement**: Hybrid can replace dense-only with minimal code changes

---

## 🔍 Reranking with Cross-Encoders (Week 2 / Video 6)

### Overview

**Reranking** implements two-stage retrieval to improve search precision using cross-encoder models.

**Location:** `notebooks/week2/04-Reranking.ipynb`

**Provider:** Cohere Rerank API (`rerank-v4.0-pro`)

### Key Concepts for AI Assistants

#### 1. Two-Stage Retrieval Architecture

**Stage 1: Hybrid Search (Bi-Encoder)**
- Fast initial retrieval with broad candidate set
- Combines dense + sparse vectors with RRF fusion
- Returns top-20 candidates (~115ms)
- Good recall (~90%), moderate precision (~70%)

**Stage 2: Reranking (Cross-Encoder)**
- Slower but more accurate refinement
- Cohere rerank-v4.0-pro model
- Returns top-5-20 reordered results (~500ms)
- Excellent precision (~95%)

**Complete Pipeline:**
```
User Query
    ↓
Stage 1: Hybrid Search
  - Dense: text-embedding-3-small (semantic)
  - Sparse: BM25 (keyword matching)
  - Fusion: RRF (Reciprocal Rank Fusion)
  - Result: Top 20 candidates (~115ms)
    ↓
Stage 2: Reranking
  - Model: Cohere rerank-v4.0-pro
  - Input: Query + Top 20 documents
  - Output: Reordered results with relevance scores
  - Result: Top 5-20 best matches (~500ms)
    ↓
Final Results (Highly Relevant)
```

#### 2. Bi-Encoder vs Cross-Encoder

**Bi-Encoder (Retrieval Model):**
```
Query → Encoder → [0.1, 0.5, 0.8, ...]
Document → Encoder → [0.2, 0.4, 0.9, ...]
Similarity = dot_product(query_vec, doc_vec)
```
- ✅ Fast: Pre-computed document embeddings
- ✅ Scalable: Millions of documents in milliseconds
- ❌ Limited accuracy: No query-document interaction

**Cross-Encoder (Reranking Model):**
```
[Query, Document] → Encoder → Relevance Score (0-1)
```
- ✅ High accuracy: Full attention between query and document
- ✅ Better semantic understanding
- ❌ Slow: Must re-encode every query-document pair
- ❌ Not scalable: Can't pre-compute, must run on-demand

**Why Cross-Encoders Are More Accurate:**
- Full attention between query and document tokens
- Can identify nuanced semantic relationships
- Better at understanding multi-constraint queries
- Corrects errors from initial retrieval stage

#### 3. Cohere Rerank API Integration

**Client Initialization:**
```python
import cohere
cohere_client = cohere.ClientV2()  # Requires COHERE_API_KEY in environment
```

**Reranking Call:**
```python
response = cohere_client.rerank(
    model="rerank-v4.0-pro",       # Latest production reranker
    query=query,                    # User query string
    documents=to_rerank,            # List of candidate documents from Stage 1
    top_n=20,                       # Return top N reordered results
)
```

**Response Structure:**
```python
response.results = [
    {"index": 5, "relevance_score": 0.95},   # Original index=5 now ranked #1
    {"index": 2, "relevance_score": 0.87},   # Original index=2 now ranked #2
    {"index": 10, "relevance_score": 0.78},  # Original index=10 now ranked #3
    ...
]
```

**Reconstructing Reranked Results:**
```python
reranked_results = [to_rerank[result.index] for result in response.results]
```

#### 4. Performance and Cost Analysis

**Latency Breakdown:**

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Query embedding | ~100ms | 100ms |
| Dense prefetch | <10ms | 110ms |
| Sparse prefetch | <5ms | 115ms |
| RRF fusion | <1ms | 116ms |
| **Reranking (20 docs)** | **~500ms** | **~616ms** |

**Cost Analysis (1000 queries/day, 30 days):**

| Component | Cost per Query | Monthly Cost |
|-----------|---------------|--------------|
| OpenAI embeddings | $0.0002 | $6 |
| Cohere reranking | $0.002 | $60 |
| **Total** | **$0.0022** | **$66** |

**Key Insight:** Reranking dominates both latency (500ms of 616ms) and cost ($60 of $66)

#### 5. When to Use Reranking

**✅ Use Reranking When:**
- Precision is critical (customer support, legal, medical)
- Small final result set needed (top 5-10)
- Have budget for API costs ($2 per 1K queries)
- Latency budget allows ~500ms overhead
- Quality improvements justify 10x cost increase

**❌ Skip Reranking When:**
- Need sub-200ms response times
- Large result sets required (50+ results)
- Cost-sensitive application (<$0.50 per 1K queries)
- Hybrid search already provides sufficient precision
- High volume use case (millions of queries/day)

#### 6. Comparison of Approaches

| Approach | Latency | Cost/1K Queries | Precision | Recall | Best For |
|----------|---------|-----------------|-----------|--------|----------|
| **Dense only** | 50ms | $0.20 | 60% | 70% | High volume, cost-sensitive |
| **Hybrid** | 115ms | $0.20 | 70% | 90% | General purpose, balanced |
| **Hybrid + Rerank** | 616ms | $2.20 | 95% | 90% | High precision, low volume |

**Quality Improvement:**
- Dense-only → Hybrid: +10% precision, +20% recall
- Hybrid → Hybrid+Rerank: +25% precision, same recall
- Dense-only → Hybrid+Rerank: +35% precision, +20% recall

**Cost-Benefit Analysis:**
- Extra cost: $2/1K queries (10x increase)
- Extra latency: 500ms (5x increase)
- Precision gain: +25% (70% → 95%)
- **Decision: Use case dependent** (customer support = yes, search autocomplete = no)

#### 7. Implementation Pattern

**Retrieval with Reranking Support:**
```python
def retrieve_data(query, qdrant_client, k=20):
    """Stage 1: Retrieve k=20 candidates for reranking"""
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
            Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20)
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k  # k=20 for reranking (not final k=5)
    )

    return {
        "retrieved_context": [result.payload["description"] for result in results.points],
        "retrieved_context_ids": [result.payload["parent_asin"] for result in results.points],
        ...
    }
```

**Why k=20 for Reranking:**
- Too few (k=5): Reranker has limited options, can't improve much
- Too many (k=50): Slower reranking, more API cost, diminishing returns
- Sweet spot (k=20): Good diversity for reranker to optimize

**Reranking Stage:**
```python
# Stage 1: Hybrid search
results = retrieve_data(query, qdrant_client, k=20)
to_rerank = results["retrieved_context"]

# Stage 2: Rerank
response = cohere_client.rerank(
    model="rerank-v4.0-pro",
    query=query,
    documents=to_rerank,
    top_n=20  # Could set to 5 for final top-5
)

# Reconstruct in new order
reranked_results = [to_rerank[result.index] for result in response.results]
```

#### 8. Integration with RAG Pipeline

**Drop-in Enhancement:**
- Reranking added as optional stage after hybrid search
- Same data structure, just reordered
- Can be toggled with feature flag
- Minimal code changes required

**RAG Pipeline with Optional Reranking:**
```python
def rag_pipeline(question, top_k=5, use_reranking=False):
    qdrant_client = QdrantClient(url="http://localhost:6333")

    # Stage 1: Hybrid search (get more if reranking)
    k = 20 if use_reranking else top_k
    retrieved_context = retrieve_data(question, qdrant_client, k)

    # Stage 2: Optional reranking
    if use_reranking:
        reranked = cohere_client.rerank(
            query=question,
            documents=retrieved_context["retrieved_context"],
            top_n=top_k
        )
        # Reorder context using reranked indices
        context = [retrieved_context["retrieved_context"][r.index] for r in reranked.results]
    else:
        context = retrieved_context["retrieved_context"][:top_k]

    # Stage 3: LLM generation
    preprocessed_context = process_context(context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    return answer
```

#### 9. Common Patterns and Pitfalls

**Pattern: Prefetch Limit for Reranking**
```python
# Good: Higher prefetch limit for reranking
prefetch_limit = 20
final_limit = 20  # All prefetch results go to reranker

# Bad: Same prefetch and final limit
prefetch_limit = 5
final_limit = 5  # Reranker has no room to improve
```

**Pitfall: Forgetting to Install Cohere SDK**
```bash
# Add to pyproject.toml
uv add cohere>=5.11.4

# Or install directly
pip install cohere
```

**Pitfall: Missing COHERE_API_KEY**
```env
# Add to .env file
COHERE_API_KEY=your_api_key_here
```

**Pattern: Graceful Degradation**
```python
try:
    # Try reranking
    reranked = cohere_client.rerank(...)
except Exception as e:
    logger.warning(f"Reranking failed, using hybrid search results: {e}")
    # Fall back to hybrid search results
    reranked_results = to_rerank[:top_k]
```

#### 10. Production Considerations

**Cost Optimization:**
1. **Reduce top_n**: Rerank top 10 instead of top 20 (50% savings)
2. **Selective reranking**: Only rerank low-confidence queries
3. **Caching**: Cache reranked results for repeated queries
4. **Free alternatives**: Self-host reranker (bge-reranker-v2-m3)

**Latency Optimization:**
1. **Async reranking**: Don't block main thread
2. **Batch requests**: Rerank multiple queries together
3. **Cache popular queries**: Skip reranking for cached results
4. **Parallel Stage 1**: Run hybrid search while user types

**Quality Monitoring:**
1. Track reranking impact on RAGAS metrics
2. A/B test reranked vs non-reranked results
3. Monitor for model drift over time
4. Analyze failure cases where reranking didn't help

**Alternative Reranking Models:**

| Model | Cost | Latency | Accuracy | Deployment |
|-------|------|---------|----------|------------|
| **Cohere rerank-v4.0-pro** | $2/1K | ~500ms | Excellent | API (no infra) |
| **bge-reranker-v2-m3** | Free | ~200ms | Good | Self-host (GPU) |
| **GPT-4 as reranker** | $100/1K | ~2s | Good | API (expensive) |

### Key Learnings for AI Assistants

1. **Two-Stage is Critical**: Can't scale cross-encoders to full corpus, need bi-encoder first
2. **Prefetch Size Matters**: k=20 for prefetch gives reranker options (not k=5)
3. **Cost Dominates Latency**: $60/mo reranking vs $6/mo embedding for 30K queries
4. **Precision vs Speed Trade-off**: 6x slower for 25% precision improvement
5. **Use Case Dependent**: High-value queries justify 10x cost increase
6. **Drop-in Enhancement**: Can be added to existing pipeline with minimal changes
7. **Graceful Degradation**: Always have fallback to hybrid search if reranking fails

---

## 🔧 Prompt Configuration Management (Week 2 / Video 7)

### Overview

**Prompt Configuration Management** refactors hardcoded prompts into externalized YAML files with Jinja2 templates, enabling version control, A/B testing, and cleaner separation of concerns.

**Location:** `notebooks/week2/05-Prompt-Versioning.ipynb`

**New Files Created:**
- `apps/api/src/api/agents/utils/prompt_management.py` - Loading utilities
- `apps/api/src/api/agents/prompts/retrieval_generation.yaml` - YAML configuration
- `notebooks/week2/prompts/retrieval_generation.yaml` - Learning copy

### Key Concepts for AI Assistants

#### 1. The Problem with Hardcoded Prompts

**Before (Hardcoded in retrieval_generation.py)**:
```python
def build_prompt(preprocessed_context, question):
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
[... 60+ lines of hardcoded prompt text ...]

Context:
{preprocessed_context}

Question:
{question}
"""
    return prompt
```

**Problems:**
- ❌ 60+ lines of prompt text embedded in Python code
- ❌ No version control for prompts (lost in Git noise)
- ❌ Prompt changes require code deployment
- ❌ Hard for non-engineers to edit prompts
- ❌ No metadata (version, author, description)
- ❌ Can't A/B test prompts at runtime

#### 2. The Solution: YAML Configuration Files

**File Structure:**
```
apps/api/src/api/agents/
├── utils/
│   ├── __init__.py
│   └── prompt_management.py       # Loading utilities
└── prompts/
    └── retrieval_generation.yaml  # YAML configuration
```

**YAML Configuration (retrieval_generation.yaml)**:
```yaml
metadata:
  name: Retrieval Generation Prompt
  version: 1.0.0                    # Semantic versioning
  description: Retrieval Generation Prompt for RAG Pipeline
  author: Christoper Bischoff

prompts:
  retrieval_generation: |
    You are a shopping assistant that can answer questions about the products in stock.

    Context:
    {{ preprocessed_context }}      # Jinja2 variable

    Question:
    {{ question }}                  # Jinja2 variable
```

**Utility Function (prompt_management.py)**:
```python
import yaml
from jinja2 import Template

def prompt_template_config(yaml_file, prompt_key):
    """Load prompt from YAML configuration file."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt_key]
    template = Template(template_content)

    return template
```

**Updated build_prompt() Function**:
```python
from api.agents.utils.prompt_management import prompt_template_config

def build_prompt(preprocessed_context, question):
    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )
    prompt = template.render(
        preprocessed_context=preprocessed_context,
        question=question
    )
    return prompt
```

**Changes:**
- ✅ Reduced from 60+ lines to 8 lines (-87%)
- ✅ Prompt now lives in YAML file (version control)
- ✅ Metadata for documentation and versioning
- ✅ Jinja2 template engine for variable substitution
- ✅ Non-engineers can edit YAML without touching Python

#### 3. Four-Stage Evolution (Learning Path)

**Stage 1: F-String Prompts (Baseline)**
```python
prompt = f"Context: {context}\nQuestion: {question}"
```
- ✅ Simple, direct
- ❌ Hardcoded in code

**Stage 2: Jinja2 Template Strings**
```python
template = Template("Context: {{ context }}\nQuestion: {{ question }}")
prompt = template.render(context=context, question=question)
```
- ✅ Template syntax clearer than f-strings
- ❌ Still hardcoded in code

**Stage 3: YAML Configuration Files**
```python
template = prompt_template_config("file.yaml", "key")
prompt = template.render(context=context, question=question)
```
- ✅ Externalized to YAML
- ✅ Version control, metadata
- ✅ Non-engineer friendly

**Stage 4: LangSmith Prompt Registry**
```python
template = prompt_template_registry("prompt-name")
prompt = template.render(context=context, question=question)
```
- ✅ Cloud-based storage
- ✅ A/B testing built-in
- ✅ Team collaboration
- ❌ External dependency, cost

#### 4. Jinja2 Template Syntax

**Variable Substitution:**
```jinja
Context:
{{ preprocessed_context }}

Question:
{{ question }}
```

**Conditionals (Advanced):**
```jinja
{% if include_reasoning %}
Explain your reasoning step-by-step.
{% endif %}
```

**Loops (Advanced):**
```jinja
{% for item in context_items %}
- {{ item }}
{% endfor %}
```

**Filters:**
```jinja
{{ product_name | upper }}
{{ description | truncate(50) }}
```

#### 5. YAML Structure and Syntax

**Multiline Strings:**
```yaml
prompts:
  my_prompt: |               # Literal block (preserves newlines)
    Line 1
    Line 2

  another: |-                # Strip final newline
    No trailing newline
```

**Metadata Section:**
```yaml
metadata:
  name: Descriptive Name
  version: 1.0.0             # Semantic versioning
  description: What this prompt does
  author: Your Name
  created: 2026-01-26
  updated: 2026-01-26
```

**Multiple Prompts:**
```yaml
prompts:
  prompt_a: |
    First prompt...

  prompt_b: |
    Second prompt...
```

#### 6. File Path Considerations

**Local Development:**
```python
yaml_file = "apps/api/src/api/agents/prompts/retrieval_generation.yaml"
```

**Docker Container:**
- Working directory: `/app`
- Volume mount: `./apps/api/src:/app/apps/api/src`
- Same relative path works due to volume mount preserving structure

**Key Insight**: Paths relative to project root work in both environments.

#### 7. Benefits Analysis

**Code Quality:**
- 🟢 **Reduced LOC**: 60-line function → 8-line function (-87%)
- 🟢 **Cleaner Code**: Logic focused, not prompt text
- 🟢 **Easier Testing**: Mock template loader vs multiline string
- 🟢 **Better Reviews**: Prompt changes in YAML diffs, not Python diffs

**Collaboration:**
- 🟢 **Non-Engineer Friendly**: YAML is human-readable
- 🟢 **Parallel Work**: Engineers on logic, prompt engineers on prompts
- 🟢 **Clear Ownership**: Prompt files owned by prompt engineering team
- 🟢 **Reduced Merge Conflicts**: Less code overlap

**Versioning:**
- 🟢 **Semantic Versioning**: 1.0.0 → 1.1.0 for prompt updates
- 🟢 **Git History**: Clear prompt evolution in YAML file
- 🟢 **Rollback**: Revert to previous YAML version easily
- 🟢 **Documentation**: Metadata tracks author, description, version

**Deployment:**
- 🟢 **Faster Iteration**: Change YAML without code deployment
- 🟢 **A/B Testing**: Load different prompts at runtime
- 🟢 **Registry Integration**: LangSmith for cloud-based management
- 🟢 **Hot Reload**: YAML changes picked up by FastAPI auto-reload

#### 8. Performance Impact

**YAML Loading Overhead:**
- File I/O: ~1ms per load
- YAML parsing: ~1ms
- Template creation: <1ms
- **Total: ~3ms per request**

**Impact on RAG Pipeline:**
- Total RAG latency: ~1-3 seconds
- Prompt loading: ~3ms (~0.1-0.3% overhead)
- **Negligible impact**

**Optimization (Future):**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def prompt_template_config_cached(yaml_file, prompt_key):
    """Cached version: loads YAML once, reuses template."""
    return template
```
- First call: ~3ms
- Subsequent calls: <0.01ms (cache hit)

#### 9. Semantic Versioning for Prompts

**Version Format:** `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)

**Rules:**
- **PATCH** (1.0.0 → 1.0.1): Bug fixes
  - Typo corrections
  - Grammar fixes
  - Clarified existing instructions

- **MINOR** (1.0.0 → 1.1.0): New features (backward compatible)
  - Added new instructions
  - Improved clarity
  - Added optional fields

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
  - Different output format (text → JSON)
  - Removed required fields
  - Changed variable names

#### 10. Git Workflow for Prompt Changes

**1. Edit YAML File:**
```bash
vim apps/api/src/api/agents/prompts/retrieval_generation.yaml
```

**2. Update Version in Metadata:**
```yaml
metadata:
  version: 1.1.0  # Was 1.0.0
```

**3. Test Changes:**
```bash
make smoke-test  # Validates end-to-end RAG pipeline
```

**4. Commit with Descriptive Message:**
```bash
git commit -m "feat(prompts): add product rating emphasis to RAG prompt (v1.1.0)"
# or
git commit -m "fix(prompts): correct typo in system instructions (v1.0.1)"
```

**5. Deploy:**
- FastAPI hot reload picks up YAML changes automatically
- No code deployment needed

#### 11. Common Pitfalls for AI Assistants

**Pitfall 1: Wrong File Path**
```python
# ❌ Wrong: Path from container perspective only
yaml_file = "api/agents/prompts/retrieval_generation.yaml"

# ✅ Right: Path from project root (works in both local and Docker)
yaml_file = "apps/api/src/api/agents/prompts/retrieval_generation.yaml"
```

**Pitfall 2: Mixing F-String and Jinja2 Syntax**
```yaml
# ❌ Wrong: Using f-string syntax in YAML
prompts:
  my_prompt: |
    Context: {context}

# ✅ Right: Using Jinja2 syntax
prompts:
  my_prompt: |
    Context: {{ context }}
```

**Pitfall 3: YAML Multiline Syntax**
```yaml
# ❌ Wrong: Missing | for multiline
prompts:
  my_prompt:
    Line 1
    Line 2

# ✅ Right: Use | for multiline
prompts:
  my_prompt: |
    Line 1
    Line 2
```

**Pitfall 4: Missing Variables in Render**
```python
# ❌ Wrong: Missing variable
prompt = template.render(question="What is X?")
# Error: jinja2.exceptions.UndefinedError: 'preprocessed_context' is undefined

# ✅ Right: All variables provided
prompt = template.render(
    preprocessed_context="...",
    question="What is X?"
)
```

#### 12. Testing Prompt Changes

**Unit Test (Template Loading):**
```python
def test_prompt_template_config():
    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )

    prompt = template.render(
        preprocessed_context="Test context",
        question="Test question"
    )

    assert "Test context" in prompt
    assert "Test question" in prompt
    assert "shopping assistant" in prompt.lower()
```

**Integration Test (RAG Pipeline):**
```python
def test_rag_pipeline_with_template():
    result = rag_pipeline("best wireless headphones")

    assert "answer" in result
    assert len(result["answer"]) > 0
```

**Smoke Test (Production-Like):**
```bash
make smoke-test
# Validates:
# - Template loads correctly
# - Variables render properly
# - LLM generates answer
# - Response structure matches Pydantic models
```

#### 13. LangSmith Registry Integration (Future Enhancement)

**What is LangSmith?**
- Cloud-based prompt management platform by LangChain
- Centralized storage for prompt templates
- Version control with rollback support
- A/B testing infrastructure
- Analytics and performance monitoring

**Usage:**
```python
from langsmith import Client

ls_client = Client()

def prompt_template_registry(prompt_name):
    """Load prompt from LangSmith registry."""
    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template
    template = Template(template_content)
    return template

# Usage
template = prompt_template_registry("retrieval-generation")
prompt = template.render(preprocessed_context="...", question="...")
```

**Benefits:**
- ✅ Team collaboration without Git
- ✅ A/B testing with traffic splitting
- ✅ Version history with one-click rollback
- ✅ Performance analytics

**Trade-offs:**
- ❌ External dependency (network required)
- ❌ Cost ($39/month for teams)
- ✅ Local YAML fallback available

### Key Learnings for AI Assistants

1. **Separation of Concerns**: Keep prompts separate from code (YAML files)
2. **Template Engines**: Jinja2 provides powerful variable substitution
3. **Metadata Matters**: Version, author, description enable collaboration
4. **Utility Functions**: Centralize loading logic for reusability
5. **Docker Paths**: Volume mounts preserve relative paths from project root
6. **Registry Integration**: Cloud-based management enables advanced workflows
7. **Testing**: Validate templates in isolation before production
8. **Caching**: Load templates once, reuse for performance
9. **Monitoring**: Log versions and errors for debugging
10. **Migration**: Gradual refactoring with fallbacks reduces risk

---

## 📚 Related Documentation

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

**Last Review:** 2026-01-25
**Next Review:** After major architectural changes or monthly maintenance
