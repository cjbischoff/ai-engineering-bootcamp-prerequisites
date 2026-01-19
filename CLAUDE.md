# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Engineering Bootcamp prerequisites repository featuring a complete chatbot application stack with FastAPI backend, Streamlit frontend, and Qdrant vector database for RAG operations. This is a **uv workspace monorepo** with modular apps.

## Essential Commands

### Environment & Dependencies
```bash
# Install/sync all dependencies (root + all workspace apps)
uv sync

# Add dependency to root workspace
uv add <package-name>

# Add dependency to specific app
uv add --package api <package-name>
uv add --package chatbot-ui <package-name>
```

### Running Services

**Docker Compose (Recommended):**
```bash
# Start all services (API, UI, Qdrant)
make run-docker-compose
# or manually:
uv sync && docker compose up --build

# Detached mode
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Force rebuild
docker compose up --build --force-recreate
```

**Local Development (No Docker):**
```bash
# Run FastAPI backend
cd apps/api
uv run uvicorn api.app:app --reload --port 8000

# Run Streamlit frontend (separate terminal)
cd apps/chatbot_ui
uv run streamlit run src/chatbot_ui/app.py

# Run Jupyter notebooks
uv run jupyter notebook notebooks/
```

### Jupyter Notebooks
```bash
# Launch Jupyter
uv run jupyter notebook notebooks/

# Clean outputs before commit (REQUIRED)
make clean-notebook-outputs
```

### Service URLs
- Chatbot UI: http://localhost:8501
- API Docs: http://localhost:8000/docs
- API Health: http://localhost:8000/health
- Qdrant Dashboard: http://localhost:6333/dashboard
- Qdrant API: http://localhost:6333

## Architecture & Structure

### Workspace Organization
**uv workspace monorepo** with three main components:
- Root workspace: Shared dependencies and workspace config
- `apps/api`: FastAPI backend service
- `apps/chatbot_ui`: Streamlit frontend service

### Key Architectural Patterns

**1. Multi-Provider LLM Abstraction**
- `apps/api/src/api/app.py:25-47` - `run_llm()` function abstracts three providers (OpenAI, Groq, Google GenAI)
- Clients initialized once at module level (lines 20-22) for performance
- Provider selection via POST request, not environment switching

**2. Configuration Management**
- Both apps use `pydantic-settings` with `.env` file loading
- Config classes: `api/core/config.py`, `chatbot_ui/core/config.py`
- Pattern: `AppConfig` class with `model_config = SettingsConfigDict(env_file=".env")`

**3. Service Communication**
- Frontend → Backend via `API_URL` environment variable (defaults to `http://api:8000` in Docker)
- `chatbot_ui/app.py:71` - Direct POST to `/chat` endpoint
- Error handling with popup messages in `api_call()` helper (lines 6-36)

**4. RAG Pipeline (Week 1)**
- **Preprocessing**: `notebooks/week1/02-RAG-preprocessing-Amazon.ipynb`
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Vector Storage**: Qdrant collection `"Amazon-items-collection-00"`
- **Distance Metric**: Cosine similarity for semantic search
- **Data Flow**: Product descriptions → embeddings → Qdrant → semantic retrieval

**5. RAG API Implementation**

This is the production FastAPI implementation of the RAG (Retrieval-Augmented Generation) pipeline for answering product questions.

**Architecture Overview:**
- **Location**: `apps/api/src/api/`
- **Pattern**: FastAPI with modular routing (APIRouter)
- **Request Flow**: Client → Middleware → Router → Endpoint → RAG Pipeline → LLM → Response

**Key Components:**

**a) Application Setup** (`api/app.py`)
- FastAPI instance with auto-generated OpenAPI docs at `/docs`
- Middleware stack (order matters - added first = runs first):
  1. `RequestIDMiddleware` - Generates UUID for request tracing
  2. `CORSMiddleware` - Enables cross-origin requests from frontend
- Router registration: `api_router` mounted at root with `/rag` prefix
- Why CORS: Allows Streamlit frontend (port 8501) to call API (port 8000)

**b) Request Tracing Middleware** (`api/middleware.py`)
- Implements `BaseHTTPMiddleware` pattern
- Generates UUID v4 for every incoming request
- Attaches `request_id` to `request.state` (accessible in endpoints)
- Adds `X-Request-ID` response header for client-side tracking
- Logs request start/completion for distributed tracing
- Benefits: Debugging, correlation across services, client support tickets

**c) Request/Response Models** (`api/models.py`)
- Uses Pydantic `BaseModel` for automatic validation
- `RAGRequest`: Validates incoming JSON with required `query` field
- `RAGResponse`: Structured response with `request_id` and `answer`
- Why Pydantic: Type safety, auto-validation, OpenAPI schema generation
- FastAPI returns 422 Unprocessable Entity if validation fails

**d) API Endpoints** (`api/endpoints.py`)
- `POST /rag/` - Main Q&A endpoint
- Receives `RAGRequest`, calls `rag_pipeline()`, returns `RAGResponse`
- Uses `APIRouter` pattern for modularity (easy to add `/rag/health`, `/rag/feedback`)
- Tags endpoints as "rag" for OpenAPI documentation grouping
- Request ID automatically available via `request.state.request_id`

**e) RAG Pipeline Logic** (`agents/retrieval_generation.py`)

Complete 5-step RAG workflow:

1. **Embedding Generation** (`get_embedding`)
   - Converts text to 1536-dimensional vector using OpenAI
   - Same model as preprocessing: `text-embedding-3-small`
   - Enables semantic similarity matching (not just keyword search)

2. **Vector Retrieval** (`retrieve_data`)
   - Queries Qdrant for k-nearest neighbors (default k=5)
   - Uses cosine similarity to find semantically similar products
   - HNSW algorithm for fast approximate nearest neighbor search
   - Returns: product IDs, descriptions, ratings, similarity scores

3. **Context Formatting** (`process_context`)
   - Converts structured data (lists) into human-readable text
   - Format: `- ID: {asin}, rating: {rating}, description: {description}`
   - Why: LLM needs formatted text, not raw Python data structures
   - Uses `zip()` to iterate through parallel lists

4. **Prompt Construction** (`build_prompt`)
   - Combines system instructions, retrieved context, and user question
   - System role: "shopping assistant"
   - Key constraint: "Only use provided context" (prevents hallucination)
   - Prompt engineering: Clear role → Task → Context → Constraint → Question

5. **Answer Generation** (`generate_answer`)
   - Uses OpenAI `gpt-5-nano` with `reasoning_effort="minimal"`
   - Why nano: Cost-effective for straightforward retrieval-based answers
   - Single system message (no conversation history)
   - Returns natural language answer grounded in retrieved products

**Docker Integration:**
- Qdrant connection: `http://qdrant:6333` (Docker Compose service name)
- Container networking: Services communicate by name, not localhost
- Volume mount: `./qdrant_storage:/qdrant/storage:z` for persistence

**Lessons Learned:**

1. **TypeError with zip() and tuple()**
   - Problem: `zip(tuple(list1, list2, list3))` is invalid syntax
   - Cause: `tuple()` takes one iterable, not multiple arguments
   - Fix: Use `zip(list1, list2, list3)` directly - no tuple wrapper needed
   - When multi-line formatting splits arguments, the error becomes visible
   - Detection: Runtime `TypeError: tuple expected at most 1 argument, got 3`

2. **Qdrant Connection in Docker**
   - Use service name `http://qdrant:6333`, not `http://localhost:6333`
   - Docker Compose creates internal DNS for service-to-service communication
   - Localhost refers to container itself, not the host machine
   - Verify collection exists: `docker compose exec qdrant curl http://localhost:6333/collections`

3. **Middleware Ordering Matters**
   - Middleware added first runs first (outermost layer)
   - RequestIDMiddleware before CORS ensures UUID generation happens first
   - Order: Request → RequestID (generate UUID) → CORS (validate) → Endpoint
   - Response flows back through middleware in reverse order

4. **Pydantic Validation Benefits**
   - FastAPI automatically validates request JSON against model schema
   - Returns 422 with detailed errors if validation fails (not 500 crashes)
   - Field descriptions appear in auto-generated OpenAPI documentation
   - Type hints enable IDE autocomplete and catch bugs early

5. **RAG vs Pure LLM**
   - Pure LLM: May hallucinate product details, knowledge cutoff limits
   - RAG: Grounds answers in actual product data, always current
   - Trade-off: RAG requires vector database setup but provides verifiable answers
   - Best for: Product recommendations, Q&A over specific datasets

6. **OpenAI Embedding Consistency**
   - Critical: Use same embedding model for preprocessing AND runtime queries
   - Preprocessing used `text-embedding-3-small` → runtime must match
   - Different models = different vector spaces = poor retrieval quality
   - Dimension mismatch will cause Qdrant errors

7. **Request Tracing Value**
   - UUID in response body AND header enables multiple use cases
   - Client can display: "Something wrong? Reference request ID: abc123"
   - Logs can be filtered: `grep "request_id: abc123" logs/`
   - Distributed tracing: Track request across multiple services
   - Support tickets: "Request XYZ returned wrong answer"

**Production Considerations:**

Not implemented (intentional MVP scope):
- Error handling: No try/except blocks around rag_pipeline() calls
- Rate limiting: API is unprotected, could be abused
- Timeout handling: Long-running queries could hang
- Input validation: No query length limits or content sanitization
- Connection pooling: New Qdrant client created per request (inefficient)
- Caching: Common queries could be cached to reduce API calls
- Monitoring: No metrics on retrieval quality, answer relevance, latency
- Authentication: No API keys, public access
- Query logging: No analytics on what users are asking
- Response streaming: Answers returned all-at-once, not streamed

When to add these:
- Error handling: Before any production deployment
- Rate limiting: When opening to public users
- Monitoring: When analyzing system performance
- Authentication: When controlling access or charging for usage
- Caching: When reducing OpenAI API costs becomes priority

### Critical Implementation Details

**API Endpoint Structure:**
- `POST /chat` expects: `{"provider": str, "model_name": str, "messages": list[dict]}`
- Returns: `{"message": str}`
- Provider-specific handling in `run_llm()` - Google uses different API pattern (contents array)

**Provider Model Names:**
- OpenAI: `gpt-4o-mini`, `o1-mini` (includes `reasoning_effort="low"` parameter)
- Groq: `llama-3.3-70b-versatile`
- Google: `gemini-2.0-flash-exp`

**Session State Management:**
- Streamlit maintains conversation history in `st.session_state.messages`
- Format: `[{"role": "user"|"assistant", "content": str}]`
- Initial message at `chatbot_ui/app.py:58`

**Docker Volume Mounts:**
- API source: `./apps/api/src:/app/apps/api/src` (hot reload)
- UI source: `./apps/chatbot_ui/src:/app/apps/chatbot_ui/src` (hot reload)
- Qdrant storage: `./qdrant_storage:/qdrant/storage:z` (persistent vectors)

### Data Pipeline Architecture

**Week 1 Dataset Processing:**
1. Raw data: Amazon Electronics reviews (2.5M+ records)
2. Filter: Products from 2022-2023 with valid categories
3. Filter: 100+ ratings threshold → 17,162 products
4. Sample: Reproducible 1,000-item subset (seed=42)
5. Output: `data/meta_Electronics_2022_2023_with_category_ratings_over_100_sample_1000.jsonl`

**RAG Preprocessing Flow:**
1. Load 1,000-item sample
2. Combine `title + features` into rich descriptions
3. Generate embeddings via OpenAI API
4. Create Qdrant points: `PointStruct(id, vector, payload)`
5. Batch upsert to collection (wait=True for indexing)
6. Enable semantic search via `query_points(query_embedding, limit=k)`

## Development Guidelines

### Environment Variables
Required in `.env` (use `env.example` as template):
```env
OPENAI_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
API_URL=http://api:8000  # or http://localhost:8000 for local dev
```

### Adding New LLM Provider
1. Add API key to config classes in both `api/core/config.py` and `chatbot_ui/core/config.py`
2. Initialize client at module level in `api/app.py` (lines 19-22)
3. Add provider case to `run_llm()` function (line 25)
4. Add provider to Streamlit sidebar dropdown (line 44)
5. Add model options for provider (lines 45-50)

### Working with Qdrant
- Collection persists in `./qdrant_storage/` (gitignored)
- To reset: `rm -rf qdrant_storage/` and restart Docker
- Vector dimensions must match embedding model (1536 for text-embedding-3-small)
- Payload stores full product metadata - no separate DB needed

### Notebook Hygiene
**CRITICAL**: Always run `make clean-notebook-outputs` before committing notebooks. Outputs can contain:
- Large embedded data
- API responses
- Execution state

### Testing Strategy
No test suite currently exists. When adding tests:
- Place in `apps/api/tests/` and `apps/chatbot_ui/tests/`
- Use `pytest` as framework (common in Python ecosystem)
- Add `pytest` to root workspace: `uv add --dev pytest`

## Common Patterns

### Extending the API
Add endpoints to `apps/api/src/api/app.py`:
```python
@app.post("/new-endpoint")
def new_handler(payload: RequestModel) -> ResponseModel:
    # Implementation
    return ResponseModel(...)
```

### Modifying UI Components
Streamlit uses imperative rendering - all changes to `apps/chatbot_ui/src/chatbot_ui/app.py` hot reload automatically.

### Adding Dependencies
- Shared by all apps: `uv add <package>`
- API-specific: `uv add --package api <package>`
- UI-specific: `uv add --package chatbot-ui <package>`

## Troubleshooting

**Port conflicts:**
```bash
lsof -i :8000  # API
lsof -i :8501  # Streamlit
lsof -i :6333  # Qdrant
```

**API connection errors from UI:**
- Check `API_URL` in `.env` matches running service
- In Docker: use `http://api:8000` (service name)
- Local dev: use `http://localhost:8000`

**Docker build failures:**
```bash
docker compose down
docker system prune -f
docker compose up --build
```

**Qdrant connection issues:**
- Verify service is running: `docker compose ps`
- Check logs: `docker compose logs qdrant`
- Ensure port 6333 not already in use

## Important Notes

- **Never commit** `.env` files or Jupyter notebook outputs
- **Raw datasets** not included in repo - download from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html)
- **Qdrant storage** persists between Docker restarts - delete `qdrant_storage/` to reset
- **uv workspace** - dependency changes require `uv sync` at root level
- **Git branch strategy** - current branch is `week1/exercises`, main branch is `setup/dev-environment`

---

## 🎯 Project Defaults

**Note:** This section is built iteratively over time as practices are discovered and validated.

### MCP Server Usage Guidelines

Understanding when to use each MCP server and how they work together.

---

#### Context7 MCP Server

**Purpose**: Official library documentation lookup and framework pattern guidance

**Triggers:**
- Import statements: `import`, `require`, `from`, `use`
- Framework keywords: React, Vue, Angular, Next.js, Express, etc.
- Library-specific questions about APIs or best practices
- Need for official documentation patterns vs generic solutions
- Version-specific implementation requirements

**Choose When:**
- **Over WebSearch**: When you need curated, version-specific documentation
- **Over native knowledge**: When implementation must follow official patterns
- **For frameworks**: React hooks, Vue composition API, Angular services
- **For libraries**: Correct API usage, authentication flows, configuration
- **For compliance**: When adherence to official standards is mandatory

**Works Best With:**
- **Sequential**: Context7 provides docs → Sequential analyzes implementation strategy
- **Magic**: Context7 supplies patterns → Magic generates framework-compliant components

**Examples:**
```
"implement React useEffect" → Context7 (official React patterns)
"add authentication with Auth0" → Context7 (official Auth0 docs)
"migrate to Vue 3" → Context7 (official migration guide)
"optimize Next.js performance" → Context7 (official optimization patterns)
"just explain this function" → Native Claude (no external docs needed)
```

---

#### Serena MCP Server

**Purpose**: Semantic code understanding with project memory and session persistence

**Triggers:**
- Symbol operations: rename, extract, move functions/classes
- Project-wide code navigation and exploration
- Multi-language projects requiring LSP integration
- Session lifecycle: `/sc:load`, `/sc:save`, project activation
- Memory-driven development workflows
- Large codebase analysis (>50 files, complex architecture)

**Choose When:**
- **Over Morphllm**: For symbol operations, not pattern-based edits
- **For semantic understanding**: Symbol references, dependency tracking, LSP integration
- **For session persistence**: Project context, memory management, cross-session learning
- **For large projects**: Multi-language codebases requiring architectural understanding
- **Not for simple edits**: Basic text replacements, style enforcement, bulk operations

**Works Best With:**
- **Morphllm**: Serena analyzes semantic context → Morphllm executes precise edits
- **Sequential**: Serena provides project context → Sequential performs architectural analysis

**Examples:**
```
"rename getUserData function everywhere" → Serena (symbol operation with dependency tracking)
"find all references to this class" → Serena (semantic search and navigation)
"load my project context" → Serena (/sc:load with project activation)
"save my current work session" → Serena (/sc:save with memory persistence)
"update all console.log to logger" → Morphllm (pattern-based replacement)
"create a login form" → Magic (UI component generation)
```

---

#### Playwright MCP Server

**Purpose**: Browser automation and E2E testing with real browser interaction

**Triggers:**
- Browser testing and E2E test scenarios
- Visual testing, screenshot, or UI validation requests
- Form submission and user interaction testing
- Cross-browser compatibility validation
- Performance testing requiring real browser rendering
- Accessibility testing with automated WCAG compliance

**Choose When:**
- **For real browser interaction**: When you need actual rendering, not just code
- **Over unit tests**: For integration testing, user journeys, visual validation
- **For E2E scenarios**: Login flows, form submissions, multi-page workflows
- **For visual testing**: Screenshot comparisons, responsive design validation
- **Not for code analysis**: Static code review, syntax checking, logic validation

**Works Best With:**
- **Sequential**: Sequential plans test strategy → Playwright executes browser automation
- **Magic**: Magic creates UI components → Playwright validates accessibility and behavior

**Examples:**
```
"test the login flow" → Playwright (browser automation)
"check if form validation works" → Playwright (real user interaction)
"take screenshots of responsive design" → Playwright (visual testing)
"validate accessibility compliance" → Playwright (automated WCAG testing)
"review this function's logic" → Native Claude (static analysis)
"explain the authentication code" → Native Claude (code review)
```

---

#### Sequential MCP Server

**Purpose**: Multi-step reasoning engine for complex analysis and systematic problem solving

**Triggers:**
- Complex debugging scenarios with multiple layers
- Architectural analysis and system design questions
- `--think`, `--think-hard`, `--ultrathink` flags
- Problems requiring hypothesis testing and validation
- Multi-component failure investigation
- Performance bottleneck identification requiring methodical approach

**Choose When:**
- **Over native reasoning**: When problems have 3+ interconnected components
- **For systematic analysis**: Root cause analysis, architecture review, security assessment
- **When structure matters**: Problems benefit from decomposition and evidence gathering
- **For cross-domain issues**: Problems spanning frontend, backend, database, infrastructure
- **Not for simple tasks**: Basic explanations, single-file changes, straightforward fixes

**Works Best With:**
- **Context7**: Sequential coordinates analysis → Context7 provides official patterns
- **Magic**: Sequential analyzes UI logic → Magic implements structured components
- **Playwright**: Sequential identifies testing strategy → Playwright executes validation

**Examples:**
```
"why is this API slow?" → Sequential (systematic performance analysis)
"design a microservices architecture" → Sequential (structured system design)
"debug this authentication flow" → Sequential (multi-component investigation)
"analyze security vulnerabilities" → Sequential (comprehensive threat modeling)
"explain this function" → Native Claude (simple explanation)
"fix this typo" → Native Claude (straightforward change)
```

---

#### Episodic Memory MCP Server

**Purpose**: Conversation history search and memory across sessions

**Triggers:**
- Starting any new task or project
- Feeling like you've solved something before
- Stuck or going in circles on a problem
- User asks "how should I..." or "what's the best approach..."
- Unfamiliar workflows or patterns in a codebase
- User references past work or decisions
- After exploring code and need to remember past context

**Choose When:**
- **Before starting work**: Search for past decisions, solutions, and lessons learned
- **When repeating work**: Feeling like you've already solved this problem
- **For workflow guidance**: Don't remember the best approach for a task
- **After exploration**: Tried to solve something and are stuck
- **Not for current session**: Only for cross-session memory, not current conversation

**Works Best With:**
- **Serena**: Episodic Memory recalls past project decisions → Serena loads project context
- **Sequential**: Episodic Memory finds past debugging approaches → Sequential applies systematic analysis
- **Any workflow**: Always check memory before major decisions or architectural changes

**Examples:**
```
"start implementing auth" → Episodic Memory first (check past auth decisions)
"how did we handle errors before?" → Episodic Memory (recall past patterns)
"stuck on this performance issue" → Episodic Memory (find similar past solutions)
"I remember discussing this..." → Episodic Memory (specific recall)
"what was the result of that test?" → Current session context (not episodic)
"explain this function" → Native Claude (no memory needed)
```

---

#### Ralph Loop MCP Server

**Purpose**: Iterative workflow automation for refinement and autonomous task execution

**Triggers:**
- Repetitive refinement cycles (UI tweaks, prompt optimization, config tuning)
- Tasks requiring multiple iterations to converge on solution
- Background work you want to run autonomously
- Complex multi-step workflows needing coordination
- When you'd normally do "try this, then adjust, then try again"
- Experimenting with variations to find optimal solution

**Choose When:**
- **For iterative refinement**: UI polish, prompt engineering, configuration optimization
- **For autonomous work**: Set it running on a problem while you work on something else
- **For multi-step coordination**: Complex workflows with many sequential dependencies
- **When exploration is needed**: Try different approaches to find the best solution
- **Not for simple tasks**: One-shot implementations, straightforward fixes

**Works Best With:**
- **Playwright**: Ralph Loop runs test iterations → Playwright executes browser validation
- **Sequential**: Ralph Loop coordinates refinement cycles → Sequential analyzes each iteration
- **Magic**: Ralph Loop generates UI variations → Magic implements each iteration

**Examples:**
```
"refine this UI until it looks right" → Ralph Loop (iterative refinement)
"optimize these prompts for best results" → Ralph Loop (autonomous experimentation)
"fix this bug, I'll be back in 30 mins" → Ralph Loop (background autonomous work)
"coordinate these 5 service updates" → Ralph Loop (multi-step workflow)
"fix this typo" → Native Claude (simple one-shot task)
"explain this code" → Native Claude (no iteration needed)
```

---

#### Chrome DevTools MCP Server

**Purpose**: Chrome DevTools integration for debugging, performance analysis, and development workflows

**Triggers:**
- Performance profiling and Core Web Vitals analysis
- Detailed network request inspection (headers, timing, payload)
- Console message analysis beyond basic logs
- DOM inspection and element state debugging
- CPU/memory profiling and throttling
- Geolocation emulation and device emulation
- Development-time debugging vs production E2E testing

**Choose When:**
- **For DevTools features**: Performance panel, Network waterfall, detailed console analysis
- **For Chrome specifics**: Chrome-only debugging, DevTools-specific insights
- **For development debugging**: Active development and troubleshooting in Chrome
- **Complement to Playwright**: Use both - DevTools for debugging, Playwright for E2E
- **Not for cross-browser**: Chrome-only; use Playwright for multi-browser testing

**Works Best With:**
- **Playwright**: Playwright runs E2E tests → Chrome DevTools profiles performance
- **Sequential**: Sequential identifies performance bottleneck → Chrome DevTools provides detailed profiling
- **Ralph Loop**: Ralph Loop iterates on optimization → Chrome DevTools measures improvements

**Examples:**
```
"profile this page's performance" → Chrome DevTools (Performance panel)
"why is this request slow?" → Chrome DevTools (Network panel timing)
"analyze Core Web Vitals" → Chrome DevTools (performance insights)
"debug console errors in Chrome" → Chrome DevTools (Console panel)
"emulate mobile device" → Chrome DevTools (device emulation)
"run E2E tests across browsers" → Playwright (cross-browser testing)
"test this form submission" → Playwright or Chrome DevTools (both work)
```

---

### Claude Code Behavioral Rules

Actionable rules for enhanced Claude Code framework operation.

#### Rule Priority System

**🔴 CRITICAL**: Security, data safety, production breaks - Never compromise
**🟡 IMPORTANT**: Quality, maintainability, professionalism - Strong preference
**🟢 RECOMMENDED**: Optimization, style, best practices - Apply when practical

**Conflict Resolution Hierarchy:**
1. **Safety First**: Security/data rules always win
2. **Scope > Features**: Build only what's asked > complete everything
3. **Quality > Speed**: Except in genuine emergencies
4. **Context Matters**: Prototype vs Production requirements differ

---

#### Agent Orchestration
**Priority**: 🔴 **Triggers**: Task execution and post-implementation

**Task Execution Layer** (Existing Auto-Activation):
- **Auto-Selection**: Claude Code automatically selects appropriate specialist agents based on context
- **Keywords**: Security, performance, frontend, backend, architecture keywords trigger specialist agents
- **File Types**: `.py`, `.jsx`, `.ts`, etc. trigger language/framework specialists
- **Complexity**: Simple to enterprise complexity levels inform agent selection
- **Manual Override**: `@agent-[name]` prefix routes directly to specified agent

**Self-Improvement Layer** (PM Agent Meta-Layer):
- **Post-Implementation**: PM Agent activates after task completion to document learnings
- **Mistake Detection**: PM Agent activates immediately when errors occur for root cause analysis
- **Monthly Maintenance**: PM Agent performs systematic documentation health reviews
- **Knowledge Capture**: Transforms experiences into reusable patterns and best practices
- **Documentation Evolution**: Maintains fresh, minimal, high-signal documentation

**Orchestration Flow**:
1. **Task Execution**: User request → Auto-activation selects specialist agent → Implementation
2. **Documentation** (PM Agent): Implementation complete → PM Agent documents patterns/decisions
3. **Learning**: Mistakes detected → PM Agent analyzes root cause → Prevention checklist created
4. **Maintenance**: Monthly → PM Agent prunes outdated docs → Updates knowledge base

✅ **Right**: User request → backend-architect implements → PM Agent documents patterns
✅ **Right**: Error detected → PM Agent stops work → Root cause analysis → Documentation updated
✅ **Right**: `@agent-security "review auth"` → Direct to security-engineer (manual override)
❌ **Wrong**: Skip documentation after implementation (no PM Agent activation)
❌ **Wrong**: Continue implementing after mistake (no root cause analysis)

---

#### Workflow Rules
**Priority**: 🟡 **Triggers**: All development tasks

- **Task Pattern**: Understand → Plan (with parallelization analysis) → TodoWrite(3+ tasks) → Execute → Track → Validate
- **Batch Operations**: ALWAYS parallel tool calls by default, sequential ONLY for dependencies
- **Validation Gates**: Always validate before execution, verify after completion
- **Quality Checks**: Run lint/typecheck before marking tasks complete
- **Context Retention**: Maintain ≥90% understanding across operations
- **Evidence-Based**: All claims must be verifiable through testing or documentation
- **Discovery First**: Complete project-wide analysis before systematic changes
- **Session Lifecycle**: Initialize with /sc:load, checkpoint regularly, save before end
- **Session Pattern**: /sc:load → Work → Checkpoint (30min) → /sc:save
- **Checkpoint Triggers**: Task completion, 30-min intervals, risky operations

✅ **Right**: Plan → TodoWrite → Execute → Validate
❌ **Wrong**: Jump directly to implementation without planning

---

#### Planning Efficiency
**Priority**: 🔴 **Triggers**: All planning phases, TodoWrite operations, multi-step tasks

- **Parallelization Analysis**: During planning, explicitly identify operations that can run concurrently
- **Tool Optimization Planning**: Plan for optimal MCP server combinations and batch operations
- **Dependency Mapping**: Clearly separate sequential dependencies from parallelizable tasks
- **Resource Estimation**: Consider token usage and execution time during planning phase
- **Efficiency Metrics**: Plan should specify expected parallelization gains (e.g., "3 parallel ops = 60% time saving")

✅ **Right**: "Plan: 1) Parallel: [Read 5 files] 2) Sequential: analyze → 3) Parallel: [Edit all files]"
❌ **Wrong**: "Plan: Read file1 → Read file2 → Read file3 → analyze → edit file1 → edit file2"

---

#### Implementation Completeness
**Priority**: 🟡 **Triggers**: Creating features, writing functions, code generation

- **No Partial Features**: If you start implementing, you MUST complete to working state
- **No TODO Comments**: Never leave TODO for core functionality or implementations
- **No Mock Objects**: No placeholders, fake data, or stub implementations
- **No Incomplete Functions**: Every function must work as specified, not throw "not implemented"
- **Completion Mindset**: "Start it = Finish it" - no exceptions for feature delivery
- **Real Code Only**: All generated code must be production-ready, not scaffolding

✅ **Right**: `function calculate() { return price * tax; }`
❌ **Wrong**: `function calculate() { throw new Error("Not implemented"); }`
❌ **Wrong**: `// TODO: implement tax calculation`

---

#### Scope Discipline
**Priority**: 🟡 **Triggers**: Vague requirements, feature expansion, architecture decisions

- **Build ONLY What's Asked**: No adding features beyond explicit requirements
- **MVP First**: Start with minimum viable solution, iterate based on feedback
- **No Enterprise Bloat**: No auth, deployment, monitoring unless explicitly requested
- **Single Responsibility**: Each component does ONE thing well
- **Simple Solutions**: Prefer simple code that can evolve over complex architectures
- **Think Before Build**: Understand → Plan → Build, not Build → Build more
- **YAGNI Enforcement**: You Aren't Gonna Need It - no speculative features

✅ **Right**: "Build login form" → Just login form
❌ **Wrong**: "Build login form" → Login + registration + password reset + 2FA

---

#### Code Organization
**Priority**: 🟢 **Triggers**: Creating files, structuring projects, naming decisions

- **Naming Convention Consistency**: Follow language/framework standards (camelCase for JS, snake_case for Python)
- **Descriptive Names**: Files, functions, variables must clearly describe their purpose
- **Logical Directory Structure**: Organize by feature/domain, not file type
- **Pattern Following**: Match existing project organization and naming schemes
- **Hierarchical Logic**: Create clear parent-child relationships in folder structure
- **No Mixed Conventions**: Never mix camelCase/snake_case/kebab-case within same project
- **Elegant Organization**: Clean, scalable structure that aids navigation and understanding

✅ **Right**: `getUserData()`, `user_data.py`, `components/auth/`
❌ **Wrong**: `get_userData()`, `userdata.py`, `files/everything/`

---

#### Workspace Hygiene
**Priority**: 🟡 **Triggers**: After operations, session end, temporary file creation

- **Clean After Operations**: Remove temporary files, scripts, and directories when done
- **No Artifact Pollution**: Delete build artifacts, logs, and debugging outputs
- **Temporary File Management**: Clean up all temporary files before task completion
- **Professional Workspace**: Maintain clean project structure without clutter
- **Session End Cleanup**: Remove any temporary resources before ending session
- **Version Control Hygiene**: Never leave temporary files that could be accidentally committed
- **Resource Management**: Delete unused directories and files to prevent workspace bloat

✅ **Right**: `rm temp_script.py` after use
❌ **Wrong**: Leaving `debug.sh`, `test.log`, `temp/` directories

---

#### Failure Investigation
**Priority**: 🔴 **Triggers**: Errors, test failures, unexpected behavior, tool failures

- **Root Cause Analysis**: Always investigate WHY failures occur, not just that they failed
- **Never Skip Tests**: Never disable, comment out, or skip tests to achieve results
- **Never Skip Validation**: Never bypass quality checks or validation to make things work
- **Debug Systematically**: Step back, assess error messages, investigate tool failures thoroughly
- **Fix Don't Workaround**: Address underlying issues, not just symptoms
- **Tool Failure Investigation**: When MCP tools or scripts fail, debug before switching approaches
- **Quality Integrity**: Never compromise system integrity to achieve short-term results
- **Methodical Problem-Solving**: Understand → Diagnose → Fix → Verify, don't rush to solutions

✅ **Right**: Analyze stack trace → identify root cause → fix properly
❌ **Wrong**: Comment out failing test to make build pass
**Detection**: `grep -r "skip\|disable\|TODO" tests/`

---

#### Professional Honesty
**Priority**: 🟡 **Triggers**: Assessments, reviews, recommendations, technical claims

- **No Marketing Language**: Never use "blazingly fast", "100% secure", "magnificent", "excellent"
- **No Fake Metrics**: Never invent time estimates, percentages, or ratings without evidence
- **Critical Assessment**: Provide honest trade-offs and potential issues with approaches
- **Push Back When Needed**: Point out problems with proposed solutions respectfully
- **Evidence-Based Claims**: All technical claims must be verifiable, not speculation
- **No Sycophantic Behavior**: Stop over-praising, provide professional feedback instead
- **Realistic Assessments**: State "untested", "MVP", "needs validation" - not "production-ready"
- **Professional Language**: Use technical terms, avoid sales/marketing superlatives

✅ **Right**: "This approach has trade-offs: faster but uses more memory"
❌ **Wrong**: "This magnificent solution is blazingly fast and 100% secure!"

---

#### Git Workflow
**Priority**: 🔴 **Triggers**: Session start, before changes, risky operations

- **Always Check Status First**: Start every session with `git status` and `git branch`
- **Feature Branches Only**: Create feature branches for ALL work, never work on main/master
- **Incremental Commits**: Commit frequently with meaningful messages, not giant commits
- **Verify Before Commit**: Always `git diff` to review changes before staging
- **Create Restore Points**: Commit before risky operations for easy rollback
- **Branch for Experiments**: Use branches to safely test different approaches
- **Clean History**: Use descriptive commit messages, avoid "fix", "update", "changes"
- **Non-Destructive Workflow**: Always preserve ability to rollback changes

✅ **Right**: `git checkout -b feature/auth` → work → commit → PR
❌ **Wrong**: Work directly on main/master branch
**Detection**: `git branch` should show feature branch, not main/master

---

#### Tool Optimization
**Priority**: 🟢 **Triggers**: Multi-step operations, performance needs, complex tasks

- **Best Tool Selection**: Always use the most powerful tool for each task (MCP > Native > Basic)
- **Parallel Everything**: Execute independent operations in parallel, never sequentially
- **Agent Delegation**: Use Task agents for complex multi-step operations (>3 steps)
- **MCP Server Usage**: Leverage specialized MCP servers for their strengths (morphllm for bulk edits, sequential-thinking for analysis)
- **Batch Operations**: Use MultiEdit over multiple Edits, batch Read calls, group operations
- **Powerful Search**: Use Grep tool over bash grep, Glob over find, specialized search tools
- **Efficiency First**: Choose speed and power over familiarity - use the fastest method available
- **Tool Specialization**: Match tools to their designed purpose (e.g., playwright for web, context7 for docs)

✅ **Right**: Use MultiEdit for 3+ file changes, parallel Read calls
❌ **Wrong**: Sequential Edit calls, bash grep instead of Grep tool

---

#### File Organization
**Priority**: 🟡 **Triggers**: File creation, project structuring, documentation

- **Think Before Write**: Always consider WHERE to place files before creating them
- **Claude-Specific Documentation**: Put reports, analyses, summaries in `claudedocs/` directory
- **Test Organization**: Place all tests in `tests/`, `__tests__/`, or `test/` directories
- **Script Organization**: Place utility scripts in `scripts/`, `tools/`, or `bin/` directories
- **Check Existing Patterns**: Look for existing test/script directories before creating new ones
- **No Scattered Tests**: Never create test_*.py or *.test.js next to source files
- **No Random Scripts**: Never create debug.sh, script.py, utility.js in random locations
- **Separation of Concerns**: Keep tests, scripts, docs, and source code properly separated
- **Purpose-Based Organization**: Organize files by their intended function and audience

✅ **Right**: `tests/auth.test.js`, `scripts/deploy.sh`, `claudedocs/analysis.md`
❌ **Wrong**: `auth.test.js` next to `auth.js`, `debug.sh` in project root

---

#### Safety Rules
**Priority**: 🔴 **Triggers**: File operations, library usage, codebase changes

- **Framework Respect**: Check package.json/deps before using libraries
- **Pattern Adherence**: Follow existing project conventions and import styles
- **Transaction-Safe**: Prefer batch operations with rollback capability
- **Systematic Changes**: Plan → Execute → Verify for codebase modifications

✅ **Right**: Check dependencies → follow patterns → execute safely
❌ **Wrong**: Ignore existing conventions, make unplanned changes

---

#### Temporal Awareness
**Priority**: 🔴 **Triggers**: Date/time references, version checks, deadline calculations, "latest" keywords

- **Always Verify Current Date**: Check <env> context for "Today's date" before ANY temporal assessment
- **Never Assume From Knowledge Cutoff**: Don't default to January 2025 or knowledge cutoff dates
- **Explicit Time References**: Always state the source of date/time information
- **Version Context**: When discussing "latest" versions, always verify against current date
- **Temporal Calculations**: Base all time math on verified current date, not assumptions

✅ **Right**: "Checking env: Today is 2025-08-15, so the Q3 deadline is..."
❌ **Wrong**: "Since it's January 2025..." (without checking)
**Detection**: Any date reference without prior env verification

---

#### Quick Reference & Decision Trees

**🔴 Before Any File Operations**
```
File operation needed?
├─ Writing/Editing? → Read existing first → Understand patterns → Edit
├─ Creating new? → Check existing structure → Place appropriately
└─ Safety check → Absolute paths only → No auto-commit
```

**🟡 Starting New Feature**
```
New feature request?
├─ Scope clear? → No → Brainstorm mode first
├─ >3 steps? → Yes → TodoWrite required
├─ Patterns exist? → Yes → Follow exactly
├─ Tests available? → Yes → Run before starting
└─ Framework deps? → Check package.json first
```

**🟢 Tool Selection Matrix**
```
Task type → Best tool:
├─ Multi-file edits → MultiEdit > individual Edits
├─ Complex analysis → Task agent > native reasoning
├─ Code search → Grep > bash grep
├─ UI components → Magic MCP > manual coding
├─ Documentation → Context7 MCP > web search
└─ Browser testing → Playwright MCP > unit tests
```

**Priority-Based Quick Actions**

**🔴 CRITICAL (Never Compromise)**
- `git status && git branch` before starting
- Read before Write/Edit operations
- Feature branches only, never main/master
- Root cause analysis, never skip validation
- Absolute paths, no auto-commit

**🟡 IMPORTANT (Strong Preference)**
- TodoWrite for >3 step tasks
- Complete all started implementations
- Build only what's asked (MVP first)
- Professional language (no marketing superlatives)
- Clean workspace (remove temp files)

**🟢 RECOMMENDED (Apply When Practical)**
- Parallel operations over sequential
- Descriptive naming conventions
- MCP tools over basic alternatives
- Batch operations when possible
