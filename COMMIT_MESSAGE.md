# feat(test): add infrastructure health checks and end-to-end smoke tests

## Summary

Implements comprehensive test infrastructure for verifying application health and RAG pipeline functionality. Adds Python-based health check and smoke test scripts integrated with Makefile for easy invocation during development workflows.

## Motivation

**Problem:** Developers waste time debugging code when the real issue is infrastructure (containers not running, collections not loaded, ports blocked). No automated way to verify the RAG pipeline works end-to-end after code changes.

**Solution:** Create diagnostic scripts that quickly verify:
1. Infrastructure is healthy (containers, ports, data)
2. RAG pipeline works correctly (API responses, data enrichment, response times)

## Changes

### New Files

#### `scripts/health_check.py` - Infrastructure Health Verification
- **Purpose:** Verify all services are running before development
- **Checks:** Docker containers, network ports, Qdrant collection, API connectivity
- **Usage:** `make health` (full output) or `make health-silent` (CI mode)
- **Performance:** < 5 seconds, no LLM calls
- **Exit codes:** 0=healthy, 1=failed (for CI/CD integration)

**What it verifies:**
- Docker containers (api, streamlit-app, qdrant) have State="running"
- Ports (8000, 8501, 6333) are accepting TCP connections
- Qdrant collection "Amazon-items-collection-00" exists with documents
- FastAPI service responds to HTTP requests (accepts 404 for unimplemented /health endpoint)

**Why Python vs Bash:**
- Better error handling and cross-platform compatibility
- Reuses project dependencies (qdrant-client, requests)
- More maintainable for complex logic (JSON parsing, type checking)

#### `scripts/smoke_test.py` - End-to-End RAG Pipeline Test
- **Purpose:** Validate RAG pipeline after code changes
- **Tests:** API response, JSON structure, Pydantic models, response time, product enrichment
- **Usage:** `make smoke-test` (summary) or `make smoke-test-verbose` (full JSON)
- **Performance:** 10-15 seconds (uses actual LLM and Qdrant)
- **Test query:** "best wireless headphones under $100"

**What it validates:**
- POST /rag/ returns 200 status with valid JSON
- Response structure matches RAGResponse Pydantic model
- Required fields present: request_id (str), answer (str), used_context (list)
- used_context items have description (required), image_url (optional), price (optional)
- Response time < 20s (accounts for cold start)
- LLM generates non-empty answer
- At least one product returned in context

**Why comprehensive validation:**
- Catches schema changes that break frontend
- Verifies Video 3 product enrichment (images, prices) still works
- Ensures acceptable performance for user experience

#### `scripts/__init__.py`
- Package marker for test utilities
- Enables future imports like `from scripts.health_check import check_port`

### Modified Files

#### `Makefile`
- Added `health` target: Full infrastructure health check with colored output
- Added `health-silent` target: CI-friendly version (only shows failures)
- Added `smoke-test` target: End-to-end RAG pipeline test
- Added `smoke-test-verbose` target: Includes full JSON response
- Updated help text to include new testing commands
- All targets run `uv sync` before execution to ensure dependencies

**Integration pattern:**
```makefile
health:
    uv sync
    uv run scripts/health_check.py
```

#### `README.md`
- Added "Testing & Health Checks" section after Quick Start
- Documented both scripts with usage examples
- Added "Development Workflow" section with recommended session flow
- Updated "Makefile Commands" with categorized list (Service Management, Testing, Development)
- Enhanced Quick Start with Step 6: "Verify Everything Works"

**Key additions:**
- When to use each script (session startup, after changes, before commits)
- Example output showing what success looks like
- Recommended workflow: start → health → code → smoke-test → commit

#### `CLAUDE.md` (AI Context Documentation)
- Updated "Session Startup Workflow" to include `make health` verification
- Added comprehensive "Test Scripts for Verification" section
- Updated "Testing Strategy" to integrate new scripts as primary testing methods
- Updated "Common Commands" table with actual Makefile targets
- Fixed architecture diagram (client → chatbot_ui, added scripts/)

**Educational content added:**
- Purpose and usage of each script
- When to run checks in development lifecycle
- Exit code patterns for CI/CD integration
- Performance characteristics (health < 5s, smoke 10-15s)

#### `apps/chatbot_ui/src/chatbot_ui/app.py` (Video 4 Feature)
- Added sidebar with "Suggestions" tab (lines 60-75)
- Displays product cards with images, prices, descriptions
- Uses session state `used_context` from API responses
- Graceful handling of missing images/prices (Optional fields)

**Why sidebar:**
- Provides visual grounding for LLM recommendations
- Enables user action (see products mentioned in answer)
- Demonstrates Video 3 product enrichment working in UI

## Technical Details

### Design Decisions

**1. Python instead of Bash:**
- Cross-platform compatibility (Windows, Mac, Linux)
- Better error handling (try/except vs checking $?)
- Type hints for code clarity
- Can reuse project dependencies

**2. Script invocation via uv run:**
```bash
uv run scripts/health_check.py
```
- Uses project virtual environment automatically
- No need to activate .venv manually
- Consistent with project's uv-based workflow

**3. ANSI color codes for output:**
- Green ✓ for success, Red ✗ for failure
- Makes test results scannable at a glance
- Colors work in most modern terminals

**4. Tuple return pattern:**
```python
def check_something() -> Tuple[bool, str]:
    return True, "Success message"
```
- Consistent interface across all check functions
- Allows main() to handle results uniformly
- Separates success/failure (bool) from details (str)

**5. Silent mode for CI/CD:**
```bash
make health-silent  # Only shows failures
```
- Reduces log noise in automated pipelines
- Exit code still indicates pass/fail
- Failures always shown (never silent about problems)

### Performance Characteristics

**health_check.py:**
- 6 checks run sequentially
- ~4-5 seconds total execution time
- No LLM calls (just infrastructure probing)
- Safe to run frequently

**smoke_test.py:**
- Full RAG pipeline execution
- 10-15 seconds on cold start
- Uses actual OpenAI embeddings and LLM
- Run after code changes, not continuously

### Error Handling

**Graceful degradation:**
- Missing dependencies → clear error message with fix command
- Connection failures → actionable error messages
- Timeouts → reasonable defaults (2s for ports, 30s for API)

**Early exit on critical failures:**
- health_check: continues after failures to show ALL problems
- smoke_test: exits immediately on connection errors (can't test if API is down)

### Code Quality

**Type hints throughout:**
```python
def check_port(port: int, service_name: str) -> Tuple[bool, str]:
```
- Improves IDE autocomplete
- Documents expected types
- Enables static analysis tools

**Comprehensive comments:**
- Every function has docstring
- Complex logic explained inline
- Educational comments for learning

**Consistent patterns:**
- Color helper functions (print_success, print_failure)
- Tuple return (success: bool, message: str)
- Try/except with specific error messages

## Testing

**Manual testing performed:**
- ✅ `make health` - All checks passed on running infrastructure
- ✅ `make health-silent` - No output when all healthy
- ✅ `make smoke-test` - RAG pipeline responded with enriched products
- ✅ `make smoke-test-verbose` - Full JSON response displayed
- ✅ Exit codes correct (0=success, 1=failure)
- ✅ Error messages actionable (tested by stopping services)

**Test scenarios validated:**
- All services running → all checks pass
- Qdrant stopped → port check fails, collection check fails
- API stopped → port check fails, API health fails
- Collection empty → collection check fails
- Invalid query → smoke test shows parsing errors

## Documentation

**Updated:**
- README.md: User-facing documentation with examples
- CLAUDE.md: AI context with workflow integration
- Makefile: Inline comments explaining each target
- Script docstrings: Comprehensive function documentation

**Follows:**
- Python PEP 257 (docstring conventions)
- Conventional commits format (feat/fix/docs)
- Project's existing documentation style

## Breaking Changes

None. All changes are additive:
- New scripts don't affect existing code
- New Makefile targets don't conflict with existing ones
- Documentation updates are additions, not replacements

## Future Enhancements

**Potential improvements:**
1. Add pytest-based unit tests for test scripts themselves
2. Create integration test suite using smoke test patterns
3. Add performance benchmarking (track RAG response times over time)
4. Implement actual /health endpoint in FastAPI (currently returns 404)
5. Add metrics collection (success rate, response times, error types)
6. Create GitHub Actions workflow using these scripts
7. Add --json flag for machine-readable output

## Migration Guide

**For existing developers:**

1. **First time:** Run `make health` to verify environment
2. **Session startup:** Add to workflow:
   ```bash
   make run-docker-compose  # Terminal 1
   make health              # Terminal 2
   ```
3. **After changes:** Run `make smoke-test` before committing
4. **CI integration:** Use `make health-silent && make smoke-test` in pipeline

**No action required:**
- Scripts are opt-in (not automatically run)
- Existing workflows continue to work
- New workflows are documented in README

## Related Work

- Video 3: Product enrichment (images, prices) validated by smoke test
- Video 4: Sidebar feature (this PR includes both test infrastructure + sidebar)
- Week 1: RAGAS evaluation (different from smoke tests - this is quick sanity check)

## Checklist

- [x] Scripts tested manually with all services running
- [x] Scripts tested with services stopped (error scenarios)
- [x] Documentation updated (README.md, CLAUDE.md)
- [x] Makefile targets documented with comments
- [x] Exit codes verified (0=success, 1=failure)
- [x] Color output works in terminal
- [x] Silent mode suppresses success messages
- [x] Verbose mode shows full JSON response
- [x] All checks have meaningful error messages
- [x] Code follows project conventions (Python 3.12+, type hints, docstrings)

---

## Commit Message (Conventional Commits Format)

```
feat(test): add infrastructure health checks and RAG smoke tests

Add Python-based diagnostic scripts for verifying infrastructure health
and end-to-end RAG pipeline functionality. Integrates with Makefile for
seamless developer workflow.

New scripts:
- scripts/health_check.py: Verify Docker, ports, Qdrant, API (< 5s)
- scripts/smoke_test.py: End-to-end RAG test with validation (10-15s)

New Makefile targets:
- make health: Full infrastructure health check
- make health-silent: CI-friendly (only show failures)
- make smoke-test: RAG pipeline validation
- make smoke-test-verbose: Include full JSON response

Updated documentation:
- README.md: Testing section with usage examples
- CLAUDE.md: Integrated tests into development workflow

Benefits:
- Quick environment verification at session startup
- Catch infrastructure issues before debugging code
- Validate RAG pipeline after changes (prevent regressions)
- CI/CD ready with proper exit codes

BREAKING CHANGE: None (additive only)

Closes #[issue-number if applicable]
```

---

**For PR Title:**
```
feat(test): add infrastructure health checks and RAG smoke tests
```

**For PR Description:** Use the "Summary" and "Motivation" sections above.
