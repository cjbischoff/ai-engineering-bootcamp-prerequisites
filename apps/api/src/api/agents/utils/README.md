# Prompt Management Utilities

Centralized utilities for loading and rendering prompt templates from YAML configuration files and LangSmith registry.

## Overview

This directory contains helper functions for prompt configuration management introduced in Week 2 / Video 7. The utilities enable externalized prompt storage with version control, A/B testing, and cleaner separation of concerns.

## Files

```
apps/api/src/api/agents/utils/
├── __init__.py                    # Makes directory a Python package
└── prompt_management.py           # Prompt loading utilities
```

## Functions

### `prompt_template_config(yaml_file, prompt_key)`

Loads a prompt template from a local YAML configuration file.

**Parameters**:
- `yaml_file: str` - Path to YAML file (relative to project root)
- `prompt_key: str` - Key in YAML's `prompts:` dictionary

**Returns**: `jinja2.Template` - Template object ready for rendering

**Example**:
```python
from api.agents.utils.prompt_management import prompt_template_config

# Load template from YAML
template = prompt_template_config(
    "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
    "retrieval_generation"
)

# Render with variables
prompt = template.render(
    preprocessed_context="- Product A: Wireless headphones\n- Product B: USB cable",
    question="What are the best headphones?"
)

print(prompt)
```

**How It Works**:
1. Opens YAML file and parses with `yaml.safe_load()`
2. Extracts template content from `config["prompts"][prompt_key]`
3. Creates Jinja2 `Template` object
4. Returns template ready for `.render()`

**File Path Considerations**:
- Path is relative to **project root** (not current file)
- Works in both local development and Docker containers
- Docker working directory: `/app` with volume mount preserving structure

---

### `prompt_template_registry(prompt_name)`

Loads a prompt template from the LangSmith prompt registry (cloud-based).

**Parameters**:
- `prompt_name: str` - Name of prompt in LangSmith registry (e.g., "retrieval-generation")

**Returns**: `jinja2.Template` - Template object ready for rendering

**Example**:
```python
from api.agents.utils.prompt_management import prompt_template_registry

# Load template from LangSmith
template = prompt_template_registry("retrieval-generation")

# Render with variables
prompt = template.render(
    preprocessed_context="...",
    question="..."
)
```

**How It Works**:
1. Connects to LangSmith using `Client()` (requires `LANGCHAIN_API_KEY` env var)
2. Pulls prompt with `ls_client.pull_prompt(prompt_name)`
3. Extracts template from `.messages[0].prompt.template`
4. Creates Jinja2 `Template` object
5. Returns template ready for `.render()`

**Requirements**:
- LangSmith account: https://smith.langchain.com
- Environment variables:
  ```bash
  export LANGCHAIN_API_KEY=<your-api-key>
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_PROJECT=<your-project-name>
  ```

**Benefits**:
- ✅ Cloud-based storage (no local files)
- ✅ Team collaboration without Git
- ✅ A/B testing with traffic splitting
- ✅ Version history with one-click rollback
- ✅ Performance analytics

**Trade-offs**:
- ❌ External dependency (network required)
- ❌ Cost ($39/month for teams)
- ✅ Local YAML fallback available

## Implementation Details

### Dependencies

```python
import yaml              # YAML parsing
from jinja2 import Template  # Template rendering
from langsmith import Client  # LangSmith integration
```

**Install**:
```bash
pip install pyyaml jinja2 langsmith
# or with uv:
uv add pyyaml jinja2 langsmith
```

### Jinja2 Template Syntax

**Variable Substitution**:
```jinja
Context:
{{ preprocessed_context }}

Question:
{{ question }}
```

**Conditional Logic** (not used in current prompts):
```jinja
{% if include_reasoning %}
Explain your reasoning step-by-step.
{% endif %}
```

**Loops** (not used in current prompts):
```jinja
{% for item in context_items %}
- {{ item }}
{% endfor %}
```

### YAML Structure Example

```yaml
metadata:
  name: Retrieval Generation Prompt
  version: 1.0.0
  description: Retrieval Generation Prompt for RAG Pipeline
  author: Christoper Bischoff

prompts:
  retrieval_generation: |
    You are a shopping assistant...

    Context:
    {{ preprocessed_context }}

    Question:
    {{ question }}

  another_prompt: |
    Different prompt template here...
```

**Key Points**:
- `metadata:` section for documentation
- `prompts:` dictionary can contain multiple templates
- `|` operator preserves multiline formatting
- `{{ variable }}` for Jinja2 variable substitution

## Usage in RAG Pipeline

**Before (Hardcoded in retrieval_generation.py)**:
```python
def build_prompt(preprocessed_context, question):
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.
[... 60+ lines of hardcoded text ...]

Context:
{preprocessed_context}

Question:
{question}
"""
    return prompt
```

**After (Template-Based)**:
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

**Changes**:
- ❌ Removed 60+ lines of hardcoded prompt
- ✅ Added 8 lines of template loading
- ✅ Prompt now lives in YAML file (version control)
- ✅ Non-engineers can edit prompts without touching code

## Performance Considerations

### YAML Loading Overhead

**Per Request**:
- File I/O: ~1ms
- YAML parsing: ~1ms
- Template creation: <1ms
- **Total: ~3ms**

### Optimization: Template Caching

**Current** (loads YAML every request):
```python
def build_prompt(preprocessed_context, question):
    template = prompt_template_config(...)  # Loads YAML every time
    return template.render(...)
```

**Optimized** (cache template, load once):
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def prompt_template_config_cached(yaml_file, prompt_key):
    """Cached version: loads YAML once, reuses template."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt_key]
    template = Template(template_content)

    return template
```

**Impact**:
- First call: ~3ms (load + parse)
- Subsequent calls: <0.01ms (cache hit)
- FastAPI hot reload: Cache invalidates automatically

## Error Handling

### Common Errors

**1. File Not Found**:
```python
FileNotFoundError: [Errno 2] No such file or directory: 'prompts/retrieval_generation.yaml'
```

**Fix**: Use correct relative path from project root:
```python
yaml_file = "apps/api/src/api/agents/prompts/retrieval_generation.yaml"
```

**2. YAML Parsing Error**:
```python
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Fix**: Check YAML syntax (indentation, `|` for multilines)

**3. Missing Prompt Key**:
```python
KeyError: 'retrieval_generation'
```

**Fix**: Verify key exists in YAML's `prompts:` dictionary

**4. Missing Variables**:
```python
jinja2.exceptions.UndefinedError: 'preprocessed_context' is undefined
```

**Fix**: Provide all variables in `.render()` call

## Testing

### Unit Test for Template Loading

```python
def test_prompt_template_config():
    from api.agents.utils.prompt_management import prompt_template_config

    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )

    # Test rendering
    prompt = template.render(
        preprocessed_context="Test context",
        question="Test question"
    )

    # Assertions
    assert "Test context" in prompt
    assert "Test question" in prompt
    assert "shopping assistant" in prompt.lower()
```

### Integration Test with RAG Pipeline

```python
def test_build_prompt_with_template():
    from api.agents.retrieval_generation import build_prompt

    prompt = build_prompt(
        preprocessed_context="- Product A\n- Product B",
        question="What is Product A?"
    )

    assert "Product A" in prompt
    assert "Product B" in prompt
    assert len(prompt) > 100  # Ensure template was loaded
```

## Best Practices

### File Organization

```
apps/api/src/api/agents/prompts/
├── retrieval_generation.yaml       # RAG prompts
├── summarization.yaml              # Summary prompts
├── classification.yaml             # Classification prompts
└── README.md                       # Prompt documentation
```

### YAML Conventions

**1. Always Include Metadata**:
```yaml
metadata:
  name: Descriptive Name
  version: 1.0.0              # Semantic versioning
  description: What this prompt does
  author: Your Name
  created: 2026-01-26
  updated: 2026-01-26
```

**2. Use Semantic Versioning**:
- `1.0.0` → `1.0.1` for bug fixes (typos, clarifications)
- `1.0.0` → `1.1.0` for new features (new instructions)
- `1.0.0` → `2.0.0` for breaking changes (different output format)

**3. Document Variables**:
```yaml
# Variables used in this template:
# - preprocessed_context: Formatted product descriptions
# - question: User's question string
prompts:
  retrieval_generation: |
    Context:
    {{ preprocessed_context }}

    Question:
    {{ question }}
```

### Version Control

**Commit Messages**:
```bash
git commit -m "feat(prompts): update RAG prompt to include product ratings (v1.1.0)"
git commit -m "fix(prompts): correct typo in system instructions (v1.0.1)"
git commit -m "refactor(prompts): change output format to JSON (v2.0.0)"
```

**Git Workflow**:
1. Make changes to YAML file
2. Update version in metadata
3. Test with smoke test (`make smoke-test`)
4. Commit with descriptive message
5. Deploy (FastAPI hot reload picks up changes)

## Related Documentation

- **Parent Directory**: [../README.md](../README.md) - RAG pipeline implementation
- **Prompts Directory**: [../prompts/README.md](../prompts/README.md) - YAML configuration files
- **Notebook**: [../../../../notebooks/week2/05-Prompt-Versioning.ipynb](../../../../notebooks/week2/05-Prompt-Versioning.ipynb) - Learning progression
- **Project Root**: [../../../../README.md](../../../../README.md) - Overall architecture

## Further Reading

- **Jinja2 Documentation**: https://jinja.palletsprojects.com/
- **YAML Specification**: https://yaml.org/spec/1.2.2/
- **LangSmith Prompts**: https://docs.smith.langchain.com/prompts
- **Semantic Versioning**: https://semver.org/
