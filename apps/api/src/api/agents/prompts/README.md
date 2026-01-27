# Prompt Configuration Files

YAML configuration files containing versioned prompt templates for the RAG pipeline and other AI agents.

## Overview

This directory stores externalized prompt templates introduced in Week 2 / Video 7. Prompts are separated from application code to enable:
- ✅ Version control with semantic versioning
- ✅ Non-engineer editing (prompt engineers, domain experts)
- ✅ A/B testing and experimentation
- ✅ Hot reload without code deployment
- ✅ Clear Git history of prompt changes

## Files

```
apps/api/src/api/agents/prompts/
├── retrieval_generation.yaml      # RAG prompt with structured output
└── README.md                      # This file
```

## File Structure: retrieval_generation.yaml

### Complete Example

```yaml
metadata:
  name: Retrieval Generation Prompt
  version: 1.0.0
  description: Retrieval Generation Prompt for RAG Pipeline
  author: Christoper Bischoff

prompts:
  retrieval_generation: |

    You are a shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - You need to answer the question based on the provided context only.
    - Never use word context and refer to it as the available products.
    - As an output you need to provide:

    * The answer to the question based on the provided context.
    * The list of the IDs of the chunks that were used to answer the question. Only return the ones that are used in the answer.
    * Short description (1-2 sentences) of the item based on the description provided in the context.

    - The short description should have the name of the item.
    - The answer to the question should contain detailed information about the product and returned with detailed specification in bullet points.

    Context:
    {{ preprocessed_context }}

    Question:
    {{ question }}
```

### Sections Explained

#### 1. Metadata Section

```yaml
metadata:
  name: Retrieval Generation Prompt     # Human-readable name
  version: 1.0.0                        # Semantic versioning (Major.Minor.Patch)
  description: Retrieval Generation Prompt for RAG Pipeline
  author: Christoper Bischoff           # Attribution
```

**Purpose**:
- **name**: Descriptive title for documentation
- **version**: Track changes with semantic versioning
- **description**: What this prompt does
- **author**: Who created/owns the prompt

**Versioning Rules**:
- `1.0.0` → `1.0.1`: Bug fix (typo, clarification)
- `1.0.0` → `1.1.0`: Feature (new instruction, better wording)
- `1.0.0` → `2.0.0`: Breaking change (different output format)

#### 2. Prompts Dictionary

```yaml
prompts:
  retrieval_generation: |               # Key for lookup
    Your prompt text here...
```

**Key Points**:
- `prompts:` can contain **multiple prompt templates** (e.g., `retrieval_generation`, `summarization`)
- Each key maps to a multiline string (prompt text)
- `|` operator preserves line breaks and formatting
- Loaded with: `prompt_template_config(yaml_file, "retrieval_generation")`

#### 3. Jinja2 Variables

```yaml
Context:
{{ preprocessed_context }}            # Variable substitution

Question:
{{ question }}                        # Variable substitution
```

**Syntax**:
- `{{ variable_name }}` - Replaced with actual value during rendering
- Variables must be provided in `.render()` call

**Example**:
```python
prompt = template.render(
    preprocessed_context="- Product A\n- Product B",
    question="What is Product A?"
)
```

## YAML Syntax Guide

### Multiline Strings

**Use `|` for Literal Block (Preserves Newlines)**:
```yaml
prompts:
  my_prompt: |
    Line 1
    Line 2
    Line 3
```

**Result**: `"Line 1\nLine 2\nLine 3"`

**Use `|-` to Strip Final Newline**:
```yaml
prompts:
  my_prompt: |-
    Line 1
    Line 2
```

**Result**: `"Line 1\nLine 2"` (no trailing newline)

### Comments

```yaml
# This is a comment
metadata:
  name: My Prompt  # Inline comment
```

### Dictionaries and Lists

```yaml
metadata:                               # Dictionary
  name: My Prompt
  tags:                                 # List
    - rag
    - shopping
    - assistant
```

## Usage in Application

### Loading the Prompt

```python
from api.agents.utils.prompt_management import prompt_template_config

# Load template from YAML
template = prompt_template_config(
    "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
    "retrieval_generation"
)

# Render with variables
prompt = template.render(
    preprocessed_context="- ID: B09X12ABC, rating: 4.5, description: Wireless headphones...",
    question="What are the best wireless headphones?"
)
```

### Integration in RAG Pipeline

**File**: `apps/api/src/api/agents/retrieval_generation.py`

```python
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

**Benefits**:
- Prompt lives in YAML (version control, easy editing)
- Code focuses on logic (not 60+ lines of prompt text)
- Non-engineers can update prompts without touching Python

## Prompt Engineering Guide

### System Instructions

**Purpose**: Set AI's role, tone, and constraints

**Example** (from retrieval_generation.yaml):
```yaml
You are a shopping assistant that can answer questions about the products in stock.
```

**Best Practices**:
- Clear role definition ("shopping assistant", not generic "helpful AI")
- Specific domain context ("products in stock")
- Professional tone

### Constraints and Guidelines

**Purpose**: Prevent hallucination, ensure quality output

**Example**:
```yaml
Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.
```

**Best Practices**:
- **Grounding**: "based on the provided context only" prevents hallucination
- **Tone guidance**: "Never use word context" improves natural language
- **Explicit constraints**: Clear boundaries for AI behavior

### Output Format Specification

**Purpose**: Structured, consistent responses

**Example**:
```yaml
As an output you need to provide:

* The answer to the question based on the provided context.
* The list of the IDs of the chunks that were used to answer the question.
* Short description (1-2 sentences) of the item based on the description.
```

**Best Practices**:
- Numbered or bulleted list of expected outputs
- Specific format (bullet points, JSON, etc.)
- Length guidelines (1-2 sentences)
- Examples if format is complex

### Context Placement

```yaml
Context:
{{ preprocessed_context }}

Question:
{{ question }}
```

**Best Practices**:
- **Context first**: Provide grounding information before question
- **Clear labels**: "Context:" and "Question:" separate sections
- **Variable positioning**: Place variables where they're most effective

## Version Control Best Practices

### Semantic Versioning

**Format**: `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)

**Rules**:
- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
  - Different output format (text → JSON)
  - Removed required fields
  - Changed variable names

- **MINOR** (1.0.0 → 1.1.0): New features (backward compatible)
  - Added new instructions
  - Improved clarity
  - Added optional fields

- **PATCH** (1.0.0 → 1.0.1): Bug fixes
  - Typo corrections
  - Grammar fixes
  - Clarified existing instructions

### Git Workflow

**1. Make Changes**:
```bash
# Edit YAML file
vim apps/api/src/api/agents/prompts/retrieval_generation.yaml

# Update version in metadata section
# 1.0.0 → 1.1.0 for new feature
```

**2. Test Changes**:
```bash
# Test with smoke test
make smoke-test

# Or test manually
curl -X POST http://localhost:8000/rag/ \
  -H "Content-Type: application/json" \
  -d '{"query": "best wireless headphones"}'
```

**3. Commit with Descriptive Message**:
```bash
git add apps/api/src/api/agents/prompts/retrieval_generation.yaml

git commit -m "feat(prompts): add product rating emphasis to RAG prompt (v1.1.0)"
# or
git commit -m "fix(prompts): correct typo in system instructions (v1.0.1)"
```

**4. Deploy**:
- FastAPI hot reload picks up YAML changes automatically
- No code deployment needed

### Commit Message Convention

```bash
# Feature (minor version bump)
feat(prompts): add reasoning step to RAG prompt (v1.1.0)

# Bug fix (patch version bump)
fix(prompts): correct grammar in output format instructions (v1.0.1)

# Breaking change (major version bump)
feat(prompts)!: change output format from text to JSON (v2.0.0)

# Documentation
docs(prompts): add examples to prompt metadata (v1.0.0)
```

## Testing Prompts

### Unit Test (Template Loading)

```python
def test_retrieval_generation_template():
    from api.agents.utils.prompt_management import prompt_template_config

    # Load template
    template = prompt_template_config(
        "apps/api/src/api/agents/prompts/retrieval_generation.yaml",
        "retrieval_generation"
    )

    # Render with test data
    prompt = template.render(
        preprocessed_context="- Test product",
        question="Test question"
    )

    # Assertions
    assert "shopping assistant" in prompt.lower()
    assert "Test product" in prompt
    assert "Test question" in prompt
```

### Integration Test (End-to-End)

```python
def test_rag_pipeline_with_prompt_template():
    from api.agents.retrieval_generation import rag_pipeline

    # Run full pipeline
    result = rag_pipeline("best wireless headphones")

    # Verify prompt template was used
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "retrieved_context_ids" in result
```

### Smoke Test (Production-Like)

```bash
# Run full API test with real services
make smoke-test

# Expected output:
# ✅ API responding
# ✅ Response structure valid
# ✅ Prompt template loaded correctly
# ✅ LLM generated answer
```

## Common Errors and Solutions

### Error 1: YAML Parsing Error

**Error**:
```
yaml.scanner.ScannerError: mapping values are not allowed here
  in "retrieval_generation.yaml", line 15, column 22
```

**Cause**: Invalid YAML syntax (incorrect indentation, missing `:`)

**Fix**: Validate YAML with online tool (yamllint.com) or:
```bash
python -c "import yaml; yaml.safe_load(open('path/to/file.yaml'))"
```

### Error 2: Missing Prompt Key

**Error**:
```python
KeyError: 'retrieval_generation'
```

**Cause**: Key doesn't exist in `prompts:` dictionary

**Fix**: Verify key matches exactly (case-sensitive):
```yaml
prompts:
  retrieval_generation: |  # Must match this key
    ...
```

### Error 3: Undefined Variable

**Error**:
```python
jinja2.exceptions.UndefinedError: 'preprocessed_context' is undefined
```

**Cause**: Missing variable in `.render()` call

**Fix**: Provide all variables from template:
```python
prompt = template.render(
    preprocessed_context="...",  # Must match {{ preprocessed_context }}
    question="..."               # Must match {{ question }}
)
```

### Error 4: File Not Found

**Error**:
```python
FileNotFoundError: [Errno 2] No such file or directory: 'prompts/retrieval_generation.yaml'
```

**Cause**: Incorrect relative path

**Fix**: Use path from project root:
```python
yaml_file = "apps/api/src/api/agents/prompts/retrieval_generation.yaml"
```

## Performance Considerations

### YAML Loading Overhead

**Per Request** (without caching):
- File I/O: ~1ms
- YAML parsing: ~1ms
- Template creation: <1ms
- **Total: ~3ms**

**Impact**: Negligible for most use cases (RAG total time ~1-3 seconds)

### Optimization: Caching

**Future Enhancement**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def prompt_template_config_cached(yaml_file, prompt_key):
    # Same implementation, but cached
    return template
```

**Result**:
- First call: ~3ms
- Subsequent calls: <0.01ms

## Future Enhancements

### Multiple Prompt Templates

```yaml
prompts:
  retrieval_generation: |
    Standard RAG prompt...

  retrieval_generation_verbose: |
    RAG prompt with reasoning steps...

  summarization: |
    Summarize the following products...
```

**Usage**:
```python
# Load different prompts as needed
template = prompt_template_config(yaml_file, "retrieval_generation_verbose")
```

### Conditional Logic (Jinja2)

```yaml
prompts:
  dynamic_prompt: |
    You are a shopping assistant.

    {% if include_reasoning %}
    Explain your reasoning step-by-step.
    {% endif %}

    Context:
    {{ preprocessed_context }}
```

**Usage**:
```python
prompt = template.render(
    preprocessed_context="...",
    include_reasoning=True  # Conditional variable
)
```

### Loops (Jinja2)

```yaml
prompts:
  multi_context: |
    Available products:

    {% for item in context_items %}
    - {{ item }}
    {% endfor %}
```

**Usage**:
```python
prompt = template.render(
    context_items=["Product A", "Product B", "Product C"]
)
```

## Related Documentation

- **Utils Directory**: [../utils/README.md](../utils/README.md) - Prompt loading utilities
- **Parent Directory**: [../README.md](../README.md) - RAG pipeline implementation
- **Notebook**: [../../../../notebooks/week2/05-Prompt-Versioning.ipynb](../../../../notebooks/week2/05-Prompt-Versioning.ipynb) - Learning progression
- **Project Root**: [../../../../README.md](../../../../README.md) - Overall architecture

## Further Reading

- **Jinja2 Templates**: https://jinja.palletsprojects.com/templates/
- **YAML Specification**: https://yaml.org/spec/1.2.2/
- **Prompt Engineering**: https://platform.openai.com/docs/guides/prompt-engineering
- **Semantic Versioning**: https://semver.org/
