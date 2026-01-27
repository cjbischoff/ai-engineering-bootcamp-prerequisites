# Week 2 Prompt Templates (Learning Environment)

YAML prompt templates for experimentation and learning in Week 2 notebooks.

## Overview

This directory contains **duplicate copies** of production prompt templates for educational purposes. These files enable hands-on experimentation with prompt management in Jupyter notebooks without affecting the production API.

## Files

```
notebooks/week2/prompts/
├── retrieval_generation.yaml      # RAG prompt template (learning copy)
└── README.md                      # This file
```

## Purpose

**Why Duplicate Prompts?**

1. **Safe Experimentation**: Edit prompts in notebooks without breaking production
2. **Learning Environment**: Practice YAML syntax and Jinja2 templates
3. **Comparison**: Test prompt variations side-by-side
4. **Independence**: Notebooks work standalone (no API dependencies)

**Production vs Learning**:

| Aspect | Production | Learning (This Directory) |
|--------|------------|---------------------------|
| **Path** | `apps/api/src/api/agents/prompts/` | `notebooks/week2/prompts/` |
| **Purpose** | Used by FastAPI application | Used in Jupyter notebooks |
| **Editing** | Careful (affects production) | Experimental (safe to modify) |
| **Sync** | Keep in sync manually | Copy from production when needed |

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

**Note**: This is identical to the production version at `apps/api/src/api/agents/prompts/retrieval_generation.yaml`

## Usage in Notebooks

### Notebook: 05-Prompt-Versioning.ipynb

**Step 1: Import Libraries**
```python
import yaml
from jinja2 import Template
```

**Step 2: Create Loading Function**
```python
def prompt_template_config(yaml_file, prompt_key):
    """Load prompt from local YAML file."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt_key]
    template = Template(template_content)

    return template
```

**Step 3: Load and Render Template**
```python
# Load template from this directory
template = prompt_template_config(
    "notebooks/week2/prompts/retrieval_generation.yaml",
    "retrieval_generation"
)

# Render with test data
prompt = template.render(
    preprocessed_context="- Product A: Wireless headphones\n- Product B: USB cable",
    question="What are the best wireless headphones?"
)

print(prompt)
```

**Output**:
```
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
...

Context:
- Product A: Wireless headphones
- Product B: USB cable

Question:
What are the best wireless headphones?
```

## Learning Exercises

### Exercise 1: Modify Prompt Instructions

**Task**: Add a new instruction to include product ratings in the answer

**Steps**:
1. Edit `retrieval_generation.yaml` in this directory
2. Add instruction: "Always mention the product rating if available"
3. Test with `prompt_template_config()`
4. Compare output with original prompt

**Example Edit**:
```yaml
Instructions:
- You need to answer the question based on the provided context only.
- Always mention the product rating if available.
- Never use word context and refer to it as the available products.
```

### Exercise 2: Add New Variable

**Task**: Add a `max_products` variable to limit recommendations

**Steps**:
1. Edit prompt to include: `Show up to {{ max_products }} products.`
2. Update `.render()` call: `max_products=3`
3. Verify variable substitution works

**Example**:
```yaml
prompts:
  retrieval_generation: |
    You are a shopping assistant.

    Show up to {{ max_products }} products.

    Context:
    {{ preprocessed_context }}
```

```python
prompt = template.render(
    preprocessed_context="...",
    question="...",
    max_products=3  # New variable
)
```

### Exercise 3: Create Prompt Variant

**Task**: Create a verbose version with reasoning steps

**Steps**:
1. Add new key to YAML: `retrieval_generation_verbose`
2. Copy original prompt and add: "Explain your reasoning step-by-step."
3. Load with: `prompt_template_config(yaml_file, "retrieval_generation_verbose")`

**Example**:
```yaml
prompts:
  retrieval_generation: |
    Standard prompt...

  retrieval_generation_verbose: |
    You are a shopping assistant.

    Explain your reasoning step-by-step.

    Context:
    {{ preprocessed_context }}

    Question:
    {{ question }}
```

### Exercise 4: A/B Testing

**Task**: Compare two prompt versions side-by-side

**Steps**:
1. Create two prompts in YAML (`version_a`, `version_b`)
2. Load both templates
3. Render with same test data
4. Compare outputs and quality

**Example**:
```python
# Load both versions
template_a = prompt_template_config(yaml_file, "retrieval_generation")
template_b = prompt_template_config(yaml_file, "retrieval_generation_verbose")

# Test with same data
test_context = "- Product A: Wireless headphones with noise cancellation, rating: 4.5"
test_question = "What are the best wireless headphones?"

prompt_a = template_a.render(preprocessed_context=test_context, question=test_question)
prompt_b = template_b.render(preprocessed_context=test_context, question=test_question)

# Compare
print("Version A:", prompt_a)
print("\nVersion B:", prompt_b)
```

## YAML Syntax Learning

### Multiline Strings

**Literal Block (`|`) - Preserves Newlines**:
```yaml
prompts:
  my_prompt: |
    Line 1
    Line 2
    Line 3
```
**Result**: `"Line 1\nLine 2\nLine 3"`

**Folded Block (`>`) - Joins Lines**:
```yaml
prompts:
  my_prompt: >
    This is a very long
    sentence that spans
    multiple lines.
```
**Result**: `"This is a very long sentence that spans multiple lines."`

**Literal Block Chomping (`|-`) - Strip Final Newline**:
```yaml
prompts:
  my_prompt: |-
    No trailing newline
```

### Comments

```yaml
# Full line comment
metadata:
  name: My Prompt  # Inline comment
```

### Lists and Dictionaries

```yaml
metadata:
  name: My Prompt
  tags:                         # List
    - rag
    - shopping
  variables:                    # Dictionary
    context: "Product descriptions"
    question: "User query"
```

## Jinja2 Template Learning

### Variables

**Syntax**: `{{ variable_name }}`

**Example**:
```jinja
Hello, {{ user_name }}!
Your order total is ${{ total_price }}.
```

**Render**:
```python
prompt = template.render(user_name="Alice", total_price=99.99)
# Output: "Hello, Alice!\nYour order total is $99.99."
```

### Conditionals

**Syntax**: `{% if condition %} ... {% endif %}`

**Example**:
```jinja
{% if include_rating %}
Product rating: {{ rating }}
{% endif %}
```

**Render**:
```python
# With rating
prompt = template.render(include_rating=True, rating=4.5)
# Output: "Product rating: 4.5"

# Without rating
prompt = template.render(include_rating=False)
# Output: "" (empty)
```

### Loops

**Syntax**: `{% for item in items %} ... {% endfor %}`

**Example**:
```jinja
Available products:
{% for product in products %}
- {{ product.name }} ({{ product.price }})
{% endfor %}
```

**Render**:
```python
prompt = template.render(products=[
    {"name": "Headphones", "price": "$99"},
    {"name": "Cable", "price": "$10"}
])
# Output:
# Available products:
# - Headphones ($99)
# - Cable ($10)
```

### Filters

**Syntax**: `{{ variable | filter }}`

**Example**:
```jinja
{{ product_name | upper }}
{{ description | truncate(50) }}
{{ price | round(2) }}
```

**Common Filters**:
- `upper` / `lower` - Case conversion
- `title` - Title case
- `truncate(n)` - Limit string length
- `round(n)` - Round numbers
- `default(value)` - Fallback if undefined

## Best Practices for Experimentation

### 1. Keep Backup of Original

```bash
# Before experimenting
cp retrieval_generation.yaml retrieval_generation.yaml.backup

# Restore if needed
cp retrieval_generation.yaml.backup retrieval_generation.yaml
```

### 2. Use Version Control

```bash
# Commit baseline
git add retrieval_generation.yaml
git commit -m "docs: baseline prompt for experimentation"

# Make changes, test

# Revert if needed
git checkout -- retrieval_generation.yaml
```

### 3. Document Changes

```yaml
metadata:
  name: Retrieval Generation Prompt (Experiment: Concise Answers)
  version: 1.1.0-experiment
  description: Testing shorter, more concise answer format
  author: Your Name
  experiment_date: 2026-01-26
```

### 4. Test with Realistic Data

```python
# Bad: Too simple
test_context = "Product A"

# Good: Realistic complexity
test_context = """
- ID: B09X12ABC, rating: 4.5, description: Sony WH-1000XM5 Wireless Headphones with Industry Leading Noise Cancellation
- ID: B08Y34DEF, rating: 4.3, description: Bose QuietComfort 45 Bluetooth Headphones with Active Noise Cancellation
"""
```

### 5. Measure Quality

**Manual Evaluation**:
- Is the answer accurate?
- Is it well-formatted?
- Does it follow instructions?
- Is it concise yet informative?

**Automated Evaluation** (future):
- RAGAS metrics (faithfulness, relevance)
- LLM-as-judge
- User feedback scores

## Common Errors (Learning Edition)

### Error 1: Forgot to Install Libraries

**Error**:
```python
ModuleNotFoundError: No module named 'yaml'
```

**Fix**:
```bash
pip install pyyaml jinja2
# or
uv add pyyaml jinja2
```

### Error 2: Wrong File Path

**Error**:
```python
FileNotFoundError: [Errno 2] No such file or directory: 'prompts/retrieval_generation.yaml'
```

**Fix**: Use correct relative path from notebook location
```python
# ✅ Correct (from notebooks/week2/)
yaml_file = "notebooks/week2/prompts/retrieval_generation.yaml"

# ✅ Also correct (relative to current notebook)
yaml_file = "prompts/retrieval_generation.yaml"
```

### Error 3: YAML Indentation Error

**Error**:
```yaml
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Fix**: Check indentation (use 2 spaces, not tabs)
```yaml
# ❌ Wrong
prompts:
retrieval_generation: |
  Text here

# ✅ Right
prompts:
  retrieval_generation: |
    Text here
```

### Error 4: Missing Variable in Template

**Error**:
```python
jinja2.exceptions.UndefinedError: 'preprocessed_context' is undefined
```

**Fix**: Provide all variables in `.render()`
```python
# ❌ Wrong: Missing variable
prompt = template.render(question="What are the best headphones?")

# ✅ Right: All variables provided
prompt = template.render(
    preprocessed_context="...",
    question="What are the best headphones?"
)
```

## Syncing with Production

**When to Sync**:
- Production prompt updated (new features)
- Bug fixes in production prompt
- Major refactoring

**How to Sync**:
```bash
# Copy production prompt to notebooks
cp apps/api/src/api/agents/prompts/retrieval_generation.yaml \
   notebooks/week2/prompts/retrieval_generation.yaml

# Or compare and merge changes
diff apps/api/src/api/agents/prompts/retrieval_generation.yaml \
     notebooks/week2/prompts/retrieval_generation.yaml
```

**Important**: Notebook prompts may diverge for experimentation - that's OK!

## Related Documentation

- **Production Prompts**: [../../../apps/api/src/api/agents/prompts/README.md](../../../apps/api/src/api/agents/prompts/README.md) - Production YAML files
- **Prompt Utilities**: [../../../apps/api/src/api/agents/utils/README.md](../../../apps/api/src/api/agents/utils/README.md) - Loading functions
- **Week 2 Notebooks**: [../README.md](../README.md) - All Week 2 learning materials
- **Notebook**: [../05-Prompt-Versioning.ipynb](../05-Prompt-Versioning.ipynb) - Prompt management tutorial
- **Project Root**: [../../README.md](../../README.md) - Overall architecture

## Further Reading

- **Jinja2 Documentation**: https://jinja.palletsprojects.com/templates/
- **YAML Tutorial**: https://yaml.org/start.html
- **YAML Multiline Strings**: https://yaml-multiline.info/
- **Prompt Engineering Guide**: https://platform.openai.com/docs/guides/prompt-engineering
