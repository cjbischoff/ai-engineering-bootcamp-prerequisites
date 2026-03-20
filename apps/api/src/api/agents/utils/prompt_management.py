"""
Prompt loading utilities (Week 2 Video 7 / Prompt Versioning).

Loads prompts from YAML files with Jinja2 templates. Used by agents.py for
product_qa_agent, shopping_cart_agent, and coordinator_agent. Separates prompt
text from code for version control, A/B testing, and non-engineer edits.
prompt_template_registry pulls from LangSmith cloud (optional).
"""
import logging
from pathlib import Path

import yaml
from jinja2 import Template
from langsmith import Client

logger = logging.getLogger(__name__)
ls_client = Client()


def prompt_template_config(yaml_file, prompt_key):
    """Load prompt from YAML; return Jinja2 Template. Variables: {{ var }} in YAML."""
    path = Path(yaml_file)
    logger.debug("Loading prompt: file=%s key=%s", path.resolve(), prompt_key)

    if not path.exists():
        prompts_dir = path.parent
        available = list(prompts_dir.glob("*.yaml")) if prompts_dir.exists() else []
        logger.error(
            "Prompt file not found: %s (available in dir: %s)",
            path,
            [p.name for p in available],
        )
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    if prompt_key not in config.get("prompts", {}):
        logger.error(
            "Prompt key '%s' not in YAML; available keys: %s",
            prompt_key,
            list(config.get("prompts", {}).keys()),
        )
        raise KeyError(f"Prompt key '{prompt_key}' not found in {path}")

    template_content = config["prompts"][prompt_key]
    template = Template(template_content)
    logger.debug("Loaded prompt: %s/%s", path.name, prompt_key)

    return template


def prompt_template_registry(prompt_name):
    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template

    template = Template(template_content)

    return template
