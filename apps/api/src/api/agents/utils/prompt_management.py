"""
Prompt loading utilities (Week 2 Video 7 / Prompt Versioning).

Loads prompts from YAML files with Jinja2 templates. Separates prompt text from code
for version control, A/B testing, and non-engineer edits. prompt_template_registry
pulls from LangSmith cloud (optional).
"""
import yaml
from jinja2 import Template
from langsmith import Client

ls_client = Client()


def prompt_template_config(yaml_file, prompt_key):
    """Load prompt from YAML; return Jinja2 Template. Variables: {{ var }} in YAML."""
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt_key]

    template = Template(template_content)

    return template


def prompt_template_registry(prompt_name):
    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template

    template = Template(template_content)

    return template
