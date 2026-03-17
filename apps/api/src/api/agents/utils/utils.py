"""Agent utils: format_ai_message, get_tool_descriptions. Sprint 2 / Video 5; Week 4 multi-turn."""
import ast
import inspect
import re
from typing import Any

from langchain_core.messages import AIMessage, convert_to_openai_messages

# OpenAI tool names must match ^[a-zA-Z0-9_-]+$ (no dots). LLM may return "functions.tool_name".
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_tool_name(name: str) -> str:
    """Strip invalid prefixes (e.g. functions.) so OpenAI accepts the name."""
    if not name or _TOOL_NAME_PATTERN.match(name):
        return name
    if name.startswith("functions."):
        return name[len("functions.") :]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]


def messages_to_openai(messages) -> list:
    """Convert LangGraph messages to OpenAI format and sanitize tool names (checkpoint may have invalid names)."""
    out = []
    for message in messages:
        converted = convert_to_openai_messages(message)
        msgs = converted if isinstance(converted, list) else [converted]
        for m in msgs:
            if isinstance(m, dict) and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    if isinstance(tc.get("function"), dict) and "name" in tc["function"]:
                        tc["function"]["name"] = _sanitize_tool_name(tc["function"]["name"])
            out.append(m)
    return out


def format_ai_message(response, tool_call_id_prefix: str = "call"):
    """
    Convert AgentResponse to AIMessage. Includes tool_calls when agent wants to invoke tools.

    tool_call_id_prefix: Unique prefix per turn (e.g. call_0, call_1) to avoid OpenAI
    BadRequestError when same IDs reused across multi-turn conversation.
    """
    if response.tool_calls:
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            tool_calls.append({
                "id": f"{tool_call_id_prefix}_{i}",
                "name": _sanitize_tool_name(tc.name),
                "args": tc.arguments
            })

        ai_message = AIMessage(
            content=response.answer,
            tool_calls=tool_calls
        )
    else:
        ai_message = AIMessage(
            content=response.answer,
        )

    return ai_message


# --- Tool schema extraction: parse docstrings for agent's available_tools ---

def parse_function_definition(function_def: str) -> dict[str, Any]:
    """Parse a function definition string to extract metadata including type hints."""
    result = {
        "name": "",
        "description": "",
        "parameters": {"type": "object", "properties": {}},
        "required": [],
        "returns": {"type": "string", "description": ""}
    }

    # Parse the function using AST
    try:
        tree = ast.parse(function_def.strip())
    except SyntaxError:
        return result
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return result

    func = tree.body[0]
    result["name"] = func.name

    # Extract docstring
    docstring = ast.get_docstring(func) or ""
    param_descs = {}
    if docstring:
        # Extract description (first line/paragraph)
        desc_end = docstring.find('\n\n') if '\n\n' in docstring else docstring.find('\nArgs:')
        desc_end = desc_end if desc_end > 0 else docstring.find('\nParameters:')
        result["description"] = docstring[:desc_end].strip() if desc_end > 0 else docstring.strip()

        # Parse parameter descriptions
        param_descs = parse_docstring_params(docstring)

    # Extract return description
    if "Returns:" in docstring:
        result["returns"]["description"] = docstring.split("Returns:")[1].strip().split('\n')[0]

    # Extract parameters with type hints
    args = func.args
    defaults = args.defaults
    num_args = len(args.args)
    num_defaults = len(defaults)

    for i, arg in enumerate(args.args):
        if arg.arg == 'self':
            continue

        param_info = {
            "type": get_type_from_annotation(arg.annotation) if arg.annotation else "string",
            "description": param_descs.get(arg.arg, "")
        }

        # Check for default value
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0:
            try:
                param_info["default"] = ast.literal_eval(ast.unparse(defaults[default_idx]))
            except (ValueError, TypeError):
                pass
        else:
            result["required"].append(arg.arg)

        result["parameters"]["properties"][arg.arg] = param_info

    # Extract return type
    if func.returns:
        result["returns"]["type"] = get_type_from_annotation(func.returns)

    return result


def get_type_from_annotation(annotation) -> str:
    """Convert AST annotation to type string."""
    if not annotation:
        return "string"

    type_map = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'list': 'array',
        'dict': 'object',
    }

    if isinstance(annotation, ast.Name):
        return type_map.get(annotation.id, annotation.id)
    elif isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        base_type = annotation.value.id
        return type_map.get(base_type, base_type.lower())

    return "string"


def parse_docstring_params(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from docstring (handles both Args: and Parameters: formats)."""
    params = {}
    lines = docstring.split('\n')
    in_params = False
    current_param = None

    for line in lines:
        stripped = line.strip()

        # Check for parameter section start
        if stripped in ['Args:', 'Arguments:', 'Parameters:', 'Params:']:
            in_params = True
            current_param = None
        elif stripped.startswith('Returns:') or stripped.startswith('Raises:'):
            in_params = False
        elif in_params:
            # Parse parameter line (handles "param: desc" and "- param: desc" formats)
            if ':' in stripped and (stripped[0].isalpha() or stripped.startswith(('-', '*'))):
                param_name = stripped.lstrip('-*').split(':')[0].strip()
                param_desc = ':'.join(stripped.lstrip('-*').split(':')[1:]).strip()
                params[param_name] = param_desc
                current_param = param_name
            elif current_param and stripped:
                # Continuation of previous parameter description
                params[current_param] += ' ' + stripped

    return params


def get_tool_descriptions(function_list):
    """Extract tool schemas (name, description, parameters) from functions. Used as available_tools for agent."""
    descriptions = []

    for function in function_list:
        function_string = inspect.getsource(function)
        result = parse_function_definition(function_string)

        if result:
            descriptions.append(result)

    return descriptions if descriptions else []
