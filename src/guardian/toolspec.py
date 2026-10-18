"""Extraction of the ``[AVAILABLE TOOLS]`` catalogue from the system prompt.

The catalogue is used for two purposes:

1. detecting calls to tools that do not exist (an explicit hallucination
   category in the task statement);
2. validating arguments (missing required fields, wrong types, enum values).

Because the renderer is not under our control, the parser understands JSON
schemas as well as the plain-text ``Tool: name / Parameters:`` layout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{1,64}$")
_NOT_TOOLS = {
    "name", "description", "parameters", "args", "arguments", "returns", "type",
    "properties", "required", "function", "tools", "input_schema", "json",
    "schema", "string", "integer", "number", "boolean", "array", "object",
}


@dataclass
class ParamSpec:
    name: str
    type: str = "string"
    required: bool = True
    enum: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    params: dict[str, ParamSpec] = field(default_factory=dict)
    additional_properties: Optional[bool] = None

    def brief(self, max_desc: int = 100) -> str:
        desc = " ".join((self.description or "").split())
        if len(desc) > max_desc:
            desc = desc[: max_desc - 1] + "…"
        return f"- {self.name}: {desc}" if desc else f"- {self.name}"


def extract_tools_block(system_text: str) -> str:
    """Return the ``[AVAILABLE TOOLS]`` section (or "" when absent)."""
    if not system_text:
        return ""
    m = re.search(r"\[\s*AVAILABLE[\s_]*TOOLS\s*\]", system_text, re.IGNORECASE)
    if not m:
        m = re.search(r"AVAILABLE[\s_]*TOOLS\s*[:\-]", system_text, re.IGNORECASE)
    if not m:
        return ""
    rest = system_text[m.end() :]
    # Cut at the next top-level section header.
    cut = re.search(r"\n\s*\[[A-Z][^\]]{2,40}\]\s*\n", rest)
    if cut:
        rest = rest[: cut.start()]
    return rest.strip()


def _add_spec(specs: dict[str, ToolSpec], spec: ToolSpec) -> None:
    if spec.name and _NAME_RE.match(spec.name) and spec.name.lower() not in _NOT_TOOLS:
        # Prefer the entry that carries more information.
        if spec.name not in specs or len(spec.params) > len(specs[spec.name].params):
            specs[spec.name] = spec


def _spec_from_json_schema(obj: dict[str, Any], fallback_name: str = "") -> Optional[ToolSpec]:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool_name") or fallback_name
    if not name:
        return None
    schema = (
        obj.get("parameters")
        or obj.get("input_schema")
        or obj.get("args_schema")
        or obj.get("arguments")
        or {}
    )
    if not isinstance(schema, dict):
        schema = {}
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") or []
    required_set = set(required) if isinstance(required, list) else set()
    params: dict[str, ParamSpec] = {}
    for pname, pdef in props.items():
        if not isinstance(pdef, dict):
            pdef = {}
        ptype = pdef.get("type", "string")
        if isinstance(ptype, list):
            ptype = next((t for t in ptype if t != "null"), "string")
        enum = pdef.get("enum") or []
        params[str(pname)] = ParamSpec(
            name=str(pname),
            type=str(ptype),
            required=str(pname) in required_set,
            enum=[str(x) for x in enum] if isinstance(enum, list) else [],
            description=str(pdef.get("description", ""))[:300],
        )
    add_props = schema.get("additionalProperties")
    return ToolSpec(
        name=str(name),
        description=str(obj.get("description", ""))[:500],
        params=params,
        additional_properties=add_props if isinstance(add_props, bool) else None,
    )


def _parse_json_catalogue(block: str) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    decoder = json.JSONDecoder()
    # Try the whole block first, then every JSON value inside it.
    candidates: list[Any] = []
    stripped = block.strip().strip("`")
    for text in (stripped, stripped[stripped.find("[") :] if "[" in stripped else stripped,
                 stripped[stripped.find("{") :] if "{" in stripped else stripped):
        try:
            candidates.append(json.loads(re.sub(r",\s*([}\]])", r"\1", text)))
        except Exception:
            pass

    # Scan for individual JSON objects (works for bullet lists of schemas).
    idx = 0
    while idx < len(block):
        pos = block.find("{", idx)
        if pos < 0:
            break
        try:
            value, end = decoder.raw_decode(block[pos:])
            candidates.append(value)
            idx = pos + end
        except Exception:
            idx = pos + 1

    for cand in candidates:
        items: list[Any]
        if isinstance(cand, dict):
            if isinstance(cand.get("tools"), list):
                items = cand["tools"]
            elif isinstance(cand.get("functions"), list):
                items = cand["functions"]
            elif cand.get("name"):
                items = [cand]
            else:
                # mapping name -> schema
                items = []
                for key, value in cand.items():
                    if isinstance(value, dict) and _NAME_RE.match(str(key)):
                        spec = _spec_from_json_schema(value, fallback_name=str(key))
                        if spec:
                            _add_spec(specs, spec)
                continue
        elif isinstance(cand, list):
            items = cand
        else:
            continue
        for item in items:
            if isinstance(item, str):
                _add_spec(specs, ToolSpec(name=item))
            elif isinstance(item, dict):
                spec = _spec_from_json_schema(item)
                if spec:
                    _add_spec(specs, spec)
    return specs


def _parse_text_catalogue(block: str) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    lines = block.split("\n")

    current: Optional[ToolSpec] = None
    in_params = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        header = re.match(r"^\s*(?:Tool|Function)\s*[:\-]\s*`?([A-Za-z_][\w.\-]*)`?", line, re.I)
        if not header:
            header = re.match(r"^\s*#{2,4}\s*`?([A-Za-z_][\w.\-]*)`?\s*(?:\(|:)?", line)
        if not header:
            header = re.match(r"^\s*[-*•]\s*`?([A-Za-z_][\w.\-]{2,})`?\s*(?:\(|:)", line)
        is_bullet = line.lstrip().startswith(("-", "*", "•"))
        if (
            header
            and header.group(1).lower() not in _NOT_TOOLS
            and not (is_bullet and (in_params or (current and header.group(1) in current.params)))
        ):
            name = header.group(1)
            current = ToolSpec(name=name)
            specs.setdefault(name, current)
            in_params = False
            # Description may follow on the same line after "- name: text"
            after = line[header.end() :].strip(" :-")
            if after and not after.lower().startswith(("description", "parameters")):
                current.description = after[:300]
            continue

        if current is None:
            continue

        if re.match(r"^\s*description\s*:", stripped, re.I):
            current.description = stripped.split(":", 1)[1].strip()[:500]
            in_params = False
            continue
        if re.match(r"^\s*(parameters|arguments|params|input)\s*:", stripped, re.I):
            in_params = True
            continue

        param = _parse_param_line(stripped)
        if param and (in_params or param.required):
            if param.name.lower() not in _NOT_TOOLS:
                current.params[param.name] = param

    return specs


_PARAM_RE = re.compile(
    r"^\s*[-*•]?\s*`?([A-Za-z_][\w\-]*)`?\s*(?:\(([^)]*)\))?\s*(?::|-)?\s*(.*)$"
)


def _parse_param_line(line: str) -> Optional[ParamSpec]:
    m = _PARAM_RE.match(line)
    if not m:
        return None
    name, paren, rest = m.group(1), (m.group(2) or ""), (m.group(3) or "")
    if name.lower() in _NOT_TOOLS:
        return None
    combined = f"{paren} {rest}".lower()
    ptype = "string"
    for known in ("array", "object", "integer", "number", "boolean", "string"):
        if re.search(rf"\b{known}\b", combined):
            ptype = known
            break
    required = True
    if "optional" in combined:
        required = False
    if re.search(r"\(required\)", combined):
        required = True
    enum: list[str] = []
    em = re.search(r"enum\s*[:=]\s*\[([^\]]+)\]", rest, re.I)
    if em:
        enum = [x.strip().strip("'\"") for x in em.group(1).split(",") if x.strip()]
    desc = re.sub(r"\b(required|optional)\b", "", rest).strip(" -:")[:200]
    return ParamSpec(name=name, type=ptype, required=required, enum=enum, description=desc)


def extract_available_tools(system_text: str) -> dict[str, ToolSpec]:
    """Best-effort catalogue of tools declared in the system prompt."""
    block = extract_tools_block(system_text)
    if not block:
        return {}
    specs = _parse_json_catalogue(block)
    if len(specs) < 2:
        text_specs = _parse_text_catalogue(block)
        for name, spec in text_specs.items():
            if name not in specs or len(spec.params) > len(specs[name].params):
                specs[name] = spec
    return specs


def tool_names(specs: dict[str, ToolSpec]) -> list[str]:
    return sorted(specs)


def catalogue_text(
    specs: dict[str, ToolSpec],
    max_tools: int = 90,
    max_desc: int = 90,
    max_chars: int | None = None,
) -> str:
    """Compact tool catalogue: *all* names first, then descriptions.

    Names are the critical part (tool availability), so they are never dropped
    in favour of descriptions when the budget is tight.
    """
    if not specs:
        return "(tool catalogue not found in the context)"
    names = ", ".join(sorted(specs))
    header = f"Tool names ({len(specs)}): {names}"
    if max_chars is not None and len(header) > max_chars:
        return header[: max(200, max_chars - 3)] + "..."

    lines = [header]
    used = len(header) + 1
    for _, spec in sorted(specs.items())[:max_tools]:
        line = spec.brief(max_desc)
        if max_chars is not None and used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    if len(specs) > max_tools:
        lines.append(f"... ({len(specs) - max_tools} more tools)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _coerce_like(value: Any, expected: str) -> Any:
    if expected in ("integer", "number") and isinstance(value, str):
        try:
            return float(value) if expected == "number" else int(float(value))
        except Exception:
            return value
    return value


def validate_tool_call(spec: ToolSpec, args: Optional[dict[str, Any]]) -> list[str]:
    """Return a list of schema problems (empty when the call looks valid)."""
    if args is None:
        return []
    issues: list[str] = []

    for pname, pspec in spec.params.items():
        if pspec.required and pname not in args:
            issues.append(f"missing required argument '{pname}'")

    for pname, value in args.items():
        pspec = spec.params.get(pname)
        if pspec is None:
            if spec.additional_properties is False:
                issues.append(f"unknown argument '{pname}'")
            continue
        expected = (pspec.type or "").lower()
        py_types = _TYPE_MAP.get(expected)
        if py_types:
            value_checked = _coerce_like(value, expected)
            if expected == "boolean" and isinstance(value_checked, (int, float)):
                # JSON true/false only; numbers are a clear type error
                issues.append(f"argument '{pname}' must be boolean")
            elif not isinstance(value_checked, py_types):
                issues.append(
                    f"argument '{pname}' has type {type(value).__name__}, expected {expected}"
                )
        if pspec.enum:
            norm_enum = {str(x).strip().lower() for x in pspec.enum}
            if str(value).strip().lower() not in norm_enum:
                issues.append(
                    f"argument '{pname}' value {value!r} is not one of {pspec.enum}"
                )
    return issues


def parse_tool_list_from_context(text: str) -> set[str]:
    """Fallback: extract tool-like identifiers from an arbitrary tools block."""
    names: set[str] = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or ""):
        if "_" in token and token.lower() not in _NOT_TOOLS:
            names.add(token)
    return names
