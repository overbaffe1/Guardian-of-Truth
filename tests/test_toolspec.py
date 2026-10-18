from guardian.toolspec import (
    extract_available_tools,
    extract_tools_block,
    validate_tool_call,
)

JSON_SYSTEM = """⟦SYSTEM⟧
You are an agent.
<policy>some policy</policy>
[AVAILABLE TOOLS]
[
  {"name": "get_user_details",
   "description": "Fetch a user profile",
   "parameters": {"type": "object",
                  "properties": {"user_id": {"type": "string"}},
                  "required": ["user_id"]}},
  {"type": "function",
   "function": {"name": "update_reservation_flights",
                "description": "Change flights",
                "parameters": {"type": "object",
                               "properties": {
                                  "reservation_id": {"type": "string"},
                                  "cabin": {"type": "string", "enum": ["basic_economy", "economy", "business"]},
                                  "flights": {"type": "array", "items": {"type": "object"}}},
                               "required": ["reservation_id", "cabin", "flights"],
                               "additionalProperties": false}}}
]
"""

TEXT_SYSTEM = """⟦SYSTEM⟧
You are a banking agent.
[AVAILABLE TOOLS]
Tool: search_knowledge_base
Description: Search internal documents.
Parameters:
  - query: string (required) - search query
  - top_k: integer (optional) - number of results

Tool: open_bank_account_4821
Description: Open a bank account.
Parameters:
  - customer_id: string (required) - customer identifier
  - account_class: string (required) - official account class name
"""


def test_extract_json_schemas():
    specs = extract_available_tools(JSON_SYSTEM)
    assert set(specs) == {"get_user_details", "update_reservation_flights"}
    flights = specs["update_reservation_flights"]
    assert flights.params["reservation_id"].required
    assert flights.params["cabin"].enum == ["basic_economy", "economy", "business"]
    assert flights.additional_properties is False


def test_extract_text_schemas():
    specs = extract_available_tools(TEXT_SYSTEM)
    assert set(specs) == {"search_knowledge_base", "open_bank_account_4821"}
    assert specs["search_knowledge_base"].params["top_k"].required is False
    assert specs["open_bank_account_4821"].params["customer_id"].required is True


def test_tools_block_cut_at_next_section():
    system = "[AVAILABLE TOOLS]\n- a_tool: does a\n- b_tool: does b\n\n[MEMORY]\nsomething else"
    block = extract_tools_block(system)
    assert "a_tool" in block and "b_tool" in block
    assert "MEMORY" not in block


def test_validate_missing_required():
    specs = extract_available_tools(JSON_SYSTEM)
    spec = specs["update_reservation_flights"]
    issues = validate_tool_call(spec, {"reservation_id": "X", "cabin": "economy"})
    assert any("flights" in issue and "missing" in issue for issue in issues)


def test_validate_type_and_enum():
    specs = extract_available_tools(JSON_SYSTEM)
    spec = specs["update_reservation_flights"]
    issues = validate_tool_call(
        spec,
        {
            "reservation_id": "X",
            "cabin": "first",
            "flights": "HAT169",
            "surprise": 1,
        },
    )
    joined = " | ".join(issues)
    assert "not one of" in joined
    assert "expected array" in joined
    assert "unknown argument" in joined


def test_numeric_string_is_tolerated():
    specs = extract_available_tools(TEXT_SYSTEM)
    spec = specs["search_knowledge_base"]
    assert validate_tool_call(spec, {"query": "x", "top_k": "5"}) == []


def test_spec_source_tracks_parser():
    json_specs = extract_available_tools(JSON_SYSTEM)
    text_specs = extract_available_tools(TEXT_SYSTEM)
    assert json_specs["get_user_details"].source == "json"
    assert text_specs["search_knowledge_base"].source == "text"
