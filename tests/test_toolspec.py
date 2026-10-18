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


OFFICIAL_SYSTEM = """⟦SYSTEM⟧
You are an airline agent.
<policy>
Current time is 2024-05-15 15:00:00 EST.
</policy>

[AVAILABLE TOOLS]
- book_reservation — Book a reservation for a customer
    origin: string! — The origin airport code
    destination: string!
    flights: array! — The list of flights
        · flight_number: string!
        · date: string!
        · price: integer!
    payment_methods: array!
        · payment_id: string!
        · amount: integer!
    cabin: string! [enum: basic_economy|economy|business] — Cabin class
    insurance: string [enum: yes|no]
- get_user_details — Get the details of a user
    user_id: string!
"""


def test_official_catalogue_is_parsed():
    specs = extract_available_tools(OFFICIAL_SYSTEM)
    assert set(specs) == {"book_reservation", "get_user_details"}
    book = specs["book_reservation"]
    assert book.source == "text-official"
    assert set(book.params) == {
        "origin", "destination", "flights", "payment_methods", "cabin", "insurance",
    }
    assert book.params["origin"].required is True
    assert book.params["insurance"].required is False
    assert book.params["insurance"].enum == ["yes", "no"]
    assert book.params["cabin"].enum == ["basic_economy", "economy", "business"]
    # nested fields are attached to their parent, not treated as top-level
    assert set(book.params["flights"].children) == {"flight_number", "date", "price"}
    assert book.params["flights"].children["price"].type == "integer"


def test_official_schema_validation():
    specs = extract_available_tools(OFFICIAL_SYSTEM)
    book = specs["book_reservation"]
    good = {
        "origin": "JFK",
        "destination": "SFO",
        "flights": [{"flight_number": "HAT023", "date": "2024-05-26", "price": 120}],
        "payment_methods": [{"payment_id": "credit_card_1", "amount": 120}],
        "cabin": "business",
    }
    assert validate_tool_call(book, good) == []

    missing = dict(good)
    missing.pop("origin")
    assert any("origin" in issue for issue in validate_tool_call(book, missing))

    bad_enum = dict(good, cabin="first")
    assert any("cabin" in issue for issue in validate_tool_call(book, bad_enum))

    invented = dict(good, payment_methods=[{"id": "c1", "source": "certificate"}])
    issues = validate_tool_call(book, invented)
    assert any("payment_id" in issue for issue in issues)
    assert any("amount" in issue for issue in issues)


def test_official_signature_rendering():
    from guardian.toolspec import render_signature

    specs = extract_available_tools(OFFICIAL_SYSTEM)
    text = render_signature(specs["book_reservation"])
    assert text.startswith("book_reservation(")
    assert "origin: string!" in text
    assert "cabin: string![basic_economy|economy|business]" in text
    assert "payment_id: string!" in text
