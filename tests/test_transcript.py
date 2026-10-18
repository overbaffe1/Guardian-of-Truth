from guardian.transcript import (
    detect_domain,
    extract_policy,
    parse_response,
    parse_transcript,
)

PROMPT = """⟦SYSTEM⟧
You are a customer service agent.
<policy>
# Airline Agent Policy
Before taking any actions that update the booking database you must list the action
details and obtain explicit user confirmation (yes) to proceed.
</policy>

[AVAILABLE TOOLS]
- get_reservation_details: read a reservation
- update_reservation_flights: change flights
- book_reservation: create a booking
⟦USER⟧
I want to change my flight.
⟦ASSISTANT · ход 1⟧
Sure, what is your reservation id?
⟦USER⟧
It is EHGLP3.
⟦ASSISTANT · ход 2⟧
	→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}
	← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", "user_id": "emma_kim_9957", "cabin": "economy"}
"""


def test_parse_basic_transcript():
    t = parse_transcript(PROMPT)
    kinds = [e.kind for e in t.events]
    assert kinds == [
        "system",
        "user",
        "assistant",
        "user",
        "tool_call",
        "tool_response",
    ]
    assert t.users[-1].content.strip() == "It is EHGLP3."
    call = t.tool_calls[0].tool_call
    assert call.name == "get_reservation_details"
    assert call.args == {"reservation_id": "EHGLP3"}
    assert call.parse_ok
    assert t.tool_responses[0].tool_name == "get_reservation_details"
    assert "emma_kim_9957" in t.tool_responses[0].content
    assert t.domain == "airline"


def test_policy_extraction():
    t = parse_transcript(PROMPT)
    policy = extract_policy(t.system_text)
    assert policy.startswith("# Airline Agent Policy")
    assert "confirmation" in policy


def test_parse_response_with_marker_and_call():
    response = parse_response(
        '⟦ASSISTANT · ход 3⟧\nSure, updating now.\n'
        '\t→ TOOL_CALL update_reservation_flights: {"reservation_id": "EHGLP3", "cabin": "business"}'
    )
    assert response.turn == 3
    assert "updating" in response.text
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].args["cabin"] == "business"


def test_multiline_json_arguments_and_response():
    prompt = (
        "⟦SYSTEM⟧\npolicy\n[AVAILABLE TOOLS]\n- book_reservation: x\n"
        "⟦USER⟧\nbook it\n"
        "⟦ASSISTANT · ход 1⟧\n"
        "\t→ TOOL_CALL book_reservation: {\n"
        '  "user_id": "mia_kim_4397",\n'
        '  "flights": [{"flight_number": "HAT169", "date": "2024-05-20"}]\n'
        "}\n"
        "← TOOL_RESPONSE book_reservation: {\n"
        '  "reservation_id": "ABC123"\n'
        "}\n"
    )
    t = parse_transcript(prompt)
    assert t.tool_calls[0].tool_call.parse_ok
    assert t.tool_calls[0].tool_call.args["user_id"] == "mia_kim_4397"
    assert "ABC123" in t.tool_responses[0].content


def test_multiline_plain_text_tool_response():
    prompt = (
        "⟦SYSTEM⟧\npolicy\n"
        "⟦USER⟧\nwhat does the doc say?\n"
        "⟦ASSISTANT · ход 1⟧\n"
        "\t→ TOOL_CALL shell: {\"command\": \"cat doc.md\"}\n"
        "\t← TOOL_RESPONSE shell: ## Eligibility\n"
        "\t   The customer must be verified.\n"
        "⟦ASSISTANT · ход 2⟧\nDone.\n"
    )
    t = parse_transcript(prompt)
    assert len(t.tool_responses) == 1
    assert "Eligibility" in t.tool_responses[0].content
    assert "must be verified" in t.tool_responses[0].content
    # The following assistant turn must not be swallowed.
    assert any(e.kind == "assistant" and e.content.strip() == "Done." for e in t.events)


def test_response_without_marker():
    response = parse_response('→ TOOL_CALL transfer_to_human_agents: {"summary": "x"}')
    assert len(response.tool_calls) == 1
    assert response.text == ""


def test_latin_turn_marker_is_supported():
    t = parse_transcript(
        "⟦SYSTEM⟧\npolicy\n⟦USER⟧\nhi\n⟦ASSISTANT · turn 4⟧\nhello\n"
    )
    assistant = t.assistant_texts[0]
    assert assistant.turn == 4


def test_domain_detection_retail():
    prompt = "⟦SYSTEM⟧\n# Retail Agent Policy\norders and product exchanges\n⟦USER⟧\nhi\n"
    assert detect_domain(parse_transcript(prompt)) == "retail"


def test_policy_tag_mentioned_in_prose_is_ignored():
    system = (
        "⟦SYSTEM⟧\n<instructions>\nYou are an agent. Follow the <policy> provided below.\n"
        "</instructions>\n<policy>\n# Real Policy\n\nDo not refund insurance.\n</policy>\n"
        "[AVAILABLE TOOLS]\n- a_tool: x\n"
    )
    policy = extract_policy(parse_transcript(system).system_text)
    assert policy.startswith("# Real Policy")
    assert "Follow the" not in policy


def test_plain_role_lines_are_parsed():
    prompt = (
        "System: You are a retail agent.\n"
        "<policy>\nConfirm before exchanging items.\n</policy>\n\n"
        "User: I want to exchange the lamp in order #W1234567.\n\n"
        "Assistant: Sure, which colour?\n"
    )
    transcript = parse_transcript(prompt)
    assert transcript.system_text.startswith("You are a retail agent.")
    assert any(e.kind == "user" and "exchange the lamp" in e.content for e in transcript.events)
    assert any(e.kind == "assistant" for e in transcript.events)


def test_plain_tool_lines_are_parsed():
    prompt = (
        "System: agent\nUser: check order #W1234567\nAssistant: one moment\n"
        'TOOL_CALL get_order_details: {"order_id": "#W1234567"}\n'
        'TOOL_RESPONSE get_order_details: {"order_id": "#W1234567", "status": "pending"}\n'
        "Assistant: it is pending\n"
    )
    transcript = parse_transcript(prompt)
    assert len(transcript.tool_calls) == 1
    assert transcript.tool_calls[0].tool_call.name == "get_order_details"
    assert transcript.tool_calls[0].tool_call.args == {"order_id": "#W1234567"}
    assert len(transcript.tool_responses) == 1
