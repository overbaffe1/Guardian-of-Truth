from guardian.signals import SignalsConfig, compute_signals
from guardian.toolspec import extract_available_tools
from guardian.transcript import parse_response, parse_transcript

SYSTEM = """⟦SYSTEM⟧
You are an airline agent.
<policy>
# Airline Agent Policy
Before taking any actions that update the booking database (booking, modifying flights,
editing baggage) you must list the action details and obtain explicit user confirmation (yes).
Travel insurance is not refundable.
</policy>

[AVAILABLE TOOLS]
- get_reservation_details: read reservation
- update_reservation_flights: change flights
- cancel_reservation: cancel a reservation
"""


def build(prompt_rest: str, response: str, domain_system: str = SYSTEM):
    prompt = domain_system + prompt_rest
    transcript = parse_transcript(prompt)
    specs = extract_available_tools(transcript.system_text)
    return transcript, parse_response(response), specs


def kinds(signals, severity=None):
    return {s.kind for s in signals if severity is None or s.severity == severity}


def test_unavailable_tool_is_hard():
    t, r, specs = build(
        "⟦USER⟧\nplease update\n⟦ASSISTANT · ход 1⟧\nok\n",
        '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL issue_full_refund: {"amount": 100}',
    )
    signals = compute_signals(t, r, specs)
    assert "tool_not_available" in kinds(signals, "hard")


def test_tool_suffix_family_is_accepted():
    t, r, specs = build(
        "⟦USER⟧\nhi\n⟦ASSISTANT · ход 1⟧\nok\n",
        '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL get_reservation_details_1234: {"reservation_id": "EHGLP3"}',
    )
    signals = compute_signals(t, r, specs)
    assert "tool_not_available" not in kinds(signals)


def test_fabricated_reservation_id_is_hard():
    t, r, specs = build(
        "⟦USER⟧\nmy reservation is EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", "cabin": "economy"}\n',
        '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL update_reservation_flights: '
        '{"reservation_id": "ZZ9ZZZ", "cabin": "business", "flights": []}',
    )
    signals = compute_signals(t, r, specs)
    assert "ungrounded_argument" in kinds(signals, "hard")


def test_grounded_response_has_no_hard_signals():
    t, r, specs = build(
        "⟦USER⟧\nmy reservation is EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", '
        '"cabin": "economy", "insurance_price": 134.0}\n',
        "⟦ASSISTANT · ход 2⟧\nYour reservation EHGLP3 is in economy class.",
    )
    signals = compute_signals(t, r, specs)
    assert not [s for s in signals if s.severity == "hard"]


def test_invented_amount_is_flagged_as_hint():
    t, r, specs = build(
        "⟦USER⟧\nreservation EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", '
        '"insurance_price": 134.0}\n',
        "⟦ASSISTANT · ход 2⟧\nI refunded 250.00 USD for your insurance.",
    )
    signals = compute_signals(t, r, specs)
    assert "ungrounded_number" in kinds(signals, "soft")


def test_missing_confirmation_is_flagged():
    t, r, specs = build(
        "⟦USER⟧\nplease change my flight to business\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3"}\n',
        '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL update_reservation_flights: '
        '{"reservation_id": "EHGLP3", "cabin": "business", "flights": []}',
    )
    signals = compute_signals(t, r, specs)
    assert "missing_confirmation" in kinds(signals, "soft")


def test_confirmation_present_suppresses_signal():
    t, r, specs = build(
        "⟦USER⟧\nplease change my flight to business\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        "⟦ASSISTANT · ход 2⟧\nI will change it to business for 120 USD. Shall I proceed?\n"
        "⟦USER⟧\nYes, please go ahead.\n",
        '⟦ASSISTANT · ход 3⟧\n\t→ TOOL_CALL update_reservation_flights: '
        '{"reservation_id": "EHGLP3", "cabin": "business", "flights": []}',
    )
    signals = compute_signals(t, r, specs)
    assert "missing_confirmation" not in kinds(signals)


def test_false_refusal_hint():
    t, r, specs = build(
        "⟦USER⟧\ncancel my reservation EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3"}\n',
        "⟦ASSISTANT · ход 2⟧\nI'm unable to cancel reservations, I don't have access to that.",
    )
    signals = compute_signals(t, r, specs)
    assert "possible_false_refusal" in kinds(signals, "soft")


def test_claimed_action_without_tool_call():
    t, r, specs = build(
        "⟦USER⟧\ncancel it\n⟦ASSISTANT · ход 1⟧\nok\n",
        "⟦ASSISTANT · ход 2⟧\nI have cancelled your reservation. You will get a full refund.",
    )
    signals = compute_signals(t, r, specs)
    assert "claimed_action_without_tool_call" in kinds(signals, "soft")


def test_action_targets_unrequested_entity():
    t, r, specs = build(
        "⟦USER⟧\nplease change reservation EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        "⟦ASSISTANT · ход 2⟧\nShall I proceed?\n⟦USER⟧\nyes\n",
        '⟦ASSISTANT · ход 3⟧\n\t→ TOOL_CALL update_reservation_flights: '
        '{"reservation_id": "QWR123", "cabin": "economy", "flights": []}',
    )
    signals = compute_signals(t, r, specs)
    assert "action_targets_unrequested_entity" in kinds(signals, "soft")


def test_rules_can_be_disabled():
    t, r, specs = build(
        "⟦USER⟧\nx\n⟦ASSISTANT · ход 1⟧\nok\n",
        '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL nonexistent_tool: {}',
    )
    cfg = SignalsConfig(enable_tool_availability=False)
    signals = compute_signals(t, r, specs, cfg)
    assert "tool_not_available" not in kinds(signals)
