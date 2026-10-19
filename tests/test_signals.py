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


def test_fabricated_identifier_in_text_is_hard():
    t, r, specs = build(
        "⟦USER⟧\nhi\n⟦ASSISTANT · ход 1⟧\nok\n",
        "⟦ASSISTANT · ход 2⟧\nYour confirmation code is QX7ZP4, we emailed fake@example.com",
    )
    signals = compute_signals(t, r, specs)
    assert "ungrounded_code" in kinds(signals, "hard")
    assert "ungrounded_email" in kinds(signals, "hard")


def test_identifiers_present_in_context_are_green():
    t, r, specs = build(
        "⟦USER⟧\nmy email is emma@example.com and reservation EHGLP3\n"
        "⟦ASSISTANT · ход 1⟧\nok\n",
        "⟦ASSISTANT · ход 2⟧\nWe will contact emma@example.com about EHGLP3.",
    )
    signals = compute_signals(t, r, specs)
    assert not [s for s in signals if s.kind.startswith("ungrounded_")]


def test_success_claim_after_tool_error():
    t, r, specs = build(
        "⟦USER⟧\ncancel EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE cancel_reservation: {"error": true, "message": "Reservation not found"}\n',
        "⟦ASSISTANT · ход 2⟧\nI have successfully cancelled your reservation.",
    )
    signals = compute_signals(t, r, specs)
    assert "claimed_success_after_tool_error" in kinds(signals, "soft")


def test_success_claim_with_ok_tool_is_fine():
    t, r, specs = build(
        "⟦USER⟧\ncancel EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE cancel_reservation: {"reservation_id": "EHGLP3", "status": "cancelled"}\n',
        "⟦ASSISTANT · ход 2⟧\nI have successfully cancelled your reservation.",
    )
    signals = compute_signals(t, r, specs)
    assert "claimed_success_after_tool_error" not in kinds(signals)


def test_success_claim_after_recovered_error_is_fine():
    t, r, specs = build(
        "⟦USER⟧\ncancel EHGLP3\n⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE cancel_reservation: {"error": true, "message": "Not found"}\n'
        '\t→ TOOL_CALL cancel_reservation: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE cancel_reservation: {"status": "cancelled"}\n',
        "⟦ASSISTANT · ход 2⟧\nYour reservation has been successfully cancelled.",
    )
    signals = compute_signals(t, r, specs)
    assert "claimed_success_after_tool_error" not in kinds(signals)


def test_unavailable_tool_mentioned_in_context_is_soft():
    t, r, specs = build(
        "⟦USER⟧\ncan you use issue_full_refund for me?\n⟦ASSISTANT · ход 1⟧\nlet me check\n",
        '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL issue_full_refund: {"amount": 100}',
    )
    signals = compute_signals(t, r, specs)
    assert "tool_not_available" in kinds(signals, "soft")
    assert "tool_not_available" not in kinds(signals, "hard")


def test_text_schema_violation_is_only_soft():
    system = (
        "⟦SYSTEM⟧\nYou are a banking agent.\n[AVAILABLE TOOLS]\n"
        "Tool: open_bank_account_4821\nDescription: Open an account.\nParameters:\n"
        "  - customer_id: string (required) - customer id\n"
        "  - account_class: string (required) - class name\n"
    )
    t, r, specs = build(
        "⟦USER⟧\ncustomer C1001 wants a new account, class standard\n"
        "⟦ASSISTANT · ход 1⟧\nchecking\n",
        '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL open_bank_account_4821: {"customer_id": "C1001"}',
        domain_system=system,
    )
    signals = compute_signals(t, r, specs)
    assert "invalid_tool_arguments" in kinds(signals, "soft")
    assert "invalid_tool_arguments" not in kinds(signals, "hard")


# ---------------------------------------------------------------------------
# Multilingual lexical rules (the public split is ~60% Russian)
# ---------------------------------------------------------------------------

TELECOM_SYSTEM = """⟦SYSTEM⟧
Ты — оператор связи.
<policy>
# Telecom Policy
Перед сбросом настроек нужно получить подтверждение пользователя (да).
</policy>

[AVAILABLE TOOLS]
- reset_apn_settings: сбросить настройки APN
- run_speed_test: запустить замер скорости
"""


def test_russian_claimed_action_without_tool_call_is_detected():
    t, r, specs = build(
        "⟦USER⟧\nу меня не работает MMS\n⟦ASSISTANT · ход 1⟧\nпроверяю\n",
        "⟦ASSISTANT · ход 2⟧\nЯ еще раз отправлю команду на сброс настроек APN.\n"
        "Сброс настроек APN отправлен повторно.",
        domain_system=TELECOM_SYSTEM,
    )
    signals = compute_signals(t, r, specs)
    assert "claimed_action_without_tool_call" in kinds(signals, "soft")


def test_russian_false_refusal_is_detected():
    t, r, specs = build(
        "⟦USER⟧\nпоменяйте товар в заказе\n⟦ASSISTANT · ход 1⟧\nсмотрю заказ\n",
        "⟦ASSISTANT · ход 2⟧\nЯ не смогу исправить это самостоятельно с помощью "
        "имеющихся инструментов.",
        domain_system=TELECOM_SYSTEM,
    )
    signals = compute_signals(t, r, specs)
    assert "possible_false_refusal" in kinds(signals, "soft")


def test_russian_explicit_confirmation_suppresses_missing_confirmation():
    """A bare "нет" inside a sentence must not be read as a refusal to confirm.

    Regression: anchoring the Russian negative pattern to the start of a line
    keeps "у меня нет данных" from suppressing the missing-confirmation hint.
    """
    t, r, specs = build(
        "⟦USER⟧\nотмените бронирование IFOYYZ\n⟦ASSISTANT · ход 1⟧\nок\n",
        "⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL reset_apn_settings: {}",
    )
    assert "missing_confirmation" in kinds(compute_signals(t, r, specs), "soft")

    t2, r2, specs2 = build(
        "⟦USER⟧\nДа, подтверждаю! У меня нет других вопросов.\n"
        "⟦ASSISTANT · ход 1⟧\nок\n",
        "⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL reset_apn_settings: {}",
    )
    assert "missing_confirmation" not in kinds(compute_signals(t2, r2, specs2), "soft")


def test_placeholder_argument_is_hard():
    t, r, specs = build(
        "⟦USER⟧\nзакройте мой счёт\n⟦ASSISTANT · ход 1⟧\nищу счёт\n",
        "⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL reset_apn_settings: "
        '{"account_id": "light_blue_account_id_placeholder"}',
    )
    signals = compute_signals(t, r, specs)
    assert "placeholder_argument" in kinds(signals, "hard")


def test_real_identifier_is_not_flagged_as_placeholder():
    t, r, specs = build(
        "⟦USER⟧\nотмените бронирование EHGLP3\n⟦ASSISTANT · ход 1⟧\nок\n",
        "⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL reset_apn_settings: "
        '{"reservation_id": "EHGLP3", "reason": "customer request"}',
    )
    assert "placeholder_argument" not in kinds(compute_signals(t, r, specs))


def test_nested_discoverable_tool_arguments_are_inspected():
    """The wrapper hides the real call, so the payload must be decoded."""
    system = (
        "⟦SYSTEM⟧\nYou are a banking agent.\n[AVAILABLE TOOLS]\n"
        "- log_verification: log a verification event\n"
        "- call_discoverable_agent_tool: invoke a discoverable tool\n"
    )
    t, r, specs = build(
        "⟦USER⟧\nзакройте счёт\n⟦ASSISTANT · ход 1⟧\nищу счёт\n",
        "⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL call_discoverable_agent_tool: "
        '{"agent_tool_name": "close_bank_account_7392", "arguments": '
        '"{\\"account_id\\": \\"light_blue_account_id_placeholder\\"}"}',
        domain_system=system,
    )
    signals = compute_signals(t, r, specs)
    assert "placeholder_argument" in kinds(signals, "hard")


def test_nested_tool_name_availability_is_checked():
    system = (
        "⟦SYSTEM⟧\nYou are a banking agent.\n[AVAILABLE TOOLS]\n"
        "- call_discoverable_agent_tool: invoke a discoverable tool\n"
    )
    t, r, specs = build(
        "⟦USER⟧\nзакройте счёт\n⟦ASSISTANT · ход 1⟧\nок\n",
        "⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL call_discoverable_agent_tool: "
        '{"agent_tool_name": "delete_everything", "arguments": "{}"}',
        domain_system=system,
    )
    assert "tool_not_available" in kinds(compute_signals(t, r, specs))
