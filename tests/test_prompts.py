from guardian.evidence import EvidenceConfig, build_evidence
from guardian.prompts import SYSTEM_PROMPT, build_messages, fit_user_content
from guardian.toolspec import extract_available_tools
from guardian.transcript import parse_response, parse_transcript

PROMPT = (
    "⟦SYSTEM⟧\nYou are an airline agent.\n<policy>\n"
    + ("Policy line about confirmation. " * 400)
    + "\n</policy>\n[AVAILABLE TOOLS]\n- get_reservation_details: read\n"
    "⟦USER⟧\nCheck EHGLP3 please\n"
    "⟦ASSISTANT · ход 1⟧\n"
    '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
    '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", "cabin": "economy"}\n'
)
RESPONSE = "⟦ASSISTANT · ход 2⟧\nReservation EHGLP3 is in economy class."


def test_build_messages_structure():
    transcript = parse_transcript(PROMPT)
    response = parse_response(RESPONSE)
    specs = extract_available_tools(transcript.system_text)
    evidence = build_evidence(transcript, response, specs, EvidenceConfig(max_chars=8000))
    messages = build_messages(evidence.text, response, [])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "[CONTEXT]" in user
    assert "[RESPONSE UNDER AUDIT]" in user
    assert "EHGLP3" in user
    assert "[AUTOMATED PRE-CHECKS]" in user


def test_fit_user_content_preserves_response():
    content = "[CONTEXT]\n" + ("X" * 50_000) + "\n[RESPONSE UNDER AUDIT]\n<response_under_audit>\nIMPORTANT\n</response_under_audit>\n"
    fitted = fit_user_content(content, 5_000)
    assert len(fitted) <= 5_100
    assert "IMPORTANT" in fitted
    assert "[RESPONSE UNDER AUDIT]" in fitted


def test_fit_user_content_shrinks_context_first():
    content = "[CONTEXT]\n" + ("A" * 10_000) + "\n[RESPONSE UNDER AUDIT]\n<response_under_audit>\nB\n</response_under_audit>"
    fitted = fit_user_content(content, 2_000)
    assert fitted.startswith("[CONTEXT]")
    assert fitted.endswith("</response_under_audit>")
