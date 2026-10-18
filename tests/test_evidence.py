import json
from pathlib import Path

from guardian.evidence import EvidenceConfig, build_evidence, compress_tool_response
from guardian.toolspec import extract_available_tools
from guardian.transcript import parse_response, parse_transcript

FIXTURE = Path(__file__).parent / "fixtures" / "tau2_samples.jsonl"


def test_evidence_respects_budget():
    row = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    transcript = parse_transcript(row["prompt"])
    response = parse_response(row["response"])
    specs = extract_available_tools(transcript.system_text)
    cfg = EvidenceConfig(max_chars=8000)
    evidence = build_evidence(transcript, response, specs, cfg)
    assert len(evidence.text) <= 8000 + 2000  # small overhead from headers
    assert evidence.selected_tool_responses >= 1
    assert "⟦SYSTEM⟧" in evidence.text
    assert "TOOL_CALL" in evidence.text


def test_relevant_tool_response_is_selected():
    prompt = (
        "⟦SYSTEM⟧\npolicy\n[AVAILABLE TOOLS]\n- get_reservation_details: x\n"
        "⟦USER⟧\ncheck EHGLP3\n"
        "⟦ASSISTANT · ход 1⟧\n"
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", "cabin": "business"}\n'
        '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "OTHER9"}\n'
        '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "OTHER9", "cabin": "economy"}\n'
    )
    transcript = parse_transcript(prompt)
    response = parse_response("⟦ASSISTANT · ход 2⟧\nReservation EHGLP3 is in business class.")
    specs = extract_available_tools(transcript.system_text)
    evidence = build_evidence(
        transcript, response, specs, EvidenceConfig(max_chars=6000, max_tool_response_chars=600)
    )
    assert "business" in evidence.text


def test_json_compression_keeps_relevant_items():
    payload = {
        "flights": [
            {"flight_number": f"HAT{i:03d}", "price": 100 + i} for i in range(50)
        ],
        "reservation_id": "EHGLP3",
    }
    text = compress_tool_response(
        json.dumps(payload), "get_reservation_details", "EHGLP3", 1200, 6
    )
    assert "EHGLP3" in text
    assert len(text) <= 1200


def test_truncation_of_plain_text():
    long_text = "line " * 5000
    text = compress_tool_response(long_text, "shell", "nothing", 500)
    assert len(text) <= 520


def _banking_prompt(n_docs: int = 25) -> str:
    docs = []
    for i in range(n_docs):
        docs.append(
            f"doc_{i}: ## Section {i}\n"
            + ("General banking information about accounts and cards. " * 60)
            + ("\nThe wire transfer fee is 25 USD. " if i == 17 else "")
        )
    tool_chunks = []
    for i, doc in enumerate(docs):
        indented = "\n".join("\t   " + line for line in doc.splitlines())
        tool_chunks.append(
            f'\t→ TOOL_CALL shell: {{"command": "cat doc_{i}.md"}}\n'
            f"\t← TOOL_RESPONSE shell: {indented}"
        )
    return (
        "⟦SYSTEM⟧\nYou are a banking agent.\n<policy>\n"
        + ("Always verify the customer before account actions. " * 50)
        + "\n</policy>\n[AVAILABLE TOOLS]\nTool: shell\nDescription: run unix commands\n"
        "Parameters:\n  - command: string (required) - command\n"
        "⟦USER⟧\nWhat is the wire transfer fee?\n"
        "⟦ASSISTANT · ход 1⟧\n" + "\n".join(tool_chunks)
    )


def test_long_banking_context_is_compressed():
    transcript = parse_transcript(_banking_prompt())
    assert transcript.domain == "banking_knowledge"
    response = parse_response("⟦ASSISTANT · ход 2⟧\nThe wire transfer fee is 25 USD.")
    specs = extract_available_tools(transcript.system_text)
    evidence = build_evidence(
        transcript,
        response,
        specs,
        EvidenceConfig(max_chars=12000, max_tool_response_chars=900),
    )
    assert len(evidence.text) <= 14000
    assert "25 USD" in evidence.text  # the relevant document survived
    assert evidence.selected_tool_responses < evidence.total_tool_responses
