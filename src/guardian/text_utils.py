"""Small text helpers: normalisation, lenient JSON, tokenisation, matching."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Optional

_WS_RE = re.compile(r"[ \t\u00a0]+")
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "from", "by", "as", "it", "its",
    "i", "you", "we", "they", "he", "she", "me", "my", "your", "our", "their",
    "can", "could", "would", "should", "will", "shall", "may", "might", "do",
    "does", "did", "have", "has", "had", "not", "no", "yes", "please", "would",
    "there", "here", "about", "into", "over", "after", "before", "also", "just",
    "so", "up", "out", "any", "all", "some", "more", "most", "other", "such",
}


def normalize_ws(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    # collapse 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def norm_token(text: str) -> str:
    """Normalise a value for presence checks (IDs, emails, numbers)."""
    text = strip_accents(str(text)).lower().strip()
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"^[\s\"'`(\[{]+|[\s\"'`)\]}.,;:]+$", "", text)
    return text


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]+", text.lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def find_numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text or "")


def canonical_number(raw: str) -> str:
    """Canonical string for a numeric literal.

    ``1,200`` -> ``1200``; ``134.0`` and ``134.00`` -> ``134``;
    ``134,50`` -> ``134.5``.  Falls back to the stripped input on failure.
    """
    value = (raw or "").strip()
    if not value:
        return ""
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
    if match:
        value = match.group(0)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", value):
        value = value.replace(",", "")
    else:
        value = value.replace(",", ".")
    try:
        number = float(value)
    except Exception:
        return value
    if number == int(number):
        return str(int(number))
    return ("%.6f" % number).rstrip("0").rstrip(".")


def numbers_normalized(text: str) -> set[str]:
    """Return a set of canonical number strings (``1,200`` -> ``1200``)."""
    out: set[str] = set()
    for raw in find_numbers(text):
        canonical = canonical_number(raw)
        if canonical:
            out.add(canonical)
    return out


def _balance_delta(chunk: str) -> tuple[int, int]:
    """Brace / bracket balance ignoring characters inside JSON strings."""
    braces = brackets = 0
    in_str = False
    escape = False
    for ch in chunk:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            braces += 1
        elif ch == "}":
            braces -= 1
        elif ch == "[":
            brackets += 1
        elif ch == "]":
            brackets -= 1
    return braces, brackets


def looks_like_json_start(text: str) -> bool:
    return text.lstrip()[:1] in {"{", "["}


def needs_more_json(text: str) -> bool:
    braces, brackets = _balance_delta(text)
    return braces > 0 or brackets > 0


def lenient_json_loads(text: str) -> tuple[Optional[Any], bool]:
    """Best-effort JSON parsing used for tool arguments / results.

    Returns ``(value, ok)``.  ``ok`` is False when the text is not JSON at all
    (e.g. a plain-text tool response), which is a normal situation and must not
    be treated as an error.
    """
    if text is None:
        return None, False
    raw = text.strip()
    if not raw:
        return None, False
    # Strip common renderer artefacts around JSON payloads.
    raw = raw.strip("`").strip()
    candidates = [raw]

    # Isolate the first JSON object/array.
    start = min(
        [i for i in (raw.find("{"), raw.find("[")) if i >= 0],
        default=-1,
    )
    if start > 0:
        candidates.append(raw[start:])

    for cand in candidates:
        try:
            return json.loads(cand), True
        except Exception:
            pass
        # fix trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", cand)
        # single quotes -> double quotes (only when no double quotes present)
        if '"' not in fixed:
            fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed), True
        except Exception:
            pass
        # extract balanced prefix
        try:
            value, _ = json.JSONDecoder().raw_decode(cand)
            return value, True
        except Exception:
            pass
    return None, False


def compact_json(value: Any, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(value)
    if len(text) > limit:
        text = text[: limit // 2] + " … " + text[-limit // 2 :]
    return text


def overlap_score(query: str, document: str) -> float:
    """Cheap lexical relevance (idf-free). Numbers and identifiers weigh more."""
    q_words = content_words(query)
    if not q_words and not find_numbers(query):
        return 0.0
    doc_low = document.lower()
    doc_words = set(words(document))
    score = 0.0
    for w in set(q_words):
        if w in doc_words:
            score += 1.0
    for n in numbers_normalized(query):
        if len(n) >= 2 and n in doc_low:
            score += 1.5
    return score / max(1.0, len(set(q_words))) ** 0.5


def truncate_middle(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head - 5
    return text[:head] + " … " + text[-tail:]


def estimate_tokens(text: str) -> int:
    """Conservative character-based token estimate (works for EN + JSON)."""
    if not text:
        return 0
    return int(len(text) / 3.0) + 1


def strip_markdown(text: str) -> str:
    text = re.sub(r"`{1,3}", "", text or "")
    text = re.sub(r"[*_]{1,3}", "", text)
    return text


def find_quoted(text: str) -> list[str]:
    return re.findall(r"[\"'`]([^\"'`\n]{3,120})[\"'`]", text or "")


def iter_sentences(text: str) -> Iterable[str]:
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        chunk = chunk.strip()
        if chunk:
            yield chunk
