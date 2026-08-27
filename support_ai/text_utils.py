"""Small deterministic text helpers used by the assistant."""

from __future__ import annotations

import re
from collections.abc import Iterable


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./:-]*")
ERROR_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}(?:\s+after\s+\d+s)?\b")


def normalize_text(text: object) -> str:
    """Normalize common punctuation without changing meaning."""

    value = "" if text is None else str(text)
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u2192": "->",
        "\u00d7": "x",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def normalize_whitespace(text: object) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def tokenize(text: object) -> list[str]:
    tokens = TOKEN_RE.findall(normalize_text(text).lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def phrase_hits(text: object, phrases: Iterable[str]) -> list[str]:
    haystack = normalize_text(text).lower()
    hits: list[str] = []
    for phrase in phrases:
        needle = normalize_text(phrase).lower()
        if not needle:
            continue
        if " " in needle or "_" in needle or "-" in needle:
            if needle in haystack:
                hits.append(phrase)
        elif re.search(rf"\b{re.escape(needle)}\b", haystack):
            hits.append(phrase)
    return hits


def extract_error_codes(text: object) -> list[str]:
    seen: set[str] = set()
    codes: list[str] = []
    for match in ERROR_CODE_RE.findall(normalize_text(text)):
        code = normalize_whitespace(match).upper()
        if code not in seen and code not in {"JSON", "CSV", "PDF", "API", "REST", "SAML", "SSO"}:
            seen.add(code)
            codes.append(code)
    return codes


def split_sentences(text: object) -> list[str]:
    value = normalize_text(text)
    pieces = re.split(r"(?<=[.!?])\s+|\n+", value)
    return [normalize_whitespace(piece) for piece in pieces if normalize_whitespace(piece)]


def best_sentence(text: object, query: object, fallback_chars: int = 320) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return clip(text, fallback_chars)

    query_tokens = set(tokenize(query))
    query_codes = {code.lower() for code in extract_error_codes(query)}

    def score(sentence: str) -> tuple[int, int]:
        sentence_tokens = set(tokenize(sentence))
        sentence_lower = sentence.lower()
        code_score = sum(1 for code in query_codes if code.lower() in sentence_lower)
        return (code_score * 5 + len(query_tokens & sentence_tokens), -len(sentence))

    return clip(max(sentences, key=score), fallback_chars)


def clip(text: object, max_chars: int = 320) -> str:
    value = normalize_whitespace(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def first_non_empty(*values: object) -> str:
    for value in values:
        normalized = normalize_whitespace(value)
        if normalized:
            return normalized
    return ""
