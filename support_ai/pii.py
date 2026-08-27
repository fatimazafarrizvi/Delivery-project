"""Redact likely PII before any external model call."""

from __future__ import annotations

import re


EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{8,}\d)\b")
TOKEN_RE = re.compile(r"\b(?:sk-|rk-|Bearer\s+)\S+", re.IGNORECASE)
NAME_LINE_RE = re.compile(r"^(?:from|regards|thanks),?\s+.+$", re.IGNORECASE | re.MULTILINE)
ACCOUNT_ID_RE = re.compile(r"\bACC-[A-Z0-9-]+\b", re.IGNORECASE)


def redact_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = TOKEN_RE.sub("[REDACTED_TOKEN]", value)
    value = PHONE_RE.sub("[REDACTED_PHONE]", value)
    value = ACCOUNT_ID_RE.sub("[REDACTED_ACCOUNT_ID]", value)
    value = NAME_LINE_RE.sub("[REDACTED_SIGN_OFF]", value)
    return value


def redact_ticket(ticket: dict) -> dict:
    safe = dict(ticket)
    for key in ("subject", "body"):
        if key in safe:
            safe[key] = redact_text(safe[key])
    if safe.get("company"):
        safe["company"] = "[REDACTED_COMPANY]"
    if safe.get("assigned_agent"):
        safe["assigned_agent"] = "[REDACTED_AGENT]"
    if safe.get("account_id"):
        safe["account_id"] = "[REDACTED_ACCOUNT_ID]"
    return safe


def redact_account(account: dict) -> dict:
    """Remove direct account identifiers before an external model call."""

    safe = dict(account)
    for key, placeholder in (
        ("account_id", "[REDACTED_ACCOUNT_ID]"),
        ("company", "[REDACTED_COMPANY]"),
        ("tam", "[REDACTED_TAM]"),
        ("primary_contact", "[REDACTED_CONTACT]"),
    ):
        if safe.get(key):
            safe[key] = placeholder
    if "escalation_notes" in safe:
        safe["escalation_notes"] = [redact_text(note) for note in safe.get("escalation_notes") or []]
    return safe
