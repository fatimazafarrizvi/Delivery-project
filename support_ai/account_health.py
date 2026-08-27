"""TAM account health summariser."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from .data_loader import account_lookup, load_tickets
from .llm import complete_json, use_llm
from .pii import redact_account, redact_text
from .prompts import BRIEF_COMPOSE_PROMPT, RISK_EXTRACT_PROMPT, render_prompt
from .text_utils import best_sentence, clip, normalize_whitespace, phrase_hits


ACCOUNT_PROMPT_VERSION = f"{RISK_EXTRACT_PROMPT.identifier}+{BRIEF_COMPOSE_PROMPT.identifier}"
OPEN_STATUSES = {"Open", "In Progress", "Pending Customer"}


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalize_account(account: dict[str, Any] | None, account_id: str = "") -> dict[str, Any]:
    raw = dict(account or {})
    company = raw.get("company") or account_id or "Unknown company"
    return {
        "account_id": raw.get("account_id") or account_id,
        "company": company,
        "tam": raw.get("tam") or "Unassigned",
        "plan_tier": raw.get("plan_tier") or "Unknown",
        "arr_usd": _optional_int(raw.get("arr_usd")),
        "seats_licensed": max(_int(raw.get("seats_licensed"), 0), 0),
        "seats_active": max(_int(raw.get("seats_active"), 0), 0),
        "products": list(raw.get("products") or []),
        "health_status": raw.get("health_status") or "Unknown",
        "usage_trend": raw.get("usage_trend") or "Unknown",
        "open_tickets": _int(raw.get("open_tickets"), 0),
        "p1_tickets_last_30d": _int(raw.get("p1_tickets_last_30d"), 0),
        "renewal_date": raw.get("renewal_date"),
        "last_qbr_date": raw.get("last_qbr_date"),
        "escalation_notes": list(raw.get("escalation_notes") or []),
        "nps_score": raw.get("nps_score"),
        "industry": raw.get("industry") or "Unknown",
        "region": raw.get("region") or "Unknown",
        "incomplete_fields": [
            field
            for field in ("plan_tier", "arr_usd", "health_status", "renewal_date", "products")
            if not raw.get(field)
        ],
    }


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def dataset_as_of(tickets: list[dict[str, Any]] | None = None) -> datetime:
    source = tickets or load_tickets()
    if not source:
        return datetime.now(timezone.utc)
    return max(parse_datetime(ticket["created_at"]) for ticket in source)


def coerce_as_of(as_of: str | date | datetime | None, tickets: list[dict[str, Any]]) -> datetime:
    if as_of is None:
        return dataset_as_of(tickets)
    if isinstance(as_of, datetime):
        return as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    if isinstance(as_of, date):
        return datetime.combine(as_of, time.max, tzinfo=timezone.utc)
    return datetime.combine(parse_date(as_of), time.max, tzinfo=timezone.utc)


def recent_tickets_for_account(
    account_id: str,
    tickets: list[dict[str, Any]],
    days: int = 90,
    as_of: str | date | datetime | None = None,
) -> tuple[list[dict[str, Any]], datetime, datetime]:
    as_of_dt = coerce_as_of(as_of, tickets)
    window_start = as_of_dt - timedelta(days=days)
    recent = [
        ticket
        for ticket in tickets
        if ticket.get("account_id") == account_id
        and window_start <= parse_datetime(ticket["created_at"]) <= as_of_dt
    ]
    recent.sort(key=lambda ticket: ticket["created_at"], reverse=True)
    return recent, window_start, as_of_dt


def _seat_utilization(account: dict[str, Any]) -> float | None:
    licensed = _int(account.get("seats_licensed"), 0)
    if licensed <= 0:
        return None
    return round((_int(account.get("seats_active"), 0) / licensed) * 100, 1)


def _risk(
    severity: str,
    source: str,
    reason: str,
    quote: str = "",
    ticket_id: str | None = None,
) -> dict[str, str]:
    item = {
        "severity": severity,
        "source": source,
        "reason": reason,
        "quote": quote,
    }
    if ticket_id:
        item["ticket_id"] = ticket_id
    return item


def account_level_risks(account: dict[str, Any], as_of: datetime) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    health = account.get("health_status", "Unknown")
    usage = account.get("usage_trend", "Unknown")
    open_tickets = _int(account.get("open_tickets"), 0)
    p1_count = _int(account.get("p1_tickets_last_30d"), 0)
    nps = account.get("nps_score")
    utilization = _seat_utilization(account)
    incomplete = account.get("incomplete_fields") or []

    if incomplete:
        risks.append(
            _risk(
                "medium",
                "account",
                "CRM export is incomplete; brief uses safe defaults for missing fields.",
                f"missing={', '.join(incomplete)}",
            )
        )

    if health == "Churning":
        risks.append(_risk("high", "account", "Account health is marked Churning.", f"health_status={health}"))
    elif health == "At Risk":
        risks.append(_risk("medium", "account", "Account health is marked At Risk.", f"health_status={health}"))

    if usage == "Inactive":
        risks.append(_risk("high", "account", "Usage trend is inactive.", f"usage_trend={usage}"))
    elif usage == "Declining":
        risks.append(_risk("medium", "account", "Usage trend is declining.", f"usage_trend={usage}"))

    if utilization is not None and utilization < 60:
        risks.append(_risk("high", "account", f"Seat utilization is low at {utilization}%.", f"{account.get('seats_active')} of {account.get('seats_licensed')} seats active"))
    elif utilization is not None and utilization < 75:
        risks.append(_risk("medium", "account", f"Seat utilization is below target at {utilization}%.", f"{account.get('seats_active')} of {account.get('seats_licensed')} seats active"))

    if open_tickets >= 8:
        risks.append(_risk("high", "account", f"High open-ticket load: {open_tickets} open tickets.", f"open_tickets={open_tickets}"))
    elif open_tickets >= 5:
        risks.append(_risk("medium", "account", f"Elevated open-ticket load: {open_tickets} open tickets.", f"open_tickets={open_tickets}"))

    if p1_count > 0:
        risks.append(_risk("high", "account", f"{p1_count} P1 tickets recorded in the last 30 days.", f"p1_tickets_last_30d={p1_count}"))

    if isinstance(nps, int) and nps <= 6:
        severity = "high" if nps <= 4 else "medium"
        risks.append(_risk(severity, "account", f"Low NPS score: {nps}.", f"nps_score={nps}"))

    renewal_value = account.get("renewal_date")
    if renewal_value:
        try:
            renewal_date = parse_date(str(renewal_value))
            days_to_renewal = (renewal_date - as_of.date()).days
            if days_to_renewal < 0:
                risks.append(_risk("high", "account", "Renewal date has already passed.", f"renewal_date={renewal_date.isoformat()}"))
            elif days_to_renewal <= 90:
                risks.append(_risk("medium", "account", f"Renewal is due in {days_to_renewal} days.", f"renewal_date={renewal_date.isoformat()}"))
        except ValueError:
            risks.append(_risk("low", "account", "Renewal date could not be parsed.", f"renewal_date={renewal_value}"))
    else:
        risks.append(_risk("medium", "account", "Renewal date is missing from the account record.", "renewal_date=null"))

    for note in account.get("escalation_notes") or []:
        normalized = normalize_whitespace(note)
        severity = "high" if phrase_hits(normalized, ["competing vendor", "frustration", "skipped", "p1", "champion left"]) else "medium"
        risks.append(_risk(severity, "account_note", "Escalation note indicates customer risk.", normalized))

    return risks


def p1_reconciliation_risks(
    account: dict[str, Any],
    tickets: list[dict[str, Any]],
    as_of: datetime,
) -> list[dict[str, str]]:
    """Cross-check CRM P1 claims against linked ticket records."""

    crm_count = _int(account.get("p1_tickets_last_30d"), 0)
    cutoff = as_of - timedelta(days=30)
    linked_count = sum(
        1
        for ticket in tickets
        if ticket.get("urgency") == "P1"
        and cutoff <= parse_datetime(ticket["created_at"]) <= as_of
    )
    risks: list[dict[str, str]] = []
    if crm_count != linked_count:
        risks.append(
            _risk(
                "medium",
                "data_quality",
                "CRM and linked-ticket P1 counts disagree for the last 30 days.",
                f"CRM={crm_count}; linked tickets={linked_count}",
            )
        )

    for note in account.get("escalation_notes") or []:
        match = re.search(r"\b(\d+)\s+(?:consecutive\s+)?P1(?:\s+tickets?)?", str(note), re.IGNORECASE)
        if match and int(match.group(1)) != crm_count:
            risks.append(
                _risk(
                    "high",
                    "data_quality",
                    "Escalation-note P1 claim disagrees with the CRM summary.",
                    f"note claims {match.group(1)}; CRM={crm_count}",
                )
            )
    return risks


def ticket_level_risks(tickets: list[dict[str, Any]]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    risk_phrases = [
        "cancel",
        "churn",
        "competitor",
        "competing vendor",
        "frustrat",
        "escalat",
        "urgent",
        "critical",
        "no workaround",
        "blocked",
        "production",
        "data loss",
        "unacceptable",
        "failing since",
    ]
    for ticket in tickets:
        text = f"{ticket.get('subject', '')}\n{ticket.get('body', '')}"
        text_hits = phrase_hits(text, risk_phrases)
        status = ticket.get("status")
        urgency = ticket.get("urgency")
        category = ticket.get("category")
        should_flag = (
            urgency in {"P1", "P2"}
            or category == "Data Loss"
            or status in OPEN_STATUSES and bool(text_hits)
            or isinstance(ticket.get("satisfaction_score"), int) and ticket["satisfaction_score"] <= 2
        )
        if not should_flag:
            continue

        severity = "high" if urgency == "P1" or category == "Data Loss" else "medium"
        if status in OPEN_STATUSES and urgency in {"P1", "P2"}:
            severity = "high"
        quote = (
            best_sentence(ticket.get("body", ""), " ".join(text_hits))
            if text_hits
            else clip(ticket.get("subject", ""), 180)
        )
        reason_parts = [f"{urgency} {category} ticket"]
        if status in OPEN_STATUSES:
            reason_parts.append(f"status is {status}")
        if text_hits:
            reason_parts.append(f"risk language: {', '.join(text_hits[:3])}")
        risks.append(
            _risk(
                severity=severity,
                source="ticket",
                reason="; ".join(reason_parts),
                quote=quote,
                ticket_id=ticket.get("ticket_id"),
            )
        )
    return risks


def _dedupe_risks(risks: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    for risk in sorted(risks, key=lambda item: severity_rank.get(item["severity"], 3)):
        key = (risk["source"], risk["reason"], risk.get("quote", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(risk)
    return deduped


def recommended_talking_points(account: dict[str, Any], risks: list[dict[str, str]], tickets: list[dict[str, Any]]) -> list[str]:
    points: list[str] = []
    health = account.get("health_status")
    usage = account.get("usage_trend")
    if health in {"At Risk", "Churning"}:
        points.append("Open with account health and ask what would restore confidence before renewal.")
    if usage in {"Declining", "Inactive"}:
        points.append("Review adoption blockers and agree on one usage recovery milestone.")
    if _int(account.get("open_tickets"), 0) >= 5 or tickets:
        points.append("Walk through the highest-impact support items and confirm owners, next actions, and dates.")
    if _int(account.get("p1_tickets_last_30d"), 0) > 0:
        points.append("Discuss incident follow-up and whether a recurring technical review is needed.")
    if any("competing vendor" in risk.get("quote", "").lower() for risk in risks):
        points.append("Address competitive evaluation directly and document must-win product gaps.")
    if any(risk.get("source") == "ticket" for risk in risks):
        points.append("Quote recent ticket language back to the customer and validate whether the pain is resolved.")
    if health == "New":
        points.append("Confirm onboarding progress, admin readiness, and training completion.")
    if account.get("incomplete_fields"):
        points.append("Confirm missing CRM fields with the customer team before treating this brief as complete.")
    if len(points) < 2:
        points.append("Confirm business goals, upcoming roadmap needs, and the next QBR success metric.")
    if len(points) < 2:
        points.append("Agree on owners and dates for follow-up actions before closing the QBR.")
    return list(dict.fromkeys(points))[:6]


def build_markdown_brief(summary: dict[str, Any]) -> str:
    risk_lines = []
    for risk in summary["risks"]:
        ticket_prefix = f" ({risk['ticket_id']})" if risk.get("ticket_id") else ""
        quote = f" Evidence: \"{risk['quote']}\"" if risk.get("quote") else ""
        reason = risk["reason"].rstrip(".")
        risk_lines.append(f"- [{risk['severity'].upper()}] {reason}{ticket_prefix}.{quote}")
    if not risk_lines:
        risk_lines.append("- No material risk signals found in the selected window.")

    talking_points = [f"- {point}" for point in summary["talking_points"]]
    return "\n".join(
        [
            f"# Account Brief: {summary['company']}",
            "",
            f"Prompt version: `{summary['prompt_version']}`",
            f"As of: `{summary['as_of']}` | Ticket window: `{summary['window_start']}` to `{summary['as_of']}`",
            "",
            "## Executive Summary",
            summary["executive_summary"],
            "",
            "## Open Risks & Flagged Issues",
            "\n".join(risk_lines),
            "",
            "## Recommended Talking Points",
            "\n".join(talking_points),
            "",
        ]
    )


def _ticket_snippets(tickets: list[dict[str, Any]], limit: int = 8) -> str:
    blocks = []
    for ticket in tickets[:limit]:
        blocks.append(
            json.dumps(
                {
                    "ticket_id": ticket.get("ticket_id"),
                    "subject": redact_text(ticket.get("subject", "")),
                    "urgency": ticket.get("urgency"),
                    "status": ticket.get("status"),
                    "category": ticket.get("category"),
                    "quote": redact_text(clip(ticket.get("body", ""), 280)),
                }
            )
        )
    return "\n".join(blocks) if blocks else "(no linked tickets in window)"


def _chain_llm_brief(summary: dict[str, Any], account: dict[str, Any], tickets: list[dict[str, Any]]) -> dict[str, Any]:
    safe_account = redact_account(account)
    extract_system, extract_user = render_prompt(
        RISK_EXTRACT_PROMPT,
        account_json=json.dumps({key: safe_account[key] for key in ("account_id", "company", "health_status", "usage_trend", "open_tickets", "p1_tickets_last_30d", "nps_score", "escalation_notes", "renewal_date") if key in safe_account}, indent=2),
        ticket_snippets=_ticket_snippets(tickets),
    )
    try:
        extracted = complete_json(extract_system, extract_user, temperature=0.0, seed=7)
        compose_system, compose_user = render_prompt(
            BRIEF_COMPOSE_PROMPT,
            metrics_json=json.dumps(summary["metrics"], indent=2),
            risks_json=json.dumps(extracted.get("risks", summary["risks"]), indent=2),
        )
        composed = complete_json(compose_system, compose_user, temperature=0.0, seed=7)
    except Exception as exc:  # noqa: BLE001
        summary["llm_error"] = "Hosted model unavailable; deterministic fallback used."
        summary["llm_used"] = False
        return summary

    exec_summary = normalize_whitespace(composed.get("executive_summary", ""))
    if exec_summary:
        summary["executive_summary"] = exec_summary
    talking = composed.get("talking_points")
    if isinstance(talking, list) and talking:
        summary["talking_points"] = [str(item) for item in talking[:6]]
    markdown = composed.get("markdown")
    required = ["Executive Summary", "Open Risks", "Recommended Talking Points"]
    if isinstance(markdown, str) and all(section in markdown for section in required):
        summary["markdown"] = markdown
    else:
        summary["markdown"] = build_markdown_brief(summary)
    extra_risks = extracted.get("risks")
    if isinstance(extra_risks, list):
        merged = list(summary["risks"])
        for risk in extra_risks:
            if not isinstance(risk, dict):
                continue
            merged.append(
                _risk(
                    str(risk.get("severity") or "medium"),
                    str(risk.get("source") or "ticket"),
                    str(risk.get("reason") or "LLM-extracted risk"),
                    str(risk.get("quote") or ""),
                    risk.get("ticket_id"),
                )
            )
        summary["risks"] = _dedupe_risks(merged)
        summary["markdown"] = build_markdown_brief(summary)
    summary["llm_used"] = True
    return summary


def summarize_account(
    account_id: str,
    *,
    days: int = 90,
    as_of: str | date | datetime | None = None,
    accounts: dict[str, dict[str, Any]] | None = None,
    tickets: list[dict[str, Any]] | None = None,
    account: dict[str, Any] | None = None,
    enable_llm: bool | None = None,
) -> dict[str, Any]:
    lookup = accounts or account_lookup()
    ticket_source = tickets or load_tickets()
    raw_account = account if account is not None else lookup.get(account_id)
    if not raw_account:
        return {
            "prompt_version": ACCOUNT_PROMPT_VERSION,
            "account_id": account_id,
            "error": "Account not found in accounts.json.",
            "executive_summary": "",
            "risks": [],
            "talking_points": ["Verify whether the account ID is correct or whether the account is missing from the CRM export."],
            "markdown": f"# Account Brief: {account_id}\n\nAccount not found in accounts.json.\n",
            "llm_used": False,
        }

    account_record = normalize_account(raw_account, account_id)
    recent_tickets, window_start, as_of_dt = recent_tickets_for_account(
        account_record["account_id"],
        ticket_source,
        days=days,
        as_of=as_of,
    )
    all_risks = _dedupe_risks(
        account_level_risks(account_record, as_of_dt)
        + p1_reconciliation_risks(account_record, recent_tickets, as_of_dt)
        + ticket_level_risks(recent_tickets)
    )
    products = ", ".join(account_record.get("products") or ["unknown products"])
    utilization = _seat_utilization(account_record)
    arr_text = f"${account_record['arr_usd']:,} ARR" if account_record["arr_usd"] is not None else "unknown ARR"
    utilization_text = f"{utilization}% utilization" if utilization is not None else "utilization unavailable"
    high_or_medium = [risk for risk in all_risks if risk["severity"] in {"high", "medium"}]
    ticket_sentence = (
        f"{len(recent_tickets)} linked {'ticket was' if len(recent_tickets) == 1 else 'tickets were'} found in the {days}-day ticket window."
        if recent_tickets
        else f"No linked tickets were found in the {days}-day ticket window, so the brief relies on account-level signals."
    )
    top_risk_text = (
        "The most important risks are "
        + "; ".join(clip(risk["reason"], 80).rstrip(".") for risk in high_or_medium[:3])
        + "."
        if high_or_medium
        else "No major risk signal is currently visible."
    )
    executive_summary = " ".join(
        [
            f"{account_record['company']} is a {account_record['plan_tier']} {account_record['industry']} account in {account_record['region']} with {arr_text}.",
            f"Usage is {str(account_record['usage_trend']).lower()} with {account_record['seats_active']} of {account_record['seats_licensed']} seats active ({utilization_text}) across {products}.",
            f"The account is marked {account_record['health_status']} and has {account_record['open_tickets']} open tickets in the account summary.",
            ticket_sentence,
            top_risk_text,
        ]
    )

    summary: dict[str, Any] = {
        "prompt_version": ACCOUNT_PROMPT_VERSION,
        "account_id": account_record["account_id"],
        "company": account_record["company"],
        "as_of": as_of_dt.date().isoformat(),
        "window_start": window_start.date().isoformat(),
        "ticket_count": len(recent_tickets),
        "tickets": recent_tickets,
        "metrics": {
            "plan_tier": account_record["plan_tier"],
            "arr_usd": account_record["arr_usd"],
            "health_status": account_record["health_status"],
            "usage_trend": account_record["usage_trend"],
            "seat_utilization_pct": utilization,
            "open_tickets": account_record["open_tickets"],
            "p1_tickets_last_30d": account_record["p1_tickets_last_30d"],
            "renewal_date": account_record["renewal_date"],
            "nps_score": account_record.get("nps_score"),
            "incomplete_fields": account_record.get("incomplete_fields") or [],
        },
        "executive_summary": executive_summary,
        "risks": all_risks,
        "talking_points": recommended_talking_points(account_record, all_risks, recent_tickets),
        "llm_used": False,
    }
    summary["markdown"] = build_markdown_brief(summary)

    should_llm = use_llm() if enable_llm is None else enable_llm
    if should_llm:
        summary = _chain_llm_brief(summary, account_record, recent_tickets)
    return summary
