"""Ticket triage assistant."""

from __future__ import annotations

import json
import re
from typing import Any

from .kb_retriever import KnowledgeRetriever, get_retriever
from .llm import complete_json, use_llm
from .pii import redact_ticket
from .prompts import TRIAGE_PROMPT, render_prompt
from .text_utils import clip, extract_error_codes, first_non_empty, normalize_whitespace, phrase_hits


TRIAGE_PROMPT_VERSION = TRIAGE_PROMPT.identifier
ALLOWED_PRODUCTS = {
    "DataBridge Pro",
    "CloudSync",
    "AnalyticsHub",
    "SecureVault",
    "WorkflowEngine",
    "Unknown",
}
ALLOWED_CATEGORIES = {
    "Bug",
    "Feature Request",
    "How-To",
    "Performance",
    "Billing",
    "Integration",
    "Onboarding",
    "Data Loss",
}
ALLOWED_URGENCY = {"P1", "P2", "P3", "P4"}

PRODUCT_ALIASES: dict[str, list[str]] = {
    "DataBridge Pro": [
        "databridge pro",
        "databridge",
        "data bridge",
        "pipeline",
        "schema registry",
        "schema management",
        "data ingestion",
        "batch_size",
        "records_processed",
    ],
    "CloudSync": [
        "cloudsync",
        "cloud sync",
        "file sync",
        "conflict",
        "sync job",
        "synced files",
        "bandwidth",
        "storage provider",
    ],
    "AnalyticsHub": [
        "analyticshub",
        "analytics hub",
        "dashboard",
        "report",
        "reports",
        "query profiler",
        "data source",
        "exports",
        "widgets",
    ],
    "SecureVault": [
        "securevault",
        "secure vault",
        "key rotation",
        "audit logs",
        "saml",
    ],
    "WorkflowEngine": [
        "workflowengine",
        "workflow engine",
        "workflow",
        "trigger",
        "actions",
        "cron",
        "schedule",
        "dead-letter",
        "idempotency",
    ],
}

PRODUCT_AREAS: dict[str, dict[str, list[str]]] = {
    "DataBridge Pro": {
        "Data Ingestion": ["data ingestion", "ingest", "source_type", "batch_size", "records", "file upload"],
        "Schema Management": ["schema management", "schema registry", "schema_mismatch", "schema enforcement"],
        "Pipeline Monitoring": ["pipeline monitoring", "heartbeat", "pipeline", "lag", "throughput", "monitoring"],
        "Connectors": ["connector", "connectors", "oauth", "salesforce", "snowflake", "bigquery", "jira", "hubspot"],
        "API": ["api", "rate limit", "webhook", "bearer token", "hmac"],
    },
    "CloudSync": {
        "File Sync": ["file sync", "files not updating", "sync job", "force sync", "storage provider"],
        "Conflict Resolution": ["conflict", "conflicts", "conflict storm", "resolve conflicts"],
        "Permissions": ["permissions", "read", "write", "admin", "group sync", "sso_group_not_found"],
        "Bandwidth Limits": ["bandwidth", "throttling", "upload_limit", "download_limit"],
        "Integrations": ["integration", "slack", "pagerduty", "jira", "salesforce", "webhook"],
    },
    "AnalyticsHub": {
        "Dashboard": ["dashboard", "widget", "query profiler", "visualisation", "load slowly"],
        "Reports": ["report", "scheduled report", "pdf", "csv", "excel"],
        "Data Sources": ["data source", "database", "snowflake", "bigquery", "query timeout", "connection test"],
        "Alerts": ["alert", "alerts", "threshold", "pagerduty", "cool-down"],
        "Exports": ["export", "exports", "truncated", "row limit", "parquet"],
    },
    "SecureVault": {
        "Authentication": ["authentication", "login", "token", "auth_token_expired", "mfa", "service account"],
        "Encryption": ["encryption", "aes", "tls", "customer-managed keys", "cmk"],
        "Audit Logs": ["audit log", "audit logs", "siem", "unauthorised", "unauthorized"],
        "Key Management": ["key management", "key rotation", "checksum_mismatch", "pending_rotation", "destroyed"],
        "SSO Configuration": [
            "sso",
            "saml",
            "acs url",
            "audience_mismatch",
            "group_not_mapped",
            "saml_assertion_expired",
            "cannot log in",
            "blocked by saml",
        ],
    },
    "WorkflowEngine": {
        "Triggers": ["trigger", "webhook", "app event", "manual trigger", "idempotency"],
        "Actions": ["action", "actions", "field mapping", "strict mode", "timeout"],
        "Scheduling": ["schedule", "scheduling", "cron", "timezone", "missed executions"],
        "Error Handling": ["error handling", "retry", "dead-letter", "dlq", "circuit breaker"],
        "Templates": ["template", "templates", "ticket-to-task", "onboarding sequence"],
    },
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Data Loss": [
        "data loss",
        "lost data",
        "missing data",
        "corrupted",
        "corruption",
        "checksum_mismatch",
        "integrity error",
        "cannot decrypt",
        "destroyed key",
    ],
    "Feature Request": [
        "feature request",
        "request:",
        "currently only",
        "bulk",
        "would like",
        "please add",
        "new feature",
        "enhancement",
        "not scalable",
    ],
    "Billing": [
        "billing",
        "invoice",
        "charged",
        "charge",
        "seat",
        "seats",
        "overage",
        "refund",
        "plan",
        "downgrade",
        "upgrade",
        "pricing",
        "renewal",
    ],
    "Performance": [
        "slow",
        "slowness",
        "timeout",
        "timed out",
        "latency",
        "throughput",
        "lag",
        "spinner",
        "stalled",
        "rate_limit_exceeded",
        "load slowly",
    ],
    "Integration": [
        "integration",
        "webhook",
        "connector",
        "oauth",
        "salesforce",
        "snowflake",
        "slack",
        "jira",
        "endpoint",
        "api",
        "sso",
        "saml",
        "idp",
    ],
    "Onboarding": [
        "onboarding",
        "new customer",
        "new organisation",
        "new organization",
        "setup",
        "configure",
        "provision",
        "invite",
        "rollout",
        "training",
        "first 7 days",
    ],
    "How-To": [
        "how do i",
        "how can",
        "can you explain",
        "guidance",
        "documentation",
        "what is the best",
        "steps to",
        "where do i",
        "help me",
    ],
    "Bug": [
        "bug",
        "error",
        "failing",
        "failed",
        "broken",
        "cannot",
        "can't",
        "unable",
        "exception",
        "not working",
        "mismatch",
    ],
}

TEAM_BY_CATEGORY = {
    "Billing": "Billing Operations",
    "Onboarding": "Customer Onboarding",
    "Performance": "Performance Support",
    "Integration": "Integrations Support",
    "Data Loss": "Engineering Escalation",
    "Feature Request": "Product Management",
    "How-To": "Tier 1 Support",
    "Bug": "Tier 2 Product Support",
}


def _parse_ticket(ticket_input: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ticket_input, dict):
        return ticket_input

    text = ticket_input.strip()
    if text.startswith("{"):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

    subject = ""
    body = text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(lines[0]) < 140:
        subject = lines[0]
        body = "\n".join(lines[1:]).strip() or subject
    return {"subject": subject, "body": body}


def _score_phrases(text: str, phrases: list[str]) -> tuple[float, list[str]]:
    hits = phrase_hits(text, phrases)
    score = sum(2.0 if " " in hit or "_" in hit or "-" in hit else 1.0 for hit in hits)
    return score, hits


def detect_product(text: str, provided_product: str | None = None) -> tuple[str, float, list[str]]:
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for product, aliases in PRODUCT_ALIASES.items():
        score, hits = _score_phrases(text, aliases)
        if normalize_whitespace(provided_product).lower() == product.lower():
            score += 4.0
            hits.append("provided product field")
        scores[product] = score
        evidence[product] = hits

    product, top_score = max(scores.items(), key=lambda item: item[1])
    if top_score <= 0:
        return "Unknown", 0.2, []
    second_score = sorted(scores.values(), reverse=True)[1]
    confidence = min(0.98, 0.45 + (top_score - second_score) / max(top_score + 2.0, 1.0))
    return product, round(confidence, 2), evidence[product][:5]


def detect_product_area(
    text: str,
    product: str,
    provided_area: str | None = None,
) -> tuple[str, float, list[str]]:
    area_map = PRODUCT_AREAS.get(product, {})
    if not area_map:
        return "Unknown", 0.2, []

    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for area, phrases in area_map.items():
        score, hits = _score_phrases(text, phrases)
        if normalize_whitespace(provided_area).lower() == area.lower():
            score += 3.0
            hits.append("provided product_area field")
        scores[area] = score
        evidence[area] = hits

    if not scores:
        return "Unknown", 0.2, []
    area, top_score = max(scores.items(), key=lambda item: item[1])
    if top_score <= 0:
        return "Unknown", 0.25, []
    confidence = min(0.96, 0.4 + top_score / 8.0)
    return area, round(confidence, 2), evidence[area][:5]


def detect_category(text: str) -> tuple[str, float, str, dict[str, float]]:
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for category, phrases in CATEGORY_KEYWORDS.items():
        score, hits = _score_phrases(text, phrases)
        scores[category] = score
        evidence[category] = hits

    lower = text.lower()
    if "request:" in lower or "currently only" in lower:
        scores["Feature Request"] += 3.0
    if extract_error_codes(text):
        scores["Bug"] += 1.0
    if any(code in lower for code in ["rate_limit_exceeded", "err_connection_timeout", "pipeline_stalled"]):
        scores["Performance"] += 2.0
    if any(code in lower for code in ["auth_token_expired", "saml_assertion_expired", "audience_mismatch", "group_not_mapped"]):
        scores["Integration"] += 2.5
        scores["Bug"] += 0.5
    if "truncated at 1000" in lower or "row limit" in lower:
        scores["Billing"] += 2.0
    if "conflict storm" in lower or "thousands of conflicts" in lower:
        scores["Performance"] += 3.0
        scores["Bug"] += 2.0
        scores["Feature Request"] = max(0.0, scores["Feature Request"] - 2.0)

    category, top_score = max(scores.items(), key=lambda item: item[1])
    if top_score <= 0:
        return "How-To", 0.35, "No strong issue pattern was found, so the ticket is treated as a general guidance request.", scores

    second_score = sorted(scores.values(), reverse=True)[1]
    confidence = min(0.97, 0.42 + (top_score - second_score + 1.0) / (top_score + 3.0))
    hits = evidence.get(category) or [category.lower()]
    reason = f"Matched {category.lower()} signals: {', '.join(hits[:4])}."
    return category, round(confidence, 2), reason, scores


def detect_urgency(text: str, category: str, plan_tier: str | None = None) -> tuple[str, float, str]:
    lower = text.lower()
    explicit = re.search(r"\bP([1-4])\b", text.upper())
    if explicit:
        urgency = f"P{explicit.group(1)}"
        return urgency, 0.96, f"Explicit urgency marker {urgency} was present in the ticket."

    impacted_users = [int(value) for value in re.findall(r"(\d+)\s+(?:users|employees|people)", lower)]
    failed_deliveries = [int(value) for value in re.findall(r"failed deliveries since:\s*(\d+)", lower)]
    high_user_impact = max(impacted_users or [0]) >= 25
    severe_delivery_failure = max(failed_deliveries or [0]) >= 500

    p1_hits = phrase_hits(
        lower,
        [
            "business stopped",
            "production down",
            "system down",
            "outage",
            "all users",
            "security breach",
            "data loss",
            "lost data",
            "cannot access any",
        ],
    )
    p2_hits = phrase_hits(
        lower,
        [
            "production",
            "critical",
            "urgently",
            "urgent",
            "no workaround",
            "significant impact",
            "blocked",
            "failing since",
            "unacceptable",
        ],
    )
    p4_hits = phrase_hits(lower, ["cosmetic", "minor", "nice to have", "question", "documentation"])

    if p1_hits and (category in {"Data Loss", "Bug", "Integration", "Performance"} or "all users" in p1_hits):
        return "P1", 0.88, f"Critical impact signal detected: {', '.join(p1_hits[:3])}."

    if p2_hits or high_user_impact or severe_delivery_failure or category == "Data Loss":
        evidence = p2_hits[:3]
        if high_user_impact:
            evidence.append(f"{max(impacted_users)} users impacted")
        if severe_delivery_failure:
            evidence.append(f"{max(failed_deliveries)} failed deliveries")
        return "P2", 0.82, f"Major customer impact signal detected: {', '.join(evidence[:4])}."

    if category == "Feature Request" and not phrase_hits(lower, ["urgent", "blocked", "production"]):
        return "P4", 0.78, "Feature request without a production blocker is low urgency."

    if p4_hits:
        return "P4", 0.76, f"Low-impact signal detected: {', '.join(p4_hits[:3])}."

    if normalize_whitespace(plan_tier).lower() == "enterprise" and category in {"Bug", "Integration", "Performance"}:
        return "P2", 0.72, "Enterprise customer with an operational issue receives elevated priority."

    return "P3", 0.64, "Moderate impact inferred because no critical or low-impact signal dominated."


def recommend_team(product: str, product_area: str, category: str, urgency: str) -> str:
    if urgency == "P1":
        return "Incident Command"
    if category in TEAM_BY_CATEGORY:
        return TEAM_BY_CATEGORY[category]
    if product != "Unknown":
        return f"{product} Support"
    if product_area != "Unknown":
        return f"{product_area} Support"
    return "Tier 1 Support"


def draft_first_response(
    company: str,
    product: str,
    product_area: str,
    category: str,
    urgency: str,
    team: str,
    kb_matches: list[dict[str, object]],
) -> str:
    greeting = f"Hi {company}," if company else "Hi,"
    product_phrase = product if product != "Unknown" else "the affected product"
    area_phrase = f" / {product_area}" if product_area != "Unknown" else ""
    kb_sentence = ""
    if kb_matches:
        top = kb_matches[0]
        kb_sentence = (
            f" I found a relevant knowledge-base match in {top['path']} "
            f"under {top['heading']}: {clip(top['snippet'], 180)}"
        )

    return (
        f"{greeting}\n\n"
        f"Thanks for raising this. I have routed it as {urgency} {category} for "
        f"{product_phrase}{area_phrase}, and the recommended responder group is {team}."
        f"{kb_sentence}\n\n"
        "To keep the investigation moving, please share the environment, product version, "
        "timestamp of the latest failure, and any request IDs or logs you can safely provide. "
        "We will review the relevant configuration and follow up with next steps."
    )


def _kb_excerpts(kb_results: list[dict[str, object]], limit: int = 3) -> str:
    blocks = []
    for doc in kb_results[:limit]:
        blocks.append(
            f"- {doc['path']} / {doc['heading']}: {clip(doc['snippet'], 280)}"
        )
    return "\n".join(blocks) if blocks else "(no strong KB match)"


def _overlay_llm_triage(
    ticket: dict[str, Any],
    kb_results: list[dict[str, object]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    system, user = render_prompt(
        TRIAGE_PROMPT,
        ticket_json=json.dumps(redact_ticket({
            "subject": ticket.get("subject", ""),
            "body": ticket.get("body", ""),
            "plan_tier": ticket.get("plan_tier", ""),
            "company": ticket.get("company", ""),
        }), indent=2),
        kb_excerpts=_kb_excerpts(kb_results),
    )
    try:
        llm_out = complete_json(system, user, temperature=0.0, seed=7)
    except Exception as exc:  # noqa: BLE001 - demo must keep working if the API fails
        baseline["llm_error"] = "Hosted model unavailable; deterministic fallback used."
        baseline["llm_used"] = False
        return baseline

    original_labels = (
        baseline["product"],
        baseline["product_area"],
        baseline["category"],
        baseline["urgency"],
    )
    product = str(llm_out.get("product", "")).strip()
    category = str(llm_out.get("category", "")).strip()
    urgency = str(llm_out.get("urgency", "")).strip()
    if product in ALLOWED_PRODUCTS:
        baseline["product"] = product
    if category in ALLOWED_CATEGORIES:
        baseline["category"] = category
    if urgency in ALLOWED_URGENCY:
        baseline["urgency"] = urgency
    area = normalize_whitespace(llm_out.get("product_area", ""))
    allowed_areas = set(PRODUCT_AREAS.get(baseline["product"], {})) | {"Unknown"}
    if baseline["product_area"] not in allowed_areas:
        baseline["product_area"] = "Unknown"
    if area in allowed_areas:
        baseline["product_area"] = area
    reasoning = llm_out.get("reasoning")
    if isinstance(reasoning, list) and reasoning:
        baseline["reasoning"] = [str(item) for item in reasoning[:6]]
    final_labels = (
        baseline["product"],
        baseline["product_area"],
        baseline["category"],
        baseline["urgency"],
    )
    if final_labels != original_labels:
        baseline["confidence"] = min(float(baseline["confidence"]), 0.75)
        baseline["reasoning"].append("Hosted-model labels were enum-validated against the deterministic baseline.")

    baseline["recommended_team"] = recommend_team(
        baseline["product"],
        baseline["product_area"],
        baseline["category"],
        baseline["urgency"],
    )
    company = normalize_whitespace(ticket.get("company", ""))
    baseline["draft_response"] = draft_first_response(
        company,
        baseline["product"],
        baseline["product_area"],
        baseline["category"],
        baseline["urgency"],
        baseline["recommended_team"],
        kb_results,
    )
    draft = normalize_whitespace(llm_out.get("draft_response", ""))
    if len(draft) >= 80:
        baseline["draft_response"] = llm_out["draft_response"]
    if isinstance(llm_out.get("known_issue"), bool):
        baseline["known_issue_match"] = llm_out["known_issue"] or baseline["known_issue_match"]
    team = normalize_whitespace(llm_out.get("recommended_team", ""))
    if team == baseline["recommended_team"]:
        baseline["recommended_team"] = team
    baseline["llm_used"] = True
    return baseline


def triage_ticket(
    ticket_input: str | dict[str, Any],
    retriever: KnowledgeRetriever | None = None,
    *,
    enable_llm: bool | None = None,
) -> dict[str, Any]:
    ticket = _parse_ticket(ticket_input)
    subject = normalize_whitespace(ticket.get("subject", ""))
    body = normalize_whitespace(ticket.get("body", ""))
    text = first_non_empty(f"{subject}\n{body}", ticket_input)
    retriever = retriever or get_retriever()

    product, product_confidence, product_evidence = detect_product(text, ticket.get("product"))
    product_area, area_confidence, area_evidence = detect_product_area(text, product, ticket.get("product_area"))
    category, category_confidence, category_reason, category_scores = detect_category(text)
    urgency, urgency_confidence, urgency_reason = detect_urgency(text, category, ticket.get("plan_tier"))
    kb_results = [result.to_dict() for result in retriever.search(text, top_k=3)]

    if product == "Unknown" and kb_results:
        for known_product in PRODUCT_ALIASES:
            if known_product.lower() in str(kb_results[0]["title"]).lower():
                product = known_product
                product_confidence = max(product_confidence, 0.55)
                break

    team = recommend_team(product, product_area, category, urgency)
    company = normalize_whitespace(ticket.get("company", ""))
    reasoning = [
        f"Product: {product} ({', '.join(product_evidence) if product_evidence else 'no explicit product signal'}).",
        f"Area: {product_area} ({', '.join(area_evidence) if area_evidence else 'no explicit area signal'}).",
        category_reason,
        urgency_reason,
    ]

    overall_confidence = round(
        (product_confidence + area_confidence + category_confidence + urgency_confidence) / 4,
        2,
    )

    result = {
        "prompt_version": TRIAGE_PROMPT_VERSION,
        "ticket_id": ticket.get("ticket_id"),
        "subject": subject,
        "product": product,
        "product_area": product_area,
        "category": category,
        "urgency": urgency,
        "confidence": overall_confidence,
        "reasoning": reasoning,
        "known_issue_match": bool(kb_results and kb_results[0]["score"] >= 1.0),
        "matched_kb_docs": kb_results,
        "recommended_team": team,
        "draft_response": draft_first_response(company, product, product_area, category, urgency, team, kb_results),
        "llm_used": False,
        "debug_scores": {
            "category": {key: round(value, 2) for key, value in sorted(category_scores.items())},
            "error_codes": extract_error_codes(text),
        },
    }

    should_llm = use_llm() if enable_llm is None else enable_llm
    if should_llm:
        result = _overlay_llm_triage(ticket, kb_results, result)
    return result
