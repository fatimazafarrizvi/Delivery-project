"""Evaluation harness for the Streamlit demo and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .account_health import summarize_account
from .data_loader import PROJECT_ROOT
from .triage import triage_ticket


TRIAGE_CASES: list[dict[str, Any]] = [
    {
        "name": "databridge_timeout_production",
        "input": {
            "subject": "DataBridge Pro pipeline timing out in production",
            "body": "Our production DataBridge Pro pipeline has ERR_CONNECTION_TIMEOUT after 30s and backlog is growing for 80 users. No workaround.",
            "plan_tier": "Enterprise",
        },
        "expected": {
            "product": "DataBridge Pro",
            "product_area": {"Pipeline Monitoring", "Connectors", "Data Ingestion"},
            "category": {"Performance", "Bug"},
            "urgency": {"P1", "P2"},
            "kb_contains": "Performance",
        },
    },
    {
        "name": "securevault_saml_login_blocked",
        "input": {
            "subject": "All users blocked by SAML_ASSERTION_EXPIRED in SecureVault",
            "body": "After our SSO migration, all users cannot log in to SecureVault. Audit logs show SAML_ASSERTION_EXPIRED.",
        },
        "expected": {
            "product": "SecureVault",
            "product_area": {"SSO Configuration", "Authentication"},
            "category": {"Integration", "Bug"},
            "urgency": {"P1", "P2"},
            "kb_contains": "Authentication",
        },
    },
    {
        "name": "analyticshub_export_row_limit",
        "input": {
            "subject": "AnalyticsHub export truncated at 1000 rows",
            "body": "We are on Starter and our CSV export is truncated at 1000 rows. Is this a bug or a plan limit?",
        },
        "expected": {
            "product": "AnalyticsHub",
            "category": {"Billing", "How-To"},
            "urgency": {"P3", "P4"},
            "kb_contains": "AnalyticsHub",
        },
    },
    {
        "name": "cloudsync_conflict_storm",
        "input": {
            "subject": "Conflict storm after bulk upload in CloudSync",
            "body": "CloudSync created thousands of conflicts after an offline bulk upload. Admins need a bulk conflict resolution path.",
        },
        "expected": {
            "product": "CloudSync",
            "category": {"Performance", "Bug", "How-To"},
            "urgency": {"P2", "P3"},
            "kb_contains": "CloudSync",
        },
    },
    {
        "name": "workflowengine_bulk_action_feature",
        "input": {
            "subject": "Request: bulk edit actions in WorkflowEngine",
            "body": "Currently WorkflowEngine only lets us edit one action at a time. Please add a bulk operation because manual updates are not scalable.",
        },
        "expected": {
            "product": "WorkflowEngine",
            "category": {"Feature Request"},
            "urgency": {"P3", "P4"},
            "kb_contains": "WorkflowEngine",
        },
    },
    {
        "name": "adversarial_ambiguous_ticket",
        "input": {
            "subject": "Something is broken",
            "body": "It failed again. Please fix it soon.",
        },
        "expected": {
            "product": "Unknown",
            "product_area": "Unknown",
            "category": {"Bug", "How-To"},
            "urgency": {"P3", "P4"},
            "kb_contains": "",
            "known_issue": False,
        },
    },
]

ACCOUNT_CASES: list[dict[str, Any]] = [
    {
        "name": "at_risk_inactive_with_competitor_note",
        "account_id": "ACC-3336",
        "min_risks": 4,
        "required_terms": ["At Risk", "inactive", "competing vendor", "P1 claim disagrees"],
        "required_risk_sources": ["data_quality"],
    },
    {
        "name": "new_account_onboarding_context",
        "account_id": "ACC-5748",
        "min_risks": 1,
        "required_terms": ["New", "ticket window"],
    },
    {
        "name": "churning_account",
        "account_id": "ACC-7042",
        "min_risks": 4,
        "required_terms": ["Churning", "declining"],
    },
    {
        "name": "healthy_account_low_utilization",
        "account_id": "ACC-9010",
        "min_risks": 1,
        "required_terms": ["Healthy", "Seat utilization is low"],
    },
    {
        "name": "adversarial_missing_account",
        "account_id": "ACC-NOT-REAL",
        "expect_error": True,
        "required_terms": ["Account not found"],
    },
    {
        "name": "adversarial_incomplete_account",
        "account_id": "ACC-INCOMPLETE",
        "account": {"account_id": "ACC-INCOMPLETE", "company": "Partial Corp"},
        "min_risks": 1,
        "required_terms": ["incomplete", "Partial Corp"],
        "forbidden_terms": ["$0 ARR"],
    },
]


def _as_set(value: object) -> set[object]:
    if isinstance(value, set):
        return value
    return {value}


def heuristic_triage_judge(output: dict[str, Any]) -> dict[str, Any]:
    scores = {
        "structured": 1.0 if all(output.get(key) for key in ("product", "category", "urgency", "recommended_team")) else 0.0,
        "reasoning": 1.0 if len(output.get("reasoning") or []) >= 3 else 0.4,
        "draft_actionable": 1.0 if len(output.get("draft_response") or "") >= 120 else 0.2,
        "kb_shape": 1.0 if isinstance(output.get("matched_kb_docs"), list) else 0.0,
    }
    overall = round(sum(scores.values()) / len(scores), 2)
    return {"mode": "heuristic", "scores": scores, "overall": overall}


def heuristic_account_judge(output: dict[str, Any]) -> dict[str, Any]:
    markdown = output.get("markdown") or ""
    scores = {
        "three_sections": 1.0 if all(name in markdown for name in ["Executive Summary", "Open Risks", "Recommended Talking Points"]) else 0.0,
        "uses_facts": 1.0 if output.get("company") or output.get("error") else 0.0,
        "talking_points": 1.0 if len(output.get("talking_points") or []) >= 1 else 0.0,
        "deterministic_shape": 1.0 if output.get("prompt_version") else 0.0,
    }
    overall = round(sum(scores.values()) / len(scores), 2)
    return {"mode": "heuristic", "scores": scores, "overall": overall}


def llm_judge_optional(
    task: str,
    payload: dict[str, Any],
    *,
    enabled: bool = False,
) -> dict[str, Any] | None:
    from .llm import complete_json, use_llm
    from .prompts import ACCOUNT_JUDGE_PROMPT, TRIAGE_JUDGE_PROMPT, render_prompt
    import json
    import os

    if (
        not enabled
        or not use_llm()
        or os.getenv("SUPPORT_AI_EVAL_LLM", "0").lower() not in {"1", "true", "yes"}
    ):
        return None
    if task == "triage":
        system, user = render_prompt(
            TRIAGE_JUDGE_PROMPT,
            ticket_json=json.dumps(payload["input"], indent=2),
            output_json=json.dumps(payload["output_summary"], indent=2),
        )
    else:
        system, user = render_prompt(
            ACCOUNT_JUDGE_PROMPT,
            metrics_json=json.dumps(payload.get("output_summary", {}), indent=2),
            markdown=payload.get("markdown", ""),
        )
    try:
        judged = complete_json(system, user, temperature=0.0, seed=7)
        judged["mode"] = "llm"
        return judged
    except Exception as exc:  # noqa: BLE001
        return {"mode": "llm", "error": str(exc), "overall": 0.0}


def score_triage_case(
    case: dict[str, Any],
    *,
    enable_llm_judge: bool = False,
) -> dict[str, Any]:
    output = triage_ticket(case["input"], enable_llm=False)
    expected = case["expected"]
    checks: list[tuple[str, bool]] = []
    checks.append(("product", output["product"] in _as_set(expected["product"])))
    if "product_area" in expected:
        checks.append(("product_area", output["product_area"] in _as_set(expected["product_area"])))
    checks.append(("category", output["category"] in _as_set(expected["category"])))
    checks.append(("urgency", output["urgency"] in _as_set(expected["urgency"])))
    checks.append(("reasoning", bool(output["reasoning"]) and output["confidence"] >= 0.35))
    checks.append(("draft_response", len(output["draft_response"]) >= 120))
    kb_contains = expected.get("kb_contains", "")
    if kb_contains:
        kb_blob = " ".join(
            f"{doc['title']} {doc['path']} {doc['heading']} {doc['snippet']}"
            for doc in output["matched_kb_docs"]
        ).lower()
        checks.append(("kb_match", kb_contains.lower() in kb_blob))
    else:
        checks.append(("adversarial_abstains", output["product"] == "Unknown"))
        checks.append(("no_known_issue", output["known_issue_match"] is expected.get("known_issue", False)))

    judge = heuristic_triage_judge(output)
    check_score = sum(1 for _, passed in checks if passed) / len(checks)
    score = (check_score + judge["overall"]) / 2
    result = {
        "task": "triage",
        "case": case["name"],
        "passed": all(passed for _, passed in checks) and judge["overall"] >= 0.75,
        "score": round(score, 2),
        "checks": {name: passed for name, passed in checks},
        "judge": judge,
        "output_summary": {
            "product": output["product"],
            "product_area": output["product_area"],
            "category": output["category"],
            "urgency": output["urgency"],
            "team": output["recommended_team"],
        },
    }
    llm_judged = llm_judge_optional(
        "triage",
        {"input": case["input"], "output_summary": result["output_summary"]},
        enabled=enable_llm_judge,
    )
    if llm_judged:
        result["llm_judge"] = llm_judged
    return result


def score_account_case(
    case: dict[str, Any],
    *,
    enable_llm_judge: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"enable_llm": False}
    if case.get("account") is not None:
        kwargs["account"] = case["account"]
    first = summarize_account(case["account_id"], **kwargs)
    second = summarize_account(case["account_id"], **kwargs)
    markdown = first["markdown"]
    checks: list[tuple[str, bool]] = []

    if case.get("expect_error"):
        checks.append(("expected_error", bool(first.get("error"))))
    else:
        checks.append(("no_error", not first.get("error")))
        checks.append(("three_sections", all(section in markdown for section in ["Executive Summary", "Open Risks", "Recommended Talking Points"])))
        checks.append(("min_risks", len(first["risks"]) >= case.get("min_risks", 0)))
        checks.append(("talking_points", len(first["talking_points"]) >= 2))
        checks.append(("deterministic", first["markdown"] == second["markdown"]))
        if any(risk.get("source") == "ticket" for risk in first["risks"]):
            checks.append(("ticket_quote", any(risk.get("quote") for risk in first["risks"] if risk.get("source") == "ticket")))
        for source in case.get("required_risk_sources", []):
            checks.append((f"risk_source:{source}", any(risk.get("source") == source for risk in first["risks"])))

    for term in case.get("required_terms", []):
        checks.append((f"contains:{term}", term.lower() in markdown.lower()))
    for term in case.get("forbidden_terms", []):
        checks.append((f"excludes:{term}", term.lower() not in markdown.lower()))

    judge = heuristic_account_judge(first)
    check_score = sum(1 for _, passed in checks if passed) / len(checks)
    score = (check_score + judge["overall"]) / 2
    result = {
        "task": "account_brief",
        "case": case["name"],
        "passed": all(passed for _, passed in checks) and judge["overall"] >= 0.75,
        "score": round(score, 2),
        "checks": {name: passed for name, passed in checks},
        "judge": judge,
        "output_summary": {
            "account_id": case["account_id"],
            "company": first.get("company", ""),
            "risk_count": len(first.get("risks", [])),
            "ticket_count": first.get("ticket_count", 0),
            "error": first.get("error", ""),
        },
    }
    llm_judged = llm_judge_optional(
        "account",
        {"output_summary": result["output_summary"], "markdown": markdown},
        enabled=enable_llm_judge,
    )
    if llm_judged:
        result["llm_judge"] = llm_judged
    return result


def run_all_evals(*, enable_llm_judge: bool = False) -> dict[str, Any]:
    results = [
        score_triage_case(case, enable_llm_judge=enable_llm_judge)
        for case in TRIAGE_CASES
    ]
    results.extend(
        score_account_case(case, enable_llm_judge=enable_llm_judge)
        for case in ACCOUNT_CASES
    )
    passed = sum(1 for result in results if result["passed"])
    average_score = sum(result["score"] for result in results) / len(results)
    return {
        "summary": {
            "total_cases": len(results),
            "passed_cases": passed,
            "failed_cases": len(results) - passed,
            "average_score": round(average_score, 2),
        },
        "results": results,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Total cases: {report['summary']['total_cases']}",
        f"- Passed cases: {report['summary']['passed_cases']}",
        f"- Failed cases: {report['summary']['failed_cases']}",
        f"- Average score: {report['summary']['average_score']}",
        "",
        "| Task | Case | Pass | Score | Failed checks | Output summary |",
        "|------|------|------|-------|---------------|----------------|",
    ]
    for result in report["results"]:
        output_summary = ", ".join(f"{key}={value}" for key, value in result["output_summary"].items() if value != "")
        failed_checks = ", ".join(name for name, passed in result["checks"].items() if not passed) or "—"
        lines.append(
            f"| {result['task']} | {result['case']} | {result['passed']} | {result['score']} | {failed_checks} | {output_summary} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    path: str | Path | None = None,
    *,
    enable_llm_judge: bool = False,
) -> dict[str, Any]:
    report = run_all_evals(enable_llm_judge=enable_llm_judge)
    markdown_path = Path(path) if path else PROJECT_ROOT / "eval_report.md"
    json_path = markdown_path.with_suffix(".json")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(markdown_path)
    report["json_path"] = str(json_path)
    return report


def write_markdown_report(path: str | Path | None = None) -> dict[str, Any]:
    return write_reports(path)
