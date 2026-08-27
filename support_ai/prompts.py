"""Versioned prompts used when an LLM is available.

The same prompt IDs are executed by a local deterministic fallback so the
pipeline is inspectable even without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    version: str
    changelog: str
    system: str
    user_template: str

    @property
    def identifier(self) -> str:
        return f"{self.prompt_id}@{self.version}"


TRIAGE_PROMPT = Prompt(
    prompt_id="triage-router",
    version="1.1.0",
    changelog="v1.1.0: require allowed enums, use KB excerpts, keep draft operational. v1.0.0: initial structured triage.",
    system=(
        "You are an internal technical-support triage assistant. "
        "Use only the ticket text and knowledge-base excerpts. "
        "Do not invent products, error codes, or customers. "
        "If evidence is weak, choose Unknown / How-To / P3 rather than guessing."
    ),
    user_template="""Ticket JSON:
{ticket_json}

Knowledge-base excerpts:
{kb_excerpts}

Return a JSON object with keys:
product: one of DataBridge Pro, CloudSync, AnalyticsHub, SecureVault, WorkflowEngine, Unknown
product_area: a short module name or Unknown
category: one of Bug, Feature Request, How-To, Performance, Billing, Integration, Onboarding, Data Loss
urgency: one of P1, P2, P3, P4
reasoning: 3 to 6 short strings
known_issue: boolean
recommended_team: string
draft_response: a first-response email the agent can send
""",
)

RISK_EXTRACT_PROMPT = Prompt(
    prompt_id="tam-risk-extract",
    version="1.0.0",
    changelog="v1.0.0: extract structured risks and quotes from account + tickets before writing prose.",
    system=(
        "You extract churn and escalation signals for a TAM. "
        "Only use the provided account JSON and ticket snippets. "
        "Every ticket risk must include a direct quote from that ticket."
    ),
    user_template="""Account JSON:
{account_json}

Ticket snippets:
{ticket_snippets}

Return JSON with key risks: a list of objects with severity (high|medium|low),
source (account|account_note|ticket), reason, quote, and optional ticket_id.
""",
)

BRIEF_COMPOSE_PROMPT = Prompt(
    prompt_id="tam-brief-compose",
    version="1.0.0",
    changelog="v1.0.0: compose a 3-section TAM brief from extracted risks only.",
    system=(
        "You write a concise TAM pre-QBR brief. Temperature-equivalent: be dry and factual. "
        "Do not add risks that are not in the extracted payload."
    ),
    user_template="""Account metrics JSON:
{metrics_json}

Extracted risks JSON:
{risks_json}

Write JSON with:
executive_summary: 3 to 5 sentences
talking_points: 3 to 6 short bullets
markdown: a markdown brief with exactly these headings:
## Executive Summary
## Open Risks & Flagged Issues
## Recommended Talking Points
""",
)

TRIAGE_JUDGE_PROMPT = Prompt(
    prompt_id="eval-triage-judge",
    version="1.0.0",
    changelog="v1.0.0: rubric judge for triage quality.",
    system="You are a strict evaluator for support-ticket triage. Score only from the rubric.",
    user_template="""Ticket:
{ticket_json}

Triage output:
{output_json}

Rubric (0-1 each): product_supported, category_plausible, urgency_plausible,
kb_used_if_error_code, draft_actionable, reasoning_specific.
Return JSON with scores (object) and overall (0-1) and notes (string).
""",
)

ACCOUNT_JUDGE_PROMPT = Prompt(
    prompt_id="eval-account-judge",
    version="1.0.0",
    changelog="v1.0.0: rubric judge for TAM briefs.",
    system="You are a strict evaluator for TAM account briefs.",
    user_template="""Account metrics:
{metrics_json}

Brief markdown:
{markdown}

Rubric (0-1 each): has_three_sections, uses_account_facts, flags_material_risks,
quotes_ticket_when_present, talking_points_actionable.
Return JSON with scores, overall, and notes.
""",
)

PROMPTS = {
    TRIAGE_PROMPT.prompt_id: TRIAGE_PROMPT,
    RISK_EXTRACT_PROMPT.prompt_id: RISK_EXTRACT_PROMPT,
    BRIEF_COMPOSE_PROMPT.prompt_id: BRIEF_COMPOSE_PROMPT,
    TRIAGE_JUDGE_PROMPT.prompt_id: TRIAGE_JUDGE_PROMPT,
    ACCOUNT_JUDGE_PROMPT.prompt_id: ACCOUNT_JUDGE_PROMPT,
}


def render_prompt(prompt: Prompt, **kwargs: str) -> tuple[str, str]:
    return prompt.system, prompt.user_template.format(**kwargs)
