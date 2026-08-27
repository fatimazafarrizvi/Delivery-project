# Prompt versions

| ID | Version | Used by | Notes |
|----|---------|---------|-------|
| `triage-router` | 1.1.0 | Task 1 | JSON enums, KB excerpts, operational draft |
| `tam-risk-extract` | 1.0.0 | Task 2 step 1 | Structured risks and quotes |
| `tam-brief-compose` | 1.0.0 | Task 2 step 2 | Three-section markdown from extracted risks |
| `eval-triage-judge` | 1.0.0 | Task 3 optional | LLM-as-judge rubric |
| `eval-account-judge` | 1.0.0 | Task 3 optional | LLM-as-judge rubric |

## Changelog

- **triage-router@1.1.0** — Require allowed product/category/urgency values; pass KB excerpts; keep a first-response draft.
- **triage-router@1.0.0** — Initial structured triage prompt (superseded).
- **tam-risk-extract@1.0.0** — Extract risks before writing prose so the brief cannot invent incidents.
- **tam-brief-compose@1.0.0** — Compose the TAM brief from the extracted payload only.
- **eval-*-judge@1.0.0** — Optional hosted judges; CI uses the heuristic judge.
