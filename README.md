# Support AI Assistant

Internal tooling for technical support triage and TAM account briefs, built on the US Delivery Internship starter dataset only.

## What it does

- **Task 1 — Ticket triage:** ingest free-text or JSON, classify product / area / category / urgency, retrieve knowledge-base matches, recommend a responder team, and stream a first-response draft. Callable as `triage_ticket()` or via the CLI.
- **Task 2 — TAM account brief:** load an account plus the last 90 days of tickets and produce a deterministic three-section brief with risk flags, direct ticket quotes, and CRM/ticket P1 reconciliation. An optional two-step LLM chain is explicitly enabled by the user.
- **Task 3 — Evaluation harness:** rule checks, heuristic LLM-as-judge rubrics, adversarial cases, Markdown + JSON reports.
- **Task 4 — Design note:** see [DESIGN.md](DESIGN.md).

The default path is offline and deterministic. Hosted JSON-mode overlays require a local key plus an explicit UI toggle or CLI `--llm` flag.

## Setup

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional hosted-model setup:
cp .env.example .env  # Windows: copy .env.example .env
```

## Single entry point

```powershell
python -m support_ai triage --subject "DataBridge Pro pipeline timeout" --body "ERR_CONNECTION_TIMEOUT after 30s in production, 80 users, no workaround."
python -m support_ai brief ACC-3336
python -m support_ai eval
python -m support_ai ui
```

`python -m support_ai eval` writes `eval_report.md` and `eval_report.json`.
`python -m support_ai ui` launches Streamlit (`app.py`).
Add `--llm` to `triage` or `brief` only when you intentionally want the hosted overlay. Use `brief --as-of YYYY-MM-DD` to reproduce a historical snapshot.

## Sample runs

**Task 1.** The command above deterministically returns DataBridge Pro / Pipeline Monitoring / Performance / P2, surfaces the performance troubleshooting doc, routes to Performance Support, and prints reasoning plus a first-response draft.

**Task 2.** `python -m support_ai brief ACC-3336` mentions Omni Consumer Products, At Risk / inactive usage, the competing-vendor note, and the inconsistent P1 claim, with the three required headings.

**Task 3.** `python -m support_ai eval` runs 12 deterministic cases (6 per task), including ambiguous input, missing account data, incomplete CRM data, and P1-count disagreement. A case fails if any required check fails; the Markdown report lists failed checks.

## Architecture

1. Normalize the ticket or account payload (including incomplete CRM rows).
2. Retrieve heading-aware knowledge-base chunks (TF-IDF overlap + error-code boosts).
3. Score product, area, category, and urgency with inspectable rules.
4. When explicitly enabled, run versioned prompts (`PROMPTS.md`) with direct identifiers redacted, temperature 0, and enum clamping.
5. For TAM briefs, extract risks first, then compose the three-section memo (prompt chaining). Evals always pin `enable_llm=False` so CI stays deterministic.

## Design note

The scored written section is in [DESIGN.md](DESIGN.md) (~600 words): failure modes, latency vs quality, data sensitivity, and scaling.

## Deploy on Streamlit Community Cloud

The app file is `app.py`. The hosted app runs **without a key** (deterministic path). Add a key only in Streamlit’s secret store — never in GitHub.

1. Push this repo to GitHub (do not commit `.env` or `.streamlit/secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Select this repository, branch `main`, and main file `app.py`.
4. Open **App settings → Secrets** and add the variables listed in `.streamlit/secrets.toml.example`, supplying the key only in Streamlit's secret store.
5. Deploy. If you skip secrets, the app works offline. If configured, the sidebar exposes an explicit hosted-overlay toggle.

Local Streamlit can use `.env` or a gitignored `.streamlit/secrets.toml` copied from `.streamlit/secrets.toml.example`.

## Submission checklist

- [ ] Run `pip install -r requirements.txt` in a fresh virtual environment.
- [ ] Run the Task 1 and Task 2 sample commands above.
- [ ] Run `python -m support_ai eval` and confirm all required checks pass.
- [ ] Run `python -m support_ai ui` and demo both workflows.
- [ ] Confirm `.env` and `.streamlit/secrets.toml` are not tracked.
- [ ] Share the GitHub repository.
- [ ] Record a 3–6 minute Loom showing Tasks 1 and 2 plus the eval report.

