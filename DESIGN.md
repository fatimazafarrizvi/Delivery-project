# Design Note

This assistant is an internal tool for tier-1/tier-2 support and TAMs. The default runtime is a deterministic retrieval-and-rules pipeline so a reviewer can `pip install` and run it with no secrets. When `OPENAI_API_KEY` is present, Task 1 and Task 2 overlay the same structured prompts on top of that baseline: RAG excerpts go into a JSON-mode triage prompt, and the TAM flow is a two-step chain (risk extraction, then brief composition) with temperature 0 and a fixed seed. Prompt identifiers live in `support_ai/prompts.py` and are recorded on every output.

## Failure modes

The first failure mode is misclassification of ambiguous tickets. Keyword overlap and even an LLM overlay can attach a product or category because a secondary phrase appeared in the body (for example “audit logs” on an SSO outage). Detection should include low-confidence routing, disagreement between product and product-area signals, and agent correction rates in the ticketing tool. Mitigation is a human review queue below a confidence threshold, plus clamping LLM labels to allowed enums so the model cannot invent a sixth product.

The second failure mode is retrieving a related but non-actionable knowledge-base chunk. TF-IDF overlap with error-code boosts is strong for `ERR_CONNECTION_TIMEOUT` and weak for vague “it is slow” tickets. Detection is click-through and “used in reply” telemetry on KB cards, plus eval cases that require a named doc family. Mitigation is heading-aware chunking (already in place), later embedding retrieval, and documenting freshness of each markdown file.

The third failure mode is a TAM brief that overstates churn because CRM fields are stale or internally inconsistent. The starter data itself contains this: an escalation note can mention consecutive P1s while `p1_tickets_last_30d` is zero. Detection is freshness timestamps on account exports and a warning when ticket-derived P1 counts disagree with the account summary. Mitigation is showing source quotes, listing incomplete fields instead of crashing, and never treating a generated sentence as a system of record.

## Latency vs quality

The concrete trade-off is local retrieval and rules as the default instead of a hosted completion for every token. Classification and KB ranking finish in tens of milliseconds on this dataset; drafts are templated unless an LLM key is configured. That is the right default for a demo and for CI. If latency were the hard constraint, I would keep the current path, pre-index KB chunks, cache briefs by `(account_id, as_of, prompt_version)`, and only call a model when an agent clicks “rewrite draft”. If quality were the hard constraint, I would always run the two-step TAM chain and add a reranker on KB chunks, accepting 1–3 seconds of extra latency.

## Data sensitivity

Tickets and accounts may contain names, emails, and tokens. The default design never sends data to an external API. Hosted calls require explicit user opt-in. Before a request, `support_ai/pii.py` redacts emails, account IDs, direct company/TAM/contact fields, bearer/secret prefixes, phone-like strings, and sign-offs. Free-text proper names cannot be identified reliably by regex alone, so production deployment would also require an approved enterprise endpoint, retention controls, and a named-entity redaction layer. Prompts instruct the model not to invent customers. Credentials are read from environment variables; `.env.example` documents names without values. Eval CI sets `SUPPORT_AI_USE_LLM=0` so GitHub Actions never needs a secret.

## Scaling

At 10× ticket volume the first break is loading `tickets.json` into memory and scanning it per account brief. The next break is rebuilding the KB index on every process start. A production port would keep tickets in DuckDB or Postgres with an index on `account_id` and `created_at`, persist the KB index, and run evals in CI on every prompt or rule change. LLM overlays would need rate limits and a queue; the deterministic path would still serve as the fallback when the model is unavailable. Task 1 and Task 2 stream their drafts in the UI so agents see progress without waiting on a full page refresh.
