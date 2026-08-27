from __future__ import annotations

import streamlit as st

from support_ai.account_health import dataset_as_of, summarize_account
from support_ai.data_loader import load_accounts, load_tickets
from support_ai.evals import render_markdown_report, run_all_evals
from support_ai.llm import env_status, load_dotenv_if_present, stream_text
from support_ai.triage import triage_ticket


load_dotenv_if_present()

st.set_page_config(
    page_title="Support AI Assistant",
    page_icon="AI",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_data():
    tickets = load_tickets()
    accounts = load_accounts()
    return tickets, accounts


def ticket_label(ticket: dict) -> str:
    return f"{ticket['ticket_id']} - {ticket['subject'][:78]}"


def account_label(account: dict) -> str:
    return f"{account['account_id']} - {account['company']} ({account['health_status']})"


tickets, accounts = cached_data()
as_of_default = dataset_as_of(tickets).date()

st.title("Support AI Assistant")
st.caption("Ticket triage, knowledge-base retrieval, TAM account briefs, and eval checks for the internship starter dataset.")

with st.sidebar:
    st.header("Dataset")
    st.metric("Tickets", len(tickets))
    st.metric("Accounts", len(accounts))
    st.metric("Snapshot date", as_of_default.isoformat())
    st.divider()
    llm_state = env_status()
    st.caption(f"LLM overlay: {'on' if llm_state['enabled'] else 'off'} — {llm_state['reason']}")
    use_hosted_llm = st.toggle(
        "Use hosted LLM overlay",
        value=False,
        disabled=not llm_state["key_present"],
        help="Off keeps runs deterministic. When enabled, redacted inputs are sent to the configured provider.",
    )

triage_tab, account_tab, eval_tab = st.tabs(["Ticket triage", "TAM account brief", "Evaluation"])

with triage_tab:
    st.subheader("Intelligent ticket triage")
    st.caption("Paste a ticket or load a sample. The assistant classifies, retrieves KB context, routes, and drafts a first response.")

    sample_options = ["Custom ticket"] + [ticket_label(ticket) for ticket in tickets[:50]]
    selected_ticket_label = st.selectbox("Sample input", sample_options)
    selected_ticket = None
    if selected_ticket_label != "Custom ticket":
        selected_id = selected_ticket_label.split(" - ", 1)[0]
        selected_ticket = next(ticket for ticket in tickets if ticket["ticket_id"] == selected_id)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        subject = st.text_input(
            "Subject",
            value=selected_ticket["subject"] if selected_ticket else "DataBridge Pro pipeline timing out in production",
        )
        company = st.text_input(
            "Company",
            value=selected_ticket.get("company", "") if selected_ticket else "Example Corp",
        )
        plan_tier = st.selectbox(
            "Plan tier",
            ["Starter", "Professional", "Business", "Enterprise"],
            index=["Starter", "Professional", "Business", "Enterprise"].index(selected_ticket.get("plan_tier", "Business")) if selected_ticket else 2,
        )
    with col_b:
        body = st.text_area(
            "Body",
            value=selected_ticket["body"] if selected_ticket else "Our production DataBridge Pro pipeline has ERR_CONNECTION_TIMEOUT after 30s and backlog is growing for 80 users. We have no workaround.",
            height=220,
        )

    if st.button("Run triage", type="primary"):
        ticket_input = {
            "subject": subject,
            "body": body,
            "company": company,
            "plan_tier": plan_tier,
        }
        if selected_ticket:
            for field in ("ticket_id", "account_id", "product", "product_area"):
                if selected_ticket.get(field):
                    ticket_input[field] = selected_ticket[field]

        result = triage_ticket(ticket_input, enable_llm=use_hosted_llm)
        metric_cols = st.columns(5)
        metric_cols[0].metric("Product", result["product"])
        metric_cols[1].metric("Area", result["product_area"])
        metric_cols[2].metric("Category", result["category"])
        metric_cols[3].metric("Urgency", result["urgency"])
        metric_cols[4].metric("Confidence", f"{int(result['confidence'] * 100)}%")

        st.subheader("Reasoning")
        for reason in result["reasoning"]:
            st.write(f"- {reason}")

        st.subheader("Recommended route")
        st.info(result["recommended_team"])

        st.subheader("Draft first response")
        draft_box = st.empty()
        streamed = ""
        for chunk in stream_text(result["draft_response"]):
            streamed += chunk
            draft_box.markdown(streamed)
        st.caption(f"Prompt {result['prompt_version']} · LLM overlay {'used' if result.get('llm_used') else 'not used'}")

        st.subheader("Knowledge-base matches")
        if result["matched_kb_docs"]:
            for index, doc in enumerate(result["matched_kb_docs"], start=1):
                with st.expander(f"{index}. {doc['title']} / {doc['heading']} ({doc['score']})", expanded=index == 1):
                    st.code(doc["path"], language="text")
                    st.write(doc["snippet"])
        else:
            st.warning("No strong knowledge-base match found.")

        with st.expander("Structured output JSON"):
            st.json(result)

with account_tab:
    st.subheader("TAM account health brief")
    st.caption("Select an account. The assistant joins account data with the last 90 days of linked tickets.")

    default_account_index = 0
    for idx, account in enumerate(accounts):
        if account["account_id"] == "ACC-3336":
            default_account_index = idx
            break

    selected_account_label = st.selectbox(
        "Account",
        [account_label(account) for account in accounts],
        index=default_account_index,
    )
    selected_account_id = selected_account_label.split(" - ", 1)[0]
    col_a, col_b = st.columns([1, 1])
    with col_a:
        days = st.number_input("Ticket lookback days", min_value=30, max_value=365, value=90, step=30)
    with col_b:
        as_of = st.date_input("As of date", value=as_of_default)

    if st.button("Generate brief", type="primary"):
        summary = summarize_account(
            selected_account_id,
            days=int(days),
            as_of=as_of,
            enable_llm=use_hosted_llm,
        )
        if summary.get("error"):
            st.error(summary["error"])
        else:
            metrics = summary["metrics"]
            metric_cols = st.columns(5)
            metric_cols[0].metric("Health", metrics["health_status"])
            arr_label = f"${metrics['arr_usd']:,}" if metrics["arr_usd"] is not None else "Unknown"
            utilization_label = (
                f"{metrics['seat_utilization_pct']}%"
                if metrics["seat_utilization_pct"] is not None
                else "Unknown"
            )
            metric_cols[1].metric("ARR", arr_label)
            metric_cols[2].metric("Seat use", utilization_label)
            metric_cols[3].metric("Open tickets", metrics["open_tickets"])
            metric_cols[4].metric("Linked tickets", summary["ticket_count"])

            brief_box = st.empty()
            streamed = ""
            for chunk in stream_text(summary["markdown"], chunk_size=48):
                streamed += chunk
                brief_box.markdown(streamed)
            st.download_button(
                "Download brief",
                data=summary["markdown"],
                file_name=f"{selected_account_id}_brief.md",
                mime="text/markdown",
            )

            if summary.get("risks"):
                with st.expander("Risk details"):
                    st.dataframe(summary["risks"], width="stretch")
            if summary.get("tickets"):
                with st.expander("Tickets in window"):
                    st.dataframe(
                        [
                            {
                                "ticket_id": ticket["ticket_id"],
                                "created_at": ticket["created_at"],
                                "status": ticket["status"],
                                "urgency": ticket["urgency"],
                                "category": ticket["category"],
                                "subject": ticket["subject"],
                            }
                            for ticket in summary["tickets"]
                        ],
                        width="stretch",
                    )

with eval_tab:
    st.subheader("Evaluation harness")
    st.caption("Runs deterministic checks for triage and account-brief outputs, including adversarial cases.")

    if st.button("Run evals", type="primary"):
        report = run_all_evals()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total", report["summary"]["total_cases"])
        col_b.metric("Passed", report["summary"]["passed_cases"])
        col_c.metric("Average score", report["summary"]["average_score"])
        rows = [
            {
                "task": result["task"],
                "case": result["case"],
                "passed": result["passed"],
                "score": result["score"],
                "summary": result["output_summary"],
            }
            for result in report["results"]
        ]
        st.dataframe(rows, width="stretch")
        markdown_report = render_markdown_report(report)
        st.download_button(
            "Download eval report",
            data=markdown_report,
            file_name="eval_report.md",
            mime="text/markdown",
        )
        with st.expander("Raw report JSON"):
            st.json(report)
