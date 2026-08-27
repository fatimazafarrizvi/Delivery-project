"""Single entry point: python -m support_ai <command>"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .account_health import summarize_account
from .evals import write_reports
from .llm import load_dotenv_if_present
from .triage import triage_ticket


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(
        prog="python -m support_ai",
        description="Support AI assistant: ticket triage, TAM briefs, evals, and UI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    triage_parser = sub.add_parser("triage", help="Run Task 1 on a ticket")
    triage_parser.add_argument("--subject", default="")
    triage_parser.add_argument("--body", default="")
    triage_parser.add_argument("--file", help="JSON ticket file or raw text file")
    triage_parser.add_argument("--text", help="Raw ticket text")
    triage_parser.add_argument("--llm", action="store_true", help="Enable the configured hosted LLM overlay")

    brief_parser = sub.add_parser("brief", help="Run Task 2 for an account ID")
    brief_parser.add_argument("account_id")
    brief_parser.add_argument("--days", type=int, default=90)
    brief_parser.add_argument("--as-of", help="Dataset snapshot date (YYYY-MM-DD)")
    brief_parser.add_argument("--llm", action="store_true", help="Enable the configured hosted LLM overlay")

    eval_parser = sub.add_parser("eval", help="Run Task 3 evaluation harness")
    eval_parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Also run the optional hosted-model rubric (requires explicit eval opt-in)",
    )
    sub.add_parser("ui", help="Launch the Streamlit demo")

    args = parser.parse_args(argv)

    if args.command == "triage":
        ticket: dict | str
        if args.file:
            with open(args.file, encoding="utf-8") as ticket_file:
                content = ticket_file.read()
            ticket = json.loads(content) if content.lstrip().startswith("{") else content
        elif args.text:
            ticket = args.text
        else:
            ticket = {"subject": args.subject, "body": args.body}
        print(json.dumps(triage_ticket(ticket, enable_llm=args.llm), indent=2, default=str))
        return 0

    if args.command == "brief":
        summary = summarize_account(
            args.account_id,
            days=args.days,
            as_of=args.as_of,
            enable_llm=args.llm,
        )
        print(summary["markdown"])
        return 0 if not summary.get("error") else 1

    if args.command == "eval":
        report = write_reports(enable_llm_judge=args.llm_judge)
        summary = report["summary"]
        print(f"Wrote {report['path']} and {report['json_path']}")
        print(
            f"{summary['passed_cases']}/{summary['total_cases']} passed; "
            f"average score {summary['average_score']}"
        )
        return 0 if summary["failed_cases"] == 0 else 1

    if args.command == "ui":
        from .data_loader import PROJECT_ROOT

        app_path = PROJECT_ROOT / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
