from support_ai.evals import write_reports


if __name__ == "__main__":
    report = write_reports()
    summary = report["summary"]
    print(f"Wrote {report['path']} and {report['json_path']}")
    print(
        f"{summary['passed_cases']}/{summary['total_cases']} passed; "
        f"average score {summary['average_score']}"
    )
