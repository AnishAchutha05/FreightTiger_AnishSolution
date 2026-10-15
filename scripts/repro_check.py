"""Run the pipeline three times and check reproducibility."""

import hashlib
import shutil
import subprocess
import sys

import pandas as pd

from src.config import (
    FLAGGED_ANALYSIS_FILE,
    OUTPUT_COLUMNS,
    OUTPUT_FILE,
    PROJECT_ROOT,
    REPRO_DIR,
    TOKEN_LOG_FILE,
    WEEKLY_ANALYSIS_FILE,
    settings,
)


ARTIFACTS = {
    "weekly_analysis": WEEKLY_ANALYSIS_FILE,
    "flagged_analysis": FLAGGED_ANALYSIS_FILE,
    "output": OUTPUT_FILE,
}


def sha256(path):
    """Return the SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    """Read values as strings so comparisons remain exact."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def compare_csvs(first_path, other_path, ignored_columns=()):
    """Compare CSV rows, columns and values; return useful differences."""
    first = read_csv(first_path)
    other = read_csv(other_path)
    differences = []

    if list(first.columns) != list(other.columns):
        differences.append(
            f"Column mismatch: {list(first.columns)} != "
            f"{list(other.columns)}"
        )

    if len(first) != len(other):
        differences.append(
            f"Row count mismatch: {len(first)} != {len(other)}"
        )

    common_columns = [
        column
        for column in first.columns
        if column in other.columns and column not in ignored_columns
    ]

    for row_index in range(min(len(first), len(other))):
        for column in common_columns:
            left = first.iloc[row_index][column]
            right = other.iloc[row_index][column]

            if left != right:
                differences.append(
                    f"Row {row_index + 1}, {column}: "
                    f"{left!r} != {right!r}"
                )

                if len(differences) >= 12:
                    differences.append("Further differences omitted.")
                    return differences

    return differences


def main():
    REPRO_DIR.mkdir(parents=True, exist_ok=True)

    saved_runs = []

    for run_number in range(1, 4):
        print(f"\n========== PIPELINE RUN {run_number}/3 ==========")

        subprocess.run(
            [sys.executable, "-m", "src.run"],
            cwd=PROJECT_ROOT,
            check=True,
        )

        saved = {}

        for name, source in ARTIFACTS.items():
            if not source.is_file():
                raise FileNotFoundError(
                    f"Expected pipeline output is missing: {source}"
                )

            destination = REPRO_DIR / (
                f"run_{run_number}_{name}.csv"
            )
            shutil.copy2(source, destination)
            saved[name] = destination

        if TOKEN_LOG_FILE.is_file():
            shutil.copy2(
                TOKEN_LOG_FILE,
                REPRO_DIR / f"run_{run_number}_token_log.json",
            )

        saved_runs.append(saved)

    failures = []
    report_lines = [
        "# Three-Run Reproducibility Report",
        "",
        "The complete pipeline was run three times on the same inputs "
        "and configuration.",
        "",
        f"- Embedding model: `{settings.embedding_model_name}`",
        f"- LLM model: `{settings.ollama_model_name}`",
        f"- History window: {settings.history_window_weeks} weeks",
        f"- Own-history threshold: "
        f"{settings.own_history_threshold_pct}%",
        f"- Peer threshold: {settings.peer_threshold_pct}%",
        "",
        "## SHA-256 hashes",
        "",
    ]

    for name in ARTIFACTS:
        report_lines.extend([f"### {name}", ""])

        for run_number, saved in enumerate(saved_runs, start=1):
            report_lines.append(
                f"- Run {run_number}: `{sha256(saved[name])}`"
            )

        report_lines.append("")

    # All diagnostic fields, including the candidate flags, must match.
    for name in ("weekly_analysis", "flagged_analysis"):
        reference = saved_runs[0][name]

        for run_number in (2, 3):
            differences = compare_csvs(
                reference,
                saved_runs[run_number - 1][name],
            )

            if differences:
                failures.append(
                    f"{name}: run 1 vs run {run_number}: "
                    + "; ".join(differences)
                )

    # In the submission, only reason wording may vary.
    reference_output = saved_runs[0]["output"]

    for run_number in (2, 3):
        current_output = saved_runs[run_number - 1]["output"]
        output = read_csv(current_output)

        if list(output.columns) != OUTPUT_COLUMNS:
            failures.append(
                f"Run {run_number}: output schema/order is incorrect."
            )

        differences = compare_csvs(
            reference_output,
            current_output,
            ignored_columns=("reason",),
        )

        if differences:
            failures.append(
                f"output: run 1 vs run {run_number}: "
                + "; ".join(differences)
            )

    report_lines.extend([
        "## Decision consistency",
        "",
        "The following must match exactly across all runs:",
        "",
        "- Weekly numerical analysis and row ordering",
        "- Candidate flags and candidate classification",
        "- Final verdicts and matched note IDs",
        "- Submission values and row ordering",
        "",
        "The `reason` wording is allowed to vary.",
        "",
    ])

    for name in ARTIFACTS:
        reference_hash = sha256(saved_runs[0][name])
        identical_files = all(
            sha256(saved[name]) == reference_hash
            for saved in saved_runs[1:]
        )

        report_lines.append(
            f"- `{name}` full-file hashes identical: "
            f"{'Yes' if identical_files else 'No'}"
        )

    report_lines.extend([
        "",
        "## Result",
        "",
        "PASS" if not failures else "FAIL",
        "",
    ])

    if failures:
        report_lines.append("### Differences requiring attention")
        report_lines.extend(f"- {failure}" for failure in failures)
        report_lines.append("")

    report_path = REPRO_DIR / "REPRO_REPORT.md"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"\nReproducibility report: {report_path}")

    if failures:
        print("REPRODUCIBILITY FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    print("REPRODUCIBILITY PASSED")


if __name__ == "__main__":
    main()