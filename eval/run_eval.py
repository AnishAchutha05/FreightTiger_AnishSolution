"""Evaluate candidate detection, citations, verification and retrieval."""

import json
from pathlib import Path

import pandas as pd

from src.config import (
    FLAGGED_ANALYSIS_FILE,
    OUTPUT_FILE,
    PROJECT_ROOT,
    REPRO_DIR,
    settings,
    WEEKLY_ANALYSIS_FILE,
)
from src.retrieve import retrieve_candidates
from src.verify import verify_candidates


LABELS_FILE = PROJECT_ROOT / "eval" / "labelled.csv"
REPORT_FILE = PROJECT_ROOT / "eval" / "EVAL_REPORT.md"


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def is_true(value):
    return str(value).strip().lower() == "true"


def make_candidate(
    note_id,
    routes,
    start,
    end,
    effect,
    excluded=False,
):
    return {
        "note_id": note_id,
        "distance": 0.1,
        "metadata": {
            "note_id": note_id,
            "applies_to_routes_json": json.dumps(routes),
            "verification_start_date": start,
            "verification_end_date": end,
            "effect": effect,
            "exclude_from_dataset": excluded,
        },
    }


def check_invalid_evidence():
    """Ensure invalid evidence cannot justify a cost increase."""

    cases = [
        (
            "wrong route",
            "Delhi-Jaipur",
            "2025-03-03",
            make_candidate(
                "N001", ["Chennai-Bangalore"],
                "2025-02-24", "2025-03-08",
                "positive_cost_increase",
            ),
        ),
        (
            "outside flood window",
            "Chennai-Bangalore",
            "2025-03-10",
            make_candidate(
                "N001", ["Chennai-Bangalore"],
                "2025-02-24", "2025-03-08",
                "positive_cost_increase",
            ),
        ),
        (
            "cost unaffected",
            "Mumbai-Delhi",
            "2024-07-29",
            make_candidate(
                "N005", ["Mumbai-Delhi"],
                "2024-07-29", "2024-08-04",
                "no_significant_cost_effect",
            ),
        ),
        (
            "explicitly excluded route",
            "Delhi-Jaipur",
            "2024-03-11",
            make_candidate(
                "N004", ["ALL_ROUTES"],
                "2024-03-11", "2024-03-11",
                "not_applicable_to_dataset", True,
            ),
        ),
        (
            "stable conditions",
            "Mumbai-Pune",
            "2025-09-22",
            make_candidate(
                "N006", ["ALL_ROUTES"],
                "2025-09-22", "2025-09-30",
                "stable_conditions_no_increase_evidence",
            ),
        ),
        (
            "diesel note outside conservative window",
            "Delhi-Jaipur",
            "2025-05-12",
            make_candidate(
                "N003", ["ALL_ROUTES"],
                "2025-05-05", "2025-05-11",
                "positive_cost_increase",
            ),
        ),
    ]

    accepted = []

    for name, route, week, candidate in cases:
        result = verify_candidates(route, week, [candidate])

        if result["justified"]:
            accepted.append(name)

    return accepted


def main():
    labels = load_csv(LABELS_FILE)
    analysis = load_csv(WEEKLY_ANALYSIS_FILE)
    flagged = load_csv(FLAGGED_ANALYSIS_FILE)
    output = load_csv(OUTPUT_FILE)

    assert len(labels) == 30, f"Expected 30 labels, got {len(labels)}"

    key_columns = ["route", "week_of"]

    if labels.duplicated(key_columns).any():
        raise ValueError("Evaluation labels contain duplicate route-weeks.")

    for frame in (analysis, flagged, output, labels):
        frame["week_of"] = pd.to_datetime(
            frame["week_of"], errors="raise"
        ).dt.strftime("%Y-%m-%d")

    flag_map = {
        (row["route"], row["week_of"]): is_true(row["candidate_anomaly"])
        for _, row in flagged.iterrows()
    }

    output_map = {
        (row["route"], row["week_of"]): row
        for _, row in output.iterrows()
    }

    tp = fp = fn = tn = 0
    verdict_errors = []
    correct_citations = 0
    expected_citations = 0
    incorrect_justifications = 0

    for _, label in labels.iterrows():
        key = (label["route"], label["week_of"])
        expected = is_true(label["expected_candidate"])
        actual = flag_map.get(key, False)

        if expected and actual:
            tp += 1
        elif actual and not expected:
            fp += 1
        elif expected and not actual:
            fn += 1
        else:
            tn += 1

        actual_output = output_map.get(key)

        if not expected:
            if actual_output is not None:
                verdict_errors.append(
                    f"{key}: unexpected submission row"
                )
            continue

        if actual_output is None:
            verdict_errors.append(f"{key}: candidate missing from output")
            continue

        expected_verdict = label["expected_verdict"]
        expected_note = label["expected_note_id"]
        actual_verdict = actual_output["flagged"]
        actual_note = actual_output["matched_note_id"]

        if actual_verdict != expected_verdict:
            verdict_errors.append(
                f"{key}: expected {expected_verdict!r}, "
                f"got {actual_verdict!r}"
            )

        if actual_note != expected_note:
            verdict_errors.append(
                f"{key}: expected note {expected_note!r}, "
                f"got {actual_note!r}"
            )

        if expected_verdict == "No (justified)":
            expected_citations += 1
            if actual_verdict == expected_verdict and actual_note == expected_note:
                correct_citations += 1

        elif actual_verdict == "No (justified)" or actual_note:
            incorrect_justifications += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    # Test whether known supporting notes appear in the top three results.
    retrieval_hits = 0
    retrieval_total = 0
    analysis_map = {
        (row["route"], row["week_of"]): row
        for _, row in analysis.iterrows()
    }

    for _, label in labels.iterrows():
        expected_note = label["expected_note_id"]

        if not expected_note:
            continue

        key = (label["route"], label["week_of"])
        row = analysis_map[key]

        candidates = retrieve_candidates(
            route=label["route"],
            week_of=label["week_of"],
            current_cost=float(row["cost_per_tonne_km"]),
            pct_vs_own=(
                float(row["pct_vs_own"])
                if row["pct_vs_own"] else None
            ),
            pct_vs_peers=(
                float(row["pct_vs_peers"])
                if row["pct_vs_peers"] else None
            ),
            k=3,
        )

        retrieval_total += 1

        if any(
            item["note_id"] == expected_note
            for item in candidates
        ):
            retrieval_hits += 1

    invalid_acceptances = check_invalid_evidence()

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report = [
        "# Evaluation Report",
        "",
        f"- Labeled route-weeks: {len(labels)}",
        f"- Configured history window: {settings.history_window_weeks}",
        f"- Own-history threshold: {settings.own_history_threshold_pct}%",
        f"- Peer threshold: {settings.peer_threshold_pct}%",
        "",
        "## Candidate-detection metrics",
        "",
        f"- TP: {tp}; FP: {fp}; FN: {fn}; TN: {tn}",
        f"- Precision: {precision:.3f}",
        f"- Recall: {recall:.3f}",
        f"- F1: {f1:.3f}",
        "",
        "## Evidence metrics",
        "",
        f"- Correct expected citations: {correct_citations}/{expected_citations}",
        f"- Incorrect justifications: {incorrect_justifications}",
        f"- Invalid evidence accepted: {len(invalid_acceptances)}/6",
        f"- Retrieval hit rate @3: {retrieval_hits}/{retrieval_total}",
        "",
        "## Result",
        "",
        (
            "PASS"
            if not verdict_errors and not invalid_acceptances
            else "FAIL"
        ),
        "",
    ]

    if verdict_errors:
        report += ["## Output mismatches", ""]
        report += [f"- {error}" for error in verdict_errors]
        report.append("")

    if invalid_acceptances:
        report += ["## Invalid evidence accepted", ""]
        report += [f"- {name}" for name in invalid_acceptances]
        report.append("")

    report += [
        "## Limitations",
        "",
        "This is a small hand-specified regression set. Candidate labels "
        "check conformance with the documented deterministic rule and "
        "known case-study examples; they are not independent real-world "
        "ground truth. These metrics must not be presented as evidence of "
        "production anomaly-prediction accuracy.",
        "",
    ]

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print(f"Evaluation cases: {len(labels)}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"Citation accuracy: {correct_citations}/{expected_citations}")
    print(f"Incorrect justifications: {incorrect_justifications}")
    print(f"Invalid evidence accepted: {len(invalid_acceptances)}/6")
    print(f"Retrieval hit rate @3: {retrieval_hits}/{retrieval_total}")
    print(f"Report: {REPORT_FILE}")

    if verdict_errors or invalid_acceptances:
        raise SystemExit("Evaluation failed; inspect EVAL_REPORT.md.")

    print("Evaluation passed.")


if __name__ == "__main__":
    main()
