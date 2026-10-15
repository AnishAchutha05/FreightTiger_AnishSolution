
"""Identify unusual weekly freight costs using fixed numerical rules.

This module identifies candidates for investigation.
It does not decide whether a cost increase is justified.
That decision belongs to the evidence-verification stage.
"""

import numpy as np
import pandas as pd

from src.config import (
    FLAGGED_ANALYSIS_FILE,
    OUTPUT_DIR,
    WEEKLY_ANALYSIS_FILE,
    settings,
)


def add_candidate_flags(
    analysis: pd.DataFrame,
    own_history_threshold_pct: float = settings.own_history_threshold_pct,
    peer_threshold_pct: float = settings.peer_threshold_pct,
) -> pd.DataFrame:
    """Mark route-weeks that cross the configured anomaly thresholds."""

    if not np.isfinite(own_history_threshold_pct) or own_history_threshold_pct < 0:
        raise ValueError("The own-history threshold must be finite and non-negative.")

    if not np.isfinite(peer_threshold_pct) or peer_threshold_pct < 0:
        raise ValueError("The peer threshold must be finite and non-negative.")

    required_columns = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "own_history_baseline",
        "pct_vs_own",
        "peer_baseline",
        "pct_vs_peers",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in analysis.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns required for flagging: {missing_columns}"
        )

    if analysis.empty:
        raise ValueError("There are no weekly records to evaluate.")

    data = analysis.copy()

    # Convert numeric fields explicitly. Missing baselines and deviations
    # are allowed, particularly for a route's first observed week.
    numeric_columns = [
        "cost_per_tonne_km",
        "own_history_baseline",
        "pct_vs_own",
        "peer_baseline",
        "pct_vs_peers",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

        values = data[column].dropna().to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column '{column}' contains infinite values."
            )

    if data["cost_per_tonne_km"].isna().any():
        raise ValueError("Current weekly costs cannot be missing.")

    if (data["cost_per_tonne_km"] < 0).any():
        raise ValueError("Current weekly costs cannot be negative.")

    # A route has history when a historical baseline was calculated.
    has_history = data["own_history_baseline"].notna()

    # The assignment requires a cost to be rising before a route with
    # historical data can become an anomaly candidate.
    rising = (
        has_history
        & (
            data["cost_per_tonne_km"]
            > data["own_history_baseline"]
        )
    )

    # Check each comparison independently against its configured threshold.
    exceeds_own_threshold = (
        data["pct_vs_own"]
        .ge(own_history_threshold_pct)
        .fillna(False)
    )

    exceeds_peer_threshold = (
        data["pct_vs_peers"]
        .ge(peer_threshold_pct)
        .fillna(False)
    )

    # With history, rising costs qualify if either comparison crosses
    # its threshold. The two conditions are joined with OR.
    candidate_with_history = (
        rising
        & (
            exceeds_own_threshold
            | exceeds_peer_threshold
        )
    )

    # Without history, we cannot calculate a meaningful own-history
    # deviation. Only a valid peer comparison can identify a candidate.
    has_valid_peer_baseline = (
        data["peer_baseline"].notna()
        & (data["peer_baseline"] > 0)
    )

    candidate_without_history = (
        ~has_history
        & has_valid_peer_baseline
        & exceeds_peer_threshold
    )

    # Keep this Boolean separate from the eventual user-facing verdict.
    data["candidate_anomaly"] = (
        candidate_with_history | candidate_without_history
    ).astype(bool)

    # These diagnostic fields make each decision easier to audit.
    data["rising_vs_own_history"] = rising.astype(bool)

    # Record which numerical condition triggered the candidate flag.
    # This is a diagnostic label, not the final explanation.
    both_thresholds = (
        exceeds_own_threshold & exceeds_peer_threshold
    )

    conditions = [
        candidate_without_history,
        candidate_with_history & both_thresholds,
        candidate_with_history & exceeds_own_threshold,
        candidate_with_history & exceeds_peer_threshold,
    ]

    labels = [
        "peer_only_no_history",
        "own_history_and_peers",
        "own_history",
        "similar_routes",
    ]

    data["candidate_basis"] = np.select(
        conditions,
        labels,
        default="not_flagged",
    )

    return data


def main() -> None:
    """Load baseline results, identify candidates and save diagnostics."""

    if not WEEKLY_ANALYSIS_FILE.is_file():
        raise FileNotFoundError(
            "Weekly analysis not found. Run "
            "'python -m src.aggregate' and then "
            "'python -m src.baselines' first."
        )

    analysis = pd.read_csv(WEEKLY_ANALYSIS_FILE)

    flagged = add_candidate_flags(analysis)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flagged.to_csv(
        FLAGGED_ANALYSIS_FILE,
        index=False,
        float_format="%.10f",
    )

    candidates = flagged["candidate_anomaly"]

    print(f"Weekly records evaluated: {len(flagged):,}")
    print(f"Anomaly candidates: {int(candidates.sum())}")
    print(
        "Candidates with historical data: "
        f"{int((candidates & flagged['own_history_baseline'].notna()).sum())}"
    )
    print(
        "Candidates without historical data: "
        f"{int((candidates & flagged['own_history_baseline'].isna()).sum())}"
    )
    print(
        "Own-history threshold: "
        f"{settings.own_history_threshold_pct:.1f}%"
    )
    print(
        "Peer threshold: "
        f"{settings.peer_threshold_pct:.1f}%"
    )
    print(f"Saved candidate analysis to: {FLAGGED_ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
