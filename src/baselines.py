
"""Compare weekly route costs against historical and peer baselines."""

import numpy as np
import pandas as pd

from src.config import (
    AGGREGATED_WEEKLY_FILE,
    OUTPUT_DIR,
    WEEKLY_ANALYSIS_FILE,
    settings,
)


def add_baselines(
    weekly: pd.DataFrame,
    history_window_weeks: int = settings.history_window_weeks,
) -> pd.DataFrame:
    """Add historical and same-week peer comparisons to weekly costs."""

    if history_window_weeks < 1:
        raise ValueError("The history window must be at least one week.")

    required_columns = [
        "route",
        "route_type",
        "week_of",
        "cost_per_tonne_km",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if weekly.empty:
        raise ValueError("The weekly aggregation contains no records.")

    data = weekly.copy()

    # Route names and types must be present.
    for column in ["route", "route_type"]:
        data[column] = data[column].astype("string").str.strip()

        missing = data[column].isna() | data[column].fillna("").eq("")

        if missing.any():
            raise ValueError(
                f"Missing or empty values in '{column}'."
            )

    # Make sure week dates and costs have the expected types.
    data["week_of"] = pd.to_datetime(
        data["week_of"],
        errors="coerce",
    )

    if data["week_of"].isna().any():
        raise ValueError("Invalid or missing week_of dates.")

    data["cost_per_tonne_km"] = pd.to_numeric(
        data["cost_per_tonne_km"],
        errors="coerce",
    )

    if data["cost_per_tonne_km"].isna().any():
        raise ValueError("Weekly costs contain missing or invalid values.")

    costs = data["cost_per_tonne_km"].to_numpy(dtype=float)

    if not np.isfinite(costs).all():
        raise ValueError("Weekly costs contain infinite values.")

    if (costs < 0).any():
        raise ValueError("Weekly costs cannot be negative.")

    # Every route must have only one supplied route type.
    route_type_counts = data.groupby("route")["route_type"].nunique()

    if (route_type_counts > 1).any():
        raise ValueError("A route has more than one route type.")

    # There must be only one record per route and reporting week.
    duplicate_rows = data.duplicated(
        ["route", "week_of"],
        keep=False,
    )

    if duplicate_rows.any():
        examples = data.loc[
            duplicate_rows, ["route", "week_of"]
        ].head()

        raise ValueError(
            "Duplicate route-week records found:\n"
            f"{examples.to_string(index=False)}"
        )

    # Put records in chronological order before calculating history.
    data = data.sort_values(
        ["route", "week_of"],
        kind="mergesort",
    ).reset_index(drop=True)

    # BASELINE 1: OWN HISTORY
    #
    # Shift by one week first, so the current week's cost is excluded.
    # Then average up to the previous eight available weekly values.
    data["own_history_baseline"] = (
        data.groupby("route", sort=False)["cost_per_tonne_km"]
        .transform(
            lambda route_costs: route_costs.shift(1).rolling(
                window=history_window_weeks,
                min_periods=1,
            ).mean()
        )
    )

    # BASELINE 2: SIMILAR ROUTES
    #
    # Calculate the total cost and route count for each route type
    # in each week. Subtract the current route to exclude itself.
    peer_groups = data.groupby(
        ["week_of", "route_type"],
        sort=False,
    )["cost_per_tonne_km"]

    total_peer_group_cost = peer_groups.transform("sum")
    peer_group_count = peer_groups.transform("count")

    other_route_count = peer_group_count - 1
    other_route_cost = total_peer_group_cost - data["cost_per_tonne_km"]

    data["peer_baseline"] = (
        other_route_cost / other_route_count
    ).where(other_route_count > 0)

    # PERCENTAGE DEVIATIONS
    #
    # A zero or unavailable baseline cannot safely be used as a divisor.
    own_baseline_is_positive = data["own_history_baseline"] > 0
    peer_baseline_is_positive = data["peer_baseline"] > 0

    data["pct_vs_own"] = (
        (
            data["cost_per_tonne_km"]
            - data["own_history_baseline"]
        )
        / data["own_history_baseline"]
        * 100
    ).where(own_baseline_is_positive)

    data["pct_vs_peers"] = (
        (
            data["cost_per_tonne_km"]
            - data["peer_baseline"]
        )
        / data["peer_baseline"]
        * 100
    ).where(peer_baseline_is_positive)

    return data


def main() -> None:
    """Load aggregated costs, calculate baselines and save diagnostics."""

    if not AGGREGATED_WEEKLY_FILE.is_file():
        raise FileNotFoundError(
            "Weekly aggregation not found. Run "
            "'python -m src.aggregate' first."
        )

    weekly = pd.read_csv(AGGREGATED_WEEKLY_FILE)

    analysis = add_baselines(
        weekly,
        history_window_weeks=settings.history_window_weeks,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis.to_csv(
        WEEKLY_ANALYSIS_FILE,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10f",
    )

    print(f"Weekly records analysed: {len(analysis):,}")
    print(
        "Records without a historical baseline: "
        f"{analysis['own_history_baseline'].isna().sum()}"
    )
    print(
        "Records without a peer baseline: "
        f"{analysis['peer_baseline'].isna().sum()}"
    )
    print(
        "Records without a valid historical percentage: "
        f"{analysis['pct_vs_own'].isna().sum()}"
    )
    print(
        "Records without a valid peer percentage: "
        f"{analysis['pct_vs_peers'].isna().sum()}"
    )
    print(f"Saved analysis to: {WEEKLY_ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
