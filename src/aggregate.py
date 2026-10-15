
"""Aggregate individual shipments into weekly route-level cost records.

Each output row represents one directed route, its supplied route type,
and one Monday-to-Sunday reporting week.
"""

import numpy as np
import pandas as pd

from src.config import (
    AGGREGATED_WEEKLY_FILE,
    OUTPUT_DIR,
    REQUIRED_SHIPMENT_COLUMNS,
    SHIPMENTS_FILE,
)


def validate_shipments(shipments: pd.DataFrame) -> pd.DataFrame:
    """Validate the shipment data before performing any calculations."""

    if shipments.empty:
        raise ValueError("The shipment dataset is empty.")

    # Make sure every column required by the assignment is present.
    missing_columns = [
        column
        for column in REQUIRED_SHIPMENT_COLUMNS
        if column not in shipments.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = shipments.copy()

    # Remove accidental whitespace and reject missing text fields.
    text_columns = [
        "shipment_id",
        "origin",
        "destination",
        "route_type",
        "material",
        "transporter",
    ]

    for column in text_columns:
        data[column] = data[column].astype("string").str.strip()

        missing = (
            data[column].isna()
            | data[column].fillna("").eq("")
        )

        if missing.any():
            raise ValueError(
                f"Missing or empty values in '{column}'."
            )

    # FreightTiger already supplies route types.
    # We validate them; we do not derive them from distance.
    allowed_route_types = {"Short", "Medium", "Long"}

    invalid_types = ~data["route_type"].isin(allowed_route_types)

    if invalid_types.any():
        found_types = data.loc[
            invalid_types, "route_type"
        ].unique().tolist()

        raise ValueError(
            f"Unexpected route types: {found_types}"
        )

    # Convert shipment dates into a consistent datetime format.
    data["shipment_date"] = pd.to_datetime(
        data["shipment_date"],
        errors="coerce",
    )

    if data["shipment_date"].isna().any():
        raise ValueError(
            "The dataset contains missing or invalid shipment dates."
        )

    # These fields must contain actual numeric values.
    numeric_columns = [
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if data[column].isna().any():
            raise ValueError(
                f"'{column}' contains missing or non-numeric values."
            )

        values = data[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"'{column}' contains infinite values."
            )

    # A shipment must carry a positive quantity over a positive distance.
    if (data["quantity_tonnes"] <= 0).any():
        raise ValueError(
            "'quantity_tonnes' must be greater than zero."
        )

    if (data["distance_km"] <= 0).any():
        raise ValueError(
            "'distance_km' must be greater than zero."
        )

    # Negative freight costs would distort the analysis.
    if (data["freight_cost_inr"] < 0).any():
        raise ValueError(
            "'freight_cost_inr' cannot be negative."
        )

    # Each shipment ID must identify a single record.
    duplicate_ids = data["shipment_id"].duplicated(keep=False)

    if duplicate_ids.any():
        examples = (
            data.loc[duplicate_ids, "shipment_id"]
            .unique()[:5]
            .tolist()
        )

        raise ValueError(
            f"Duplicate shipment IDs found: {examples}"
        )

    # The same directed route must always have one route type.
    route_type_counts = data.groupby(
        ["origin", "destination"]
    )["route_type"].nunique()

    inconsistent_routes = route_type_counts[
        route_type_counts > 1
    ]

    if not inconsistent_routes.empty:
        routes = [
            f"{origin}-{destination}"
            for origin, destination in inconsistent_routes.index
        ]

        raise ValueError(
            f"Routes have inconsistent route types: {routes}"
        )

    return data


def aggregate_shipments(shipments: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted freight costs for every observed route-week."""

    data = validate_shipments(shipments)

    # Weeks run Monday to Sunday.
    # For each shipment, find the Monday of its reporting week.
    data["week_of"] = (
        data["shipment_date"].dt.normalize()
        - pd.to_timedelta(
            data["shipment_date"].dt.weekday,
            unit="D",
        )
    )

    # Calculate the shipment's contribution to total tonne-kilometres.
    data["tonne_km"] = (
        data["quantity_tonnes"] * data["distance_km"]
    )

    if not np.isfinite(
        data["tonne_km"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Tonne-kilometre calculations are not finite."
        )

    # Group using the original origin and destination separately.
    # This preserves direction and uses the supplied route_type.
    weekly = (
        data.groupby(
            [
                "origin",
                "destination",
                "route_type",
                "week_of",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            shipment_count=("shipment_id", "size"),
            total_freight_cost_inr=(
                "freight_cost_inr",
                "sum",
            ),
            total_tonne_km=("tonne_km", "sum"),
        )
    )

    # Every observed group must have a positive tonne-kilometre total.
    if (weekly["total_tonne_km"] <= 0).any():
        raise ValueError(
            "A weekly group has no positive tonne-kilometre total."
        )

    # Check that aggregation has not produced invalid totals.
    total_columns = [
        "total_freight_cost_inr",
        "total_tonne_km",
    ]

    for column in total_columns:
        values = weekly[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Weekly totals in '{column}' are not finite."
            )

    # IMPORTANT:
    # Divide total weekly freight cost by total weekly tonne-km.
    # Do not average individual shipment cost-per-tonne-km ratios.
    weekly["cost_per_tonne_km"] = (
        weekly["total_freight_cost_inr"]
        / weekly["total_tonne_km"]
    )

    if not np.isfinite(
        weekly["cost_per_tonne_km"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Weekly cost calculations are not finite."
        )

    # Create a readable route label for the report.
    # The original fields were already used for grouping.
    weekly["route"] = (
        weekly["origin"] + "-" + weekly["destination"]
    )

    # Keep the aggregation output useful for auditing and later stages.
    # Historical comparisons and anomaly flags belong in separate modules.
    weekly = weekly[
        [
            "route",
            "route_type",
            "week_of",
            "shipment_count",
            "total_freight_cost_inr",
            "total_tonne_km",
            "cost_per_tonne_km",
        ]
    ]

    # Stable sorting keeps output ordering predictable across runs.
    weekly = weekly.sort_values(
        ["route", "week_of"],
        kind="mergesort",
    ).reset_index(drop=True)

    return weekly


def main() -> None:
    """Load the shipment CSV, aggregate it and save the result."""

    if not SHIPMENTS_FILE.is_file():
        raise FileNotFoundError(
            f"Shipment file not found: {SHIPMENTS_FILE}"
        )

    shipments = pd.read_csv(SHIPMENTS_FILE)

    weekly = aggregate_shipments(shipments)

    # Create the output directory if it does not exist yet.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the intermediate aggregation, not the final submission CSV.
    weekly.to_csv(
        AGGREGATED_WEEKLY_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    weeks_per_route = weekly.groupby(
        "route"
    )["week_of"].nunique()

    print(f"Input shipments: {len(shipments):,}")
    print(f"Directed routes: {weekly['route'].nunique()}")
    print(f"Distinct reporting weeks: {weekly['week_of'].nunique()}")
    print(f"Weekly route records: {len(weekly):,}")

    if not weeks_per_route.empty:
        print(
            "Weeks per route: "
            f"{weeks_per_route.min()} to {weeks_per_route.max()}"
        )

    print(
        f"Saved weekly aggregation to: {AGGREGATED_WEEKLY_FILE}"
    )


if __name__ == "__main__":
    main()
