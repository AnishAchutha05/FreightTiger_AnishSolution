
import pandas as pd
import pytest

from src.aggregate import aggregate_shipments
from src.baselines import add_baselines
from src.flagging import add_candidate_flags
from src.verify import verify_candidates
from src.config import SHIPMENTS_FILE

def sample_shipments():
    """Two shipments on the same route and reporting week."""
    return pd.DataFrame([
        {
            "shipment_id": "S001",
            "origin": "Delhi",
            "destination": "Jaipur",
            "route_type": "Short",
            "material": "Steel",
            "quantity_tonnes": 1,
            "distance_km": 10,
            "freight_cost_inr": 100,
            "shipment_date": "2024-01-01",
            "transporter": "T1",
        },
        {
            "shipment_id": "S002",
            "origin": "Delhi",
            "destination": "Jaipur",
            "route_type": "Short",
            "material": "Steel",
            "quantity_tonnes": 10,
            "distance_km": 10,
            "freight_cost_inr": 100,
            "shipment_date": "2024-01-07",
            "transporter": "T2",
        },
    ])


def test_aggregation_uses_ratio_of_sums():
    result = aggregate_shipments(sample_shipments())

    assert len(result) == 1
    row = result.iloc[0]

    assert row["route"] == "Delhi-Jaipur"
    assert row["route_type"] == "Short"
    assert row["week_of"] == pd.Timestamp("2024-01-01")
    assert row["shipment_count"] == 2
    assert row["total_freight_cost_inr"] == 200
    assert row["total_tonne_km"] == 110
    assert row["cost_per_tonne_km"] == pytest.approx(200 / 110)


def test_duplicate_shipment_ids_are_rejected():
    shipments = sample_shipments()
    shipments.loc[1, "shipment_id"] = "S001"

    with pytest.raises(ValueError, match="Duplicate shipment IDs"):
        aggregate_shipments(shipments)


def test_inconsistent_route_types_are_rejected():
    shipments = sample_shipments()
    shipments.loc[1, "route_type"] = "Medium"

    with pytest.raises(ValueError, match="inconsistent route types"):
        aggregate_shipments(shipments)


def test_baselines_exclude_current_week_and_other_route():
    weeks = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    rows = []

    for index, week in enumerate(weeks):
        rows.extend([
            {
                "route": "Route-A",
                "route_type": "Short",
                "week_of": week,
                "cost_per_tonne_km": float(index + 1),
            },
            {
                "route": "Route-B",
                "route_type": "Short",
                "week_of": week,
                "cost_per_tonne_km": float(index + 2),
            },
            {
                "route": "Route-C",
                "route_type": "Long",
                "week_of": week,
                "cost_per_tonne_km": 100.0 + index,
            },
        ])

    result = add_baselines(pd.DataFrame(rows))
    route_a = result[result["route"] == "Route-A"].reset_index(drop=True)

    assert pd.isna(route_a.loc[0, "own_history_baseline"])
    assert route_a.loc[0, "peer_baseline"] == pytest.approx(2.0)

    # Week 9 uses only Route-A's first eight costs: 1 through 8.
    assert route_a.loc[8, "own_history_baseline"] == pytest.approx(4.5)

    # Week 10 uses the costs from weeks 2 through 9.
    assert route_a.loc[9, "own_history_baseline"] == pytest.approx(5.5)

    # The peer baseline uses Route-B that week, not Route-C (Long).
    assert route_a.loc[8, "peer_baseline"] == pytest.approx(10.0)


def test_flagging_requires_rising_cost_when_history_exists():
    analysis = pd.DataFrame([
        {
            "route": "Rising",
            "week_of": "2024-01-08",
            "cost_per_tonne_km": 12.0,
            "own_history_baseline": 10.0,
            "pct_vs_own": 20.0,
            "peer_baseline": 10.0,
            "pct_vs_peers": 20.0,
        },
        {
            "route": "Not-Rising",
            "week_of": "2024-01-08",
            "cost_per_tonne_km": 10.0,
            "own_history_baseline": 12.0,
            "pct_vs_own": -16.67,
            "peer_baseline": 5.0,
            "pct_vs_peers": 100.0,
        },
        {
            "route": "First-Week",
            "week_of": "2024-01-01",
            "cost_per_tonne_km": 12.0,
            "own_history_baseline": float("nan"),
            "pct_vs_own": float("nan"),
            "peer_baseline": 10.0,
            "pct_vs_peers": 20.0,
        },
    ])

    result = add_candidate_flags(analysis).set_index("route")

    assert bool(result.loc["Rising", "candidate_anomaly"])
    assert not bool(result.loc["Not-Rising", "candidate_anomaly"])
    assert bool(result.loc["First-Week", "candidate_anomaly"])


def test_verifier_accepts_note_for_supported_route_and_week():
    candidate = {
        "note_id": "N001",
        "distance": 0.2,
        "page_content": "Flooding caused detours and higher trip costs.",
        "metadata": {
            "note_id": "N001",
            "exclude_from_dataset": False,
            "effect": "positive_cost_increase",
            "applies_to_routes_json": '["Chennai-Bangalore"]',
            "verification_start_date": "2025-02-24",
            "verification_end_date": "2025-03-08",
        },
    }

    result = verify_candidates(
        "Chennai-Bangalore",
        "2025-03-03",
        [candidate],
    )

    assert result["justified"]
    assert result["matched_note_id"] == "N001"


def test_verifier_rejects_note_outside_its_window():
    candidate = {
        "note_id": "N001",
        "distance": 0.2,
        "metadata": {
            "note_id": "N001",
            "exclude_from_dataset": False,
            "effect": "positive_cost_increase",
            "applies_to_routes_json": '["Chennai-Bangalore"]',
            "verification_start_date": "2025-02-24",
            "verification_end_date": "2025-03-08",
        },
    }

    result = verify_candidates(
        "Chennai-Bangalore",
        "2025-03-10",
        [candidate],
    )

    assert not result["justified"]
    assert result["matched_note_id"] == ""


def test_verifier_rejects_wrong_route_and_non_positive_evidence():
    wrong_route = {
        "note_id": "N001",
        "distance": 0.1,
        "metadata": {
            "note_id": "N001",
            "exclude_from_dataset": False,
            "effect": "positive_cost_increase",
            "applies_to_routes_json": '["Chennai-Bangalore"]',
            "verification_start_date": "2025-02-24",
            "verification_end_date": "2025-03-08",
        },
    }

    no_increase = {
        "note_id": "N006",
        "distance": 0.1,
        "metadata": {
            "note_id": "N006",
            "exclude_from_dataset": False,
            "effect": "stable_conditions_no_increase_evidence",
            "applies_to_routes_json": '["ALL_ROUTES"]',
            "verification_start_date": "2025-09-22",
            "verification_end_date": "2025-09-30",
        },
    }

    result = verify_candidates(
        "Mumbai-Pune",
        "2025-09-22",
        [wrong_route, no_increase],
    )

    assert not result["justified"]
    assert result["matched_note_id"] == ""

def test_baseline_uses_eight_calendar_weeks_when_a_week_is_missing():
    weeks = pd.to_datetime([
        "2024-01-01",
        "2024-01-08",
        "2024-01-15",
        "2024-01-22",
        "2024-01-29",
        "2024-02-05",
        # Week beginning 2024-02-12 is intentionally missing.
        "2024-02-19",
        "2024-02-26",
        "2024-03-04",
    ])

    costs = [1, 2, 3, 4, 5, 6, 8, 9, 100]

    weekly = pd.DataFrame({
        "route": ["Route-A"] * len(weeks),
        "route_type": ["Short"] * len(weeks),
        "week_of": weeks,
        "cost_per_tonne_km": costs,
    })

    result = add_baselines(weekly)
    current_week = result.loc[
        result["week_of"] == pd.Timestamp("2024-03-04")
    ].iloc[0]

    # The eight calendar weeks start on Jan 8 through Feb 26.
    # Jan 1 is outside the window; the missing week is not filled.
    expected = (2 + 3 + 4 + 5 + 6 + 8 + 9) / 7

    assert current_week["own_history_baseline"] == pytest.approx(expected)

@pytest.mark.parametrize(
    "route, week, expected_cost, expected_own_pct, expected_peer_pct",
    [
        ("Delhi-Jaipur", "2024-11-11", 4.17, 35.5, 21.0),
        ("Ahmedabad-Mumbai", "2025-01-20", 3.29, 29.5, 22.5),
        ("Mumbai-Pune", "2025-09-15", 3.98, 9.2, 23.6),
    ],
)
def test_case_study_reference_baselines(
    route,
    week,
    expected_cost,
    expected_own_pct,
    expected_peer_pct,
):
    shipments = pd.read_csv(SHIPMENTS_FILE)
    weekly = aggregate_shipments(shipments)
    analysis = add_baselines(weekly)

    target_week = pd.Timestamp(week)

    matches = analysis[
        (analysis["route"] == route)
        & (analysis["week_of"] == target_week)
    ]

    assert len(matches) == 1
    row = matches.iloc[0]

    assert round(float(row["cost_per_tonne_km"]), 2) == expected_cost
    assert round(float(row["pct_vs_own"]), 1) == expected_own_pct
    assert round(float(row["pct_vs_peers"]), 1) == expected_peer_pct