"""Verify whether retrieved notes can justify a freight cost increase."""

import json
from datetime import date, timedelta
import math


POSITIVE_EFFECT = "positive_cost_increase"
ALL_ROUTES = "ALL_ROUTES"


def verify_candidates(
    route: str,
    week_of: str,
    candidates: list[dict],
) -> dict:
    """Accept only notes that support the route, week and cost increase."""

    if not route.strip():
        raise ValueError("Route cannot be empty.")

    try:
        week_start = date.fromisoformat(str(week_of)[:10])
    except ValueError as exc:
        raise ValueError("week_of must be a valid YYYY-MM-DD date.") from exc

    if week_start.weekday() != 0:
        raise ValueError("week_of must be a Monday.")

    week_end = week_start + timedelta(days=6)
    verified = []
    rejected = []

    for candidate in candidates:
        note_id = str(candidate.get("note_id", ""))
        metadata = candidate.get("metadata", {})
        reasons = []

        if not note_id or metadata.get("note_id") != note_id:
            reasons.append("missing_or_mismatched_note_id")

        if metadata.get("exclude_from_dataset") is True:
            reasons.append("explicitly_excluded_from_dataset")

        if metadata.get("effect") != POSITIVE_EFFECT:
            reasons.append("does_not_support_cost_increase")

        # Check whether the note applies to this directed route.
        try:
            applicable_routes = json.loads(
                metadata.get("applies_to_routes_json", "[]")
            )
        except (TypeError, json.JSONDecodeError):
            applicable_routes = []

        if not isinstance(applicable_routes, list):
            applicable_routes = []

        if route not in applicable_routes and ALL_ROUTES not in applicable_routes:
            reasons.append("wrong_route")

        # The supported verification window must overlap this week.
        try:
            window_start = date.fromisoformat(
                metadata["verification_start_date"]
            )
            window_end = date.fromisoformat(
                metadata["verification_end_date"]
            )

            if window_start > window_end:
                reasons.append("invalid_date_window")
            elif window_start > week_end or window_end < week_start:
                reasons.append("outside_verification_window")

        except (KeyError, TypeError, ValueError):
            reasons.append("missing_or_invalid_date_window")

        try:
            distance = float(candidate["distance"])
            if not math.isfinite(distance):
                reasons.append("invalid_similarity_distance")
        except (KeyError, TypeError, ValueError):
            distance = math.inf
            reasons.append("invalid_similarity_distance")

        if reasons:
            rejected.append({
                "note_id": note_id,
                "rejection_reasons": reasons,
            })
            continue

        verified.append({
            **candidate,
            "distance": distance,
        })

    # Lowest Chroma distance wins; note ID breaks ties deterministically.
    verified.sort(
        key=lambda item: (item["distance"], item["note_id"])
    )

    if verified:
        chosen = verified[0]
        return {
            "justified": True,
            "matched_note_id": chosen["note_id"],
            "verified_note": chosen,
            "rejections": rejected,
        }

    return {
        "justified": False,
        "matched_note_id": "",
        "verified_note": None,
        "rejections": rejected,
    }


def main() -> None:
    """Test evidence verification against key assignment guardrails."""

    from src.retrieve import retrieve_candidates

    tests = [
        ("Chennai-Bangalore", "2025-03-03", True, "N001"),
        ("Chennai-Bangalore", "2025-03-10", False, ""),
        ("Mumbai-Pune", "2025-09-15", False, ""),
    ]

    for route, week, expected, expected_id in tests:
        candidates = retrieve_candidates(
            route=route,
            week_of=week,
            k=10,
        )
        result = verify_candidates(route, week, candidates)

        assert result["justified"] is expected, (
            f"Unexpected verdict for {route}, {week}: {result}"
        )
        assert result["matched_note_id"] == expected_id, (
            f"Unexpected citation for {route}, {week}: {result}"
        )

        print(
            f"{route} | {week} | "
            f"justified={result['justified']} | "
            f"note={result['matched_note_id'] or 'none'}"
        )

    print("All verification checks passed.")


if __name__ == "__main__":
    main()