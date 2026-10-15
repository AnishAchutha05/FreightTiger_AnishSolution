"""Retrieve potentially relevant context notes from ChromaDB."""

from datetime import date
from functools import lru_cache

from src.config import settings
from src.notes import open_vector_store


@lru_cache(maxsize=1)
def get_vector_store():
    """Load the persistent vector store once per process."""
    return open_vector_store()


def retrieve_candidates(
    route: str,
    week_of: str,
    current_cost: float | None = None,
    pct_vs_own: float | None = None,
    pct_vs_peers: float | None = None,
    k: int | None = None,
) -> list[dict]:
    """Retrieve candidate notes; verification happens separately."""

    if not route or not route.strip():
        raise ValueError("Route cannot be empty.")

    try:
        week = date.fromisoformat(str(week_of)[:10])
    except ValueError as exc:
        raise ValueError("week_of must be a valid YYYY-MM-DD date.") from exc

    if week.weekday() != 0:
        raise ValueError("week_of must be a Monday.")

    limit = (
        settings.retrieval_candidate_count
        if k is None
        else k
    )

    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("k must be an integer.")

    if not 1 <= limit <= 10:
        raise ValueError("k must be between 1 and 10.")

    query_parts = [
        "Find documented events that could explain increased freight "
        "transport costs.",
        f"Affected directed route: {route.strip()}.",
        f"Reporting week beginning {week.isoformat()}.",
        "Look for flooding, detours, diesel-price increases, toll changes, "
        "festival surcharges, or other events that increase transport costs.",
    ]

    if current_cost is not None:
        query_parts.append(
            f"Observed cost per tonne-kilometre: {current_cost:.6f}."
        )

    if pct_vs_own is not None:
        query_parts.append(
            f"Deviation from own history: {pct_vs_own:+.2f}%."
        )

    if pct_vs_peers is not None:
        query_parts.append(
            f"Deviation from similar routes: {pct_vs_peers:+.2f}%."
        )

    query = " ".join(query_parts)

    results = get_vector_store().similarity_search_with_score(
        query,
        k=limit,
    )

    candidates = []

    for document, distance in results:
        note_id = document.metadata.get("note_id")

        if not note_id:
            continue

        candidates.append(
            {
                "note_id": str(note_id),
                "page_content": document.page_content,
                "metadata": document.metadata,
                "distance": float(distance),
            }
        )

    # Smaller Chroma distances indicate closer matches.
    # Use note_id to resolve equal-distance results deterministically.
    candidates.sort(
        key=lambda candidate: (
            candidate["distance"],
            candidate["note_id"],
        )
    )

    return candidates


def main() -> None:
    """Run a quick retrieval smoke test."""

    candidates = retrieve_candidates(
        route="Chennai-Bangalore",
        week_of="2025-03-03",
        current_cost=3.5,
        pct_vs_own=20.0,
        pct_vs_peers=18.0,
    )

    for candidate in candidates:
        print(
            f"{candidate['note_id']} | "
            f"distance={candidate['distance']:.4f}"
        )

    if any(candidate["note_id"] == "N001" for candidate in candidates):
        print("Flooding note retrieved.")
    else:
        print("Flooding note not in returned candidates.")


if __name__ == "__main__":
    main()