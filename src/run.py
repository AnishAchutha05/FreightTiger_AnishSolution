
"""Run the complete freight cost monitoring pipeline."""

import pandas as pd

from src.aggregate import aggregate_shipments
from src.baselines import add_baselines
from src.config import (
    AGGREGATED_WEEKLY_FILE,
    FLAGGED_ANALYSIS_FILE,
    OUTPUT_DIR,
    SHIPMENTS_FILE,
    WEEKLY_ANALYSIS_FILE,
    settings,
)
from src.explain import explain_record
from src.flagging import add_candidate_flags
from src.notes import index_notes, load_note_records
from src.report import build_output_row, write_submission, write_token_log
from src.retrieve import retrieve_candidates
from src.verify import verify_candidates


def main() -> None:
    """Calculate costs, investigate anomalies and write the reports."""

    if not SHIPMENTS_FILE.is_file():
        raise FileNotFoundError(f"Shipment file not found: {SHIPMENTS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate individual shipments into weekly route costs.
    shipments = pd.read_csv(SHIPMENTS_FILE)
    weekly = aggregate_shipments(shipments)

    weekly.to_csv(
        AGGREGATED_WEEKLY_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    # 2. Calculate historical and peer baselines.
    analysis = add_baselines(
        weekly,
        history_window_weeks=settings.history_window_weeks,
    )

    analysis.to_csv(
        WEEKLY_ANALYSIS_FILE,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10f",
    )

    # 3. Identify candidates using deterministic numerical rules.
    flagged = add_candidate_flags(analysis)

    flagged.to_csv(
        FLAGGED_ANALYSIS_FILE,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10f",
    )

    candidates = flagged[flagged["candidate_anomaly"]]

    # 4. Synchronise the persistent vector index with the reviewed notes.
    index_result = index_notes(load_note_records())
    print(f"Context notes indexed: {index_result['collection_count']}")

    output_rows = []
    run_stats = {
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    # 5. Retrieve evidence, verify it, and explain each candidate.
    for _, candidate in candidates.iterrows():
        row = candidate.to_dict()
        week = pd.Timestamp(row["week_of"]).strftime("%Y-%m-%d")

        retrieved = retrieve_candidates(
            route=str(row["route"]),
            week_of=week,
            current_cost=float(row["cost_per_tonne_km"]),
            pct_vs_own=(
                float(row["pct_vs_own"])
                if pd.notna(row["pct_vs_own"])
                else None
            ),
            pct_vs_peers=(
                float(row["pct_vs_peers"])
                if pd.notna(row["pct_vs_peers"])
                else None
            ),
            k=settings.retrieval_candidate_count,
        )

        verification = verify_candidates(
            route=str(row["route"]),
            week_of=week,
            candidates=retrieved,
        )

        explanation = explain_record(row, verification)

        run_stats["llm_calls"] += explanation["llm_calls"]
        run_stats["input_tokens"] += explanation["input_tokens"]
        run_stats["output_tokens"] += explanation["output_tokens"]

        output_rows.append(
            build_output_row(row, verification, explanation)
        )

    # 6. Save the required candidate-only submission CSV and usage log.
    write_submission(output_rows)
    write_token_log(run_stats)

    print("\nPipeline completed.")
    print(f"Shipments processed: {len(shipments):,}")
    print(f"Weekly records: {len(weekly):,}")
    print(f"Anomaly candidates: {len(candidates)}")
    print(f"Justified candidates: {sum(r['flagged'] == 'No (justified)' for r in output_rows)}")
    print(f"Unexplained candidates: {sum(r['flagged'] == 'Yes' for r in output_rows)}")
    print(f"LLM calls: {run_stats['llm_calls']}")
    print(f"Input tokens: {run_stats['input_tokens']}")
    print(f"Output tokens: {run_stats['output_tokens']}")


if __name__ == "__main__":
    main()
