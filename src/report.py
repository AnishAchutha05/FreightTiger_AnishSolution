"""Format submission results and record model usage."""

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    LOG_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_DIR,
    OUTPUT_FILE,
    TOKEN_LOG_FILE,
    settings,
)


def format_comparison(value, description: str) -> str:
    """Format a percentage deviation for the submission CSV."""

    if pd.isna(value):
        return f"Unavailable ({description})"

    return f"{float(value):+.1f}% vs {description}"


def build_output_row(
    row: dict,
    verification: dict,
    explanation: dict,
) -> dict:
    """Convert one investigated candidate into the required output schema."""

    justified = verification["justified"]

    return {
        "route": str(row["route"]),
        "week_of": pd.Timestamp(row["week_of"]).strftime("%Y-%m-%d"),
        "cost_per_tonne_km": round(
            float(row["cost_per_tonne_km"]), 2
        ),
        "vs_own_history": format_comparison(
            row.get("pct_vs_own"),
            "this route's past average",
        ),
        "vs_similar_routes": format_comparison(
            row.get("pct_vs_peers"),
            "similar-length routes this week",
        ),
        "flagged": "No (justified)" if justified else "Yes",
        "matched_note_id": (
            verification["matched_note_id"] if justified else ""
        ),
        "reason": explanation["reason"],
    }


def write_submission(rows: list[dict]) -> None:
    """Write the candidate-only CSV with the exact required columns."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    if not report.empty:
        report = report.sort_values(
            ["route", "week_of"],
            kind="mergesort",
        ).reset_index(drop=True)

        if report.duplicated(["route", "week_of"]).any():
            raise ValueError("Submission contains duplicate route-week rows.")

        allowed_verdicts = {"Yes", "No (justified)"}
        if not set(report["flagged"]).issubset(allowed_verdicts):
            raise ValueError("Submission contains an invalid verdict.")

        unexplained = report["flagged"].eq("Yes")
        if report.loc[unexplained, "matched_note_id"].fillna("").ne("").any():
            raise ValueError("Unexplained rows must not cite a matched note.")

        justified = report["flagged"].eq("No (justified)")
        if report.loc[justified, "matched_note_id"].fillna("").eq("").any():
            raise ValueError("Justified rows must cite a note ID.")

    report.to_csv(
        OUTPUT_FILE,
        index=False,
        columns=OUTPUT_COLUMNS,
        float_format="%.2f",
    )

    print(f"Submission rows: {len(report)}")
    print(f"Saved submission: {OUTPUT_FILE}")


def write_token_log(run_stats: dict) -> None:
    """Record local usage and a rough published-price comparison."""

    import json
    from datetime import datetime, timezone

    from src.config import LOG_DIR, TOKEN_LOG_FILE, settings

    input_tokens = int(run_stats.get("input_tokens", 0))
    output_tokens = int(run_stats.get("output_tokens", 0))

    # Published OpenAI gpt-6-luna standard rates, USD per million tokens.
    input_rate = 0.10
    output_rate = 0.50

    hosted_estimate = (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000

    log = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_model": settings.ollama_model_name,
        "runtime": "Ollama",
        "llm_calls": int(run_stats.get("llm_calls", 0)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "direct_local_api_cost_usd": 0,
        "hosted_comparison": {
            "model": "OpenAI gpt-6-luna",
            "input_usd_per_million_tokens": input_rate,
            "output_usd_per_million_tokens": output_rate,
            "estimated_cost_usd": round(hosted_estimate, 10),
            "price_checked": "2026-10-15",
            "source": "https://developers.openai.com/api/docs/pricing",
            "note": (
                "Rough comparison only. Tokenizers, model quality and "
                "hardware costs differ; this is not an actual hosted bill."
            ),
        },
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_LOG_FILE.write_text(
        json.dumps(log, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved token log: {TOKEN_LOG_FILE}")