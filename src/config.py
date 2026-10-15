"""Central configuration for the FreightTiger cost monitor."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings, overridable through environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FT_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cost analysis
    history_window_weeks: int = Field(default=8, ge=1)
    own_history_threshold_pct: float = Field(default=15.0, ge=0)
    peer_threshold_pct: float = Field(default=15.0, ge=0)

    # Vector database and embeddings
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    chroma_collection_name: str = "freight_context_notes"
    retrieval_candidate_count: int = Field(default=10, ge=1, le=10)

    # Local LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "qwen2.5:7b"
    ollama_temperature: float = Field(default=0.0, ge=0, le=2)
    ollama_seed: int = 42


settings = Settings()


# Project directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
REPRO_DIR = PROJECT_ROOT / "repro"
CHROMA_DIR = settings.chroma_dir


# Input files
SHIPMENTS_FILE = RAW_DATA_DIR / "shipment_records.csv"
CONTEXT_NOTES_FILE = RAW_DATA_DIR / "context_notes.csv"
SAMPLE_OUTPUT_FILE = RAW_DATA_DIR / "sample_output_format_v2.csv"
STRUCTURED_NOTES_FILE = DATA_DIR / "notes_structured.json"


# Output files
FLAGGED_ANALYSIS_FILE = OUTPUT_DIR / "flagged_analysis.csv"
OUTPUT_FILE = OUTPUT_DIR / "output.csv"
WEEKLY_ANALYSIS_FILE = OUTPUT_DIR / "weekly_analysis.csv"
AGGREGATED_WEEKLY_FILE = OUTPUT_DIR / "weekly_aggregation.csv"
TOKEN_LOG_FILE = LOG_DIR / "token_log.json"


# Required shipment fields
REQUIRED_SHIPMENT_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
]


# Required context-note fields
REQUIRED_NOTE_COLUMNS = [
    "note_id",
    "date",
    "applies_to",
    "note",
]


# Exact output contract from the assignment
OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

COST_DECIMAL_PLACES = 2
PERCENTAGE_DECIMAL_PLACES = 1