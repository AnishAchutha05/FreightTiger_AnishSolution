
"""Load, validate and index context notes using LangChain and ChromaDB."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHROMA_DIR,
    CONTEXT_NOTES_FILE,
    REQUIRED_NOTE_COLUMNS,
    STRUCTURED_NOTES_FILE,
    settings,
)


def parse_optional_date(value: object, field_name: str, note_id: str) -> str:
    """Validate an optional ISO date and return an empty string when absent."""

    if value is None:
        return ""

    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} for {note_id} must be a YYYY-MM-DD string or null."
        )

    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} for {note_id}: {value!r}"
        ) from exc


def load_note_records() -> list[dict]:
    """Combine the original notes with their manually reviewed metadata."""

    if not CONTEXT_NOTES_FILE.is_file():
        raise FileNotFoundError(
            f"Context notes not found: {CONTEXT_NOTES_FILE}"
        )

    if not STRUCTURED_NOTES_FILE.is_file():
        raise FileNotFoundError(
            f"Structured note metadata not found: {STRUCTURED_NOTES_FILE}"
        )

    # Preserve note IDs as strings and do not silently convert empty text
    # fields into the word "nan".
    source = pd.read_csv(
        CONTEXT_NOTES_FILE,
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_NOTE_COLUMNS
        if column not in source.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Context notes are missing columns: {missing_columns}"
        )

    for column in REQUIRED_NOTE_COLUMNS:
        source[column] = source[column].str.strip()

        if source[column].eq("").any():
            raise ValueError(
                f"Context notes contain empty values in '{column}'."
            )

    if source["note_id"].duplicated().any():
        raise ValueError("Context notes contain duplicate note IDs.")

    # Normalise source dates before comparing them with the sidecar.
    parsed_dates = pd.to_datetime(source["date"], errors="coerce")

    if parsed_dates.isna().any():
        raise ValueError("Context notes contain invalid dates.")

    source["date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    with STRUCTURED_NOTES_FILE.open(encoding="utf-8") as file:
        structured = json.load(file)

    if structured.get("schema_version") != 2:
        raise ValueError(
            "Expected notes_structured.json schema_version 2. "
            "Check that the updated metadata file was saved."
        )

    entries = structured.get("notes")

    if not isinstance(entries, list):
        raise ValueError("The structured metadata must contain a notes list.")

    # Every original note must have exactly one metadata entry.
    source_ids = set(source["note_id"])
    metadata_ids = [
        entry.get("note_id")
        for entry in entries
        if isinstance(entry, dict)
    ]

    if len(metadata_ids) != len(entries):
        raise ValueError("Every structured note must be a JSON object.")

    if len(set(metadata_ids)) != len(metadata_ids):
        raise ValueError("Structured metadata contains duplicate note IDs.")

    if set(metadata_ids) != source_ids:
        raise ValueError(
            "Source note IDs and structured metadata IDs do not match."
        )

    structured_by_id = {
        entry["note_id"]: entry
        for entry in entries
    }

    records = []

    for row in source.to_dict(orient="records"):
        note_id = row["note_id"]
        metadata = structured_by_id[note_id]

        # The sidecar must refer to the same note and reporting date.
        if metadata.get("note_date") != row["date"]:
            raise ValueError(
                f"Note date mismatch between CSV and metadata for {note_id}."
            )

        # Do not let the sidecar silently change the source's route scope.
        source_scope = row["applies_to"]

        if source_scope.casefold() == "all routes":
            expected_routes = ["ALL_ROUTES"]
        else:
            expected_routes = [source_scope]

        if metadata.get("applies_to_routes") != expected_routes:
            raise ValueError(
                f"Route scope mismatch for {note_id}: "
                f"source={source_scope!r}, "
                f"metadata={metadata.get('applies_to_routes')!r}"
            )

        effect = metadata.get("effect")
        rationale = metadata.get("rationale")
        window = metadata.get("verification_window")

        if not isinstance(effect, str) or not effect.strip():
            raise ValueError(f"Missing effect classification for {note_id}.")

        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"Missing rationale for {note_id}.")

        if not isinstance(window, dict):
            raise ValueError(
                f"Missing verification window for {note_id}."
            )

        # Validate the dates we will use for evidence verification.
        window_start = parse_optional_date(
            window.get("start_date"),
            "verification_window.start_date",
            note_id,
        )
        window_end = parse_optional_date(
            window.get("end_date"),
            "verification_window.end_date",
            note_id,
        )

        if not window_start or not window_end:
            raise ValueError(
                f"Verification window must have a start and end for {note_id}."
            )

        if window_start > window_end:
            raise ValueError(
                f"Verification window starts after it ends for {note_id}."
            )

        basis = window.get("basis")

        if not isinstance(basis, str) or not basis.strip():
            raise ValueError(
                f"Missing verification-window basis for {note_id}."
            )

        event_start = parse_optional_date(
            metadata.get("event_start_date"),
            "event_start_date",
            note_id,
        )
        event_end = parse_optional_date(
            metadata.get("event_end_date"),
            "event_end_date",
            note_id,
        )

        if event_start and event_end and event_start > event_end:
            raise ValueError(
                f"Event start date is after its end date for {note_id}."
            )

        excluded = metadata.get("exclude_from_dataset")

        if not isinstance(excluded, bool):
            raise ValueError(
                f"exclude_from_dataset must be true or false for {note_id}."
            )

        records.append(
            {
                "note_id": note_id,
                "note_date": row["date"],
                "applies_to": source_scope,
                "note_text": row["note"],
                "applies_to_routes": expected_routes,
                "event_start_date": event_start,
                "event_end_date": event_end,
                "verification_start_date": window_start,
                "verification_end_date": window_end,
                "verification_window_basis": basis,
                "effect": effect.strip(),
                "exclude_from_dataset": excluded,
                "rationale": rationale.strip(),
            }
        )

    return records


def create_embeddings() -> HuggingFaceEmbeddings:
    """Load the configured local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def open_vector_store(
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Open the persistent ChromaDB collection used by the RAG pipeline."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if embeddings is None:
        embeddings = create_embeddings()

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def make_document(record: dict) -> Document:
    """Turn one source note into a document suitable for semantic search."""

    # Include the route and date in the text being embedded so retrieval
    # can use both the event description and its context.
    page_content = (
        f"Note ID: {record['note_id']}\n"
        f"Report date: {record['note_date']}\n"
        f"Applies to: {record['applies_to']}\n"
        f"Context note: {record['note_text']}"
    )

    # Chroma metadata must use simple scalar values. Store the route list
    # as JSON text rather than passing a Python list or a null value.
    metadata = {
        "note_id": record["note_id"],
        "note_date": record["note_date"],
        "applies_to": record["applies_to"],
        "applies_to_routes_json": json.dumps(
            record["applies_to_routes"],
            ensure_ascii=False,
        ),
        "event_start_date": record["event_start_date"],
        "event_end_date": record["event_end_date"],
        "verification_start_date": record["verification_start_date"],
        "verification_end_date": record["verification_end_date"],
        "verification_window_basis": record["verification_window_basis"],
        "effect": record["effect"],
        "exclude_from_dataset": record["exclude_from_dataset"],
        "rationale": record["rationale"],
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def index_notes(records: list[dict]) -> dict[str, int]:
    """Index changed notes, retain unchanged embeddings and remove stale IDs."""

    if not records:
        raise ValueError("There are no context notes to index.")

    documents = [make_document(record) for record in records]
    note_ids = [record["note_id"] for record in records]

    if len(set(note_ids)) != len(note_ids):
        raise ValueError("Cannot index duplicate note IDs.")

    vector_store = open_vector_store()

    # Read the existing collection so unchanged notes do not have to be
    # embedded again every time the pipeline runs.
    existing = vector_store.get(
        include=["documents", "metadatas"],
    )

    existing_by_id = {}

    for index, existing_id in enumerate(existing["ids"]):
        existing_by_id[existing_id] = {
            "page_content": existing["documents"][index],
            "metadata": existing["metadatas"][index] or {},
        }

    changed_documents = []
    changed_ids = []

    for note_id, document in zip(note_ids, documents):
        old_document = existing_by_id.get(note_id)

        if (
            old_document is None
            or old_document["page_content"] != document.page_content
            or old_document["metadata"] != document.metadata
        ):
            changed_documents.append(document)
            changed_ids.append(note_id)

    # Stable note IDs let Chroma update existing notes instead of creating
    # a new copy each time this script runs.
    if changed_documents:
        vector_store.add_documents(
            documents=changed_documents,
            ids=changed_ids,
        )

    # Remove old IDs if a note was removed from the source files.
    stale_ids = sorted(set(existing_by_id) - set(note_ids))

    if stale_ids:
        vector_store.delete(ids=stale_ids)

    # Check that the persistent index contains exactly the current corpus.
    indexed = vector_store.get(include=["metadatas"])

    if set(indexed["ids"]) != set(note_ids):
        raise RuntimeError(
            "ChromaDB contents do not match the current source notes."
        )

    return {
        "total_notes": len(note_ids),
        "added_or_updated": len(changed_ids),
        "unchanged": len(note_ids) - len(changed_ids),
        "removed": len(stale_ids),
        "collection_count": len(indexed["ids"]),
    }


def main() -> None:
    """Validate the corpus and initialise the persistent vector index."""

    print("Validating context notes and structured metadata...")
    records = load_note_records()
    print(f"Validated {len(records)} context notes.")

    print(f"Loading embedding model: {settings.embedding_model_name}")
    print("The first run may download the model into the local cache.")

    result = index_notes(records)

    print("\nChromaDB indexing complete.")
    print(f"Notes in source: {result['total_notes']}")
    print(f"Added or updated: {result['added_or_updated']}")
    print(f"Unchanged: {result['unchanged']}")
    print(f"Removed stale notes: {result['removed']}")
    print(f"Documents in collection: {result['collection_count']}")
    print(f"Persistent database directory: {CHROMA_DIR}")
    print(f"Collection name: {settings.chroma_collection_name}")


if __name__ == "__main__":
    main()
