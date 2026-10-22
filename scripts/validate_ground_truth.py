
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = ROOT / "data/evaluation/queries_v3.json"
CHUNKS_PATH = ROOT / "data/processed/retrieval_index_v3/chunks.json"

errors = []
warnings = []

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_chunks(data):
    # chunks.json may be a list or a dict containing "chunks".
    if isinstance(data, dict):
        if isinstance(data.get("chunks"), list):
            data = data["chunks"]
        else:
            raise ValueError(
                "Unsupported chunks.json structure. "
                "Expected a list or a dictionary with a 'chunks' list."
            )

    if not isinstance(data, list):
        raise ValueError("Chunk data must be a list.")

    chunk_ids = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            chunk_ids.append(item)
        elif isinstance(item, dict):
            chunk_id = item.get("chunk_id") or item.get("id")
            if not chunk_id:
                errors.append(f"Chunk at index {i} has no chunk_id/id.")
            else:
                chunk_ids.append(str(chunk_id))
        else:
            errors.append(
                f"Chunk at index {i} has unsupported type: "
                f"{type(item).__name__}"
            )

    return chunk_ids

def normalize_queries(data):
    # Support common JSON layouts without changing the source file.
    if isinstance(data, dict):
        if isinstance(data.get("queries"), list):
            data = data["queries"]
        elif isinstance(data.get("data"), list):
            data = data["data"]
        else:
            raise ValueError(
                "Unsupported queries_v3.json structure. "
                "Expected a list or a dictionary with 'queries'/'data'."
            )

    if not isinstance(data, list):
        raise ValueError("Query data must be a list.")

    return data

def get_query_text(item):
    for key in ("query", "question", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""

def main():
    global errors, warnings

    queries_data = load_json(QUERIES_PATH)
    chunks_data = load_json(CHUNKS_PATH)

    queries = normalize_queries(queries_data)
    chunk_ids = normalize_chunks(chunks_data)
    valid_chunk_ids = set(chunk_ids)

    print("=" * 68)
    print("GROUND TRUTH VALIDATION")
    print("=" * 68)
    print(f"Queries file: {QUERIES_PATH}")
    print(f"Chunks file : {CHUNKS_PATH}")
    print(f"Query count : {len(queries)}")
    print(f"Chunk count : {len(chunk_ids)}")

    # Validate the chunk registry.
    chunk_counts = Counter(chunk_ids)
    duplicate_chunk_ids = sorted(
        chunk_id for chunk_id, count in chunk_counts.items() if count > 1
    )
    if duplicate_chunk_ids:
        errors.append(
            f"Duplicate chunk IDs in registry: {duplicate_chunk_ids[:10]}"
        )

    query_ids = []
    total_labels = 0
    queries_without_labels = []
    duplicate_labels = []
    unknown_labels = []
    old_schema_queries = []
    missing_text_queries = []
    query_records_with_bad_type = []

    for i, item in enumerate(queries):
        if not isinstance(item, dict):
            query_records_with_bad_type.append(i)
            errors.append(
                f"Query at index {i} is {type(item).__name__}, not an object."
            )
            continue

        query_id = item.get("query_id", item.get("id"))
        if query_id is None:
            warnings.append(f"Query at index {i} has no query_id/id.")
        else:
            query_ids.append(str(query_id))

        if not get_query_text(item):
            missing_text_queries.append(i)

        if "relevant_sources" in item:
            old_schema_queries.append(i)

        labels = item.get("relevant_chunks")
        if labels is None:
            errors.append(
                f"Query at index {i} has no 'relevant_chunks' field."
            )
            continue

        if not isinstance(labels, list):
            errors.append(
                f"Query at index {i}: relevant_chunks must be a list."
            )
            continue

        if len(labels) == 0:
            queries_without_labels.append(i)

        normalized_labels = []
        for label in labels:
            if isinstance(label, str):
                chunk_id = label.strip()
            elif isinstance(label, dict):
                chunk_id = label.get("chunk_id") or label.get("id")
                if chunk_id is not None:
                    chunk_id = str(chunk_id).strip()
            else:
                chunk_id = None

            if not chunk_id:
                errors.append(
                    f"Query at index {i} has an empty or invalid chunk label: "
                    f"{label!r}"
                )
                continue

            normalized_labels.append(chunk_id)
            total_labels += 1

            if chunk_id not in valid_chunk_ids:
                unknown_labels.append((i, chunk_id))

        label_counts = Counter(normalized_labels)
        for chunk_id, count in label_counts.items():
            if count > 1:
                duplicate_labels.append((i, chunk_id))

    duplicate_query_ids = sorted(
        query_id for query_id, count in Counter(query_ids).items()
        if count > 1
    )

    if duplicate_query_ids:
        errors.append(f"Duplicate query IDs: {duplicate_query_ids}")

    if missing_text_queries:
        errors.append(
            f"Queries with missing/empty question text (indices): "
            f"{missing_text_queries}"
        )

    if queries_without_labels:
        errors.append(
            f"Queries with empty relevant_chunks (indices): "
            f"{queries_without_labels}"
        )

    if unknown_labels:
        errors.append(
            f"Labels referring to unknown chunk IDs: {unknown_labels[:20]}"
        )

    if duplicate_labels:
        warnings.append(
            f"Duplicate relevant chunk labels within queries: "
            f"{duplicate_labels[:20]}"
        )

    if old_schema_queries:
        warnings.append(
            f"'relevant_sources' still exists in query records at indices: "
            f"{old_schema_queries[:20]}. Check whether this is intentional."
        )

    print(f"Total relevant-chunk labels: {total_labels}")
    print(f"Unique query IDs found     : {len(set(query_ids))}")
    print(f"Queries with empty labels  : {len(queries_without_labels)}")
    print(f"Unknown chunk references   : {len(unknown_labels)}")
    print(f"Duplicate chunk IDs        : {len(duplicate_chunk_ids)}")

    print("\n" + "-" * 68)
    print(f"ERRORS   : {len(errors)}")
    for n, message in enumerate(errors, 1):
        print(f"  ERROR {n}: {message}")

    print(f"\nWARNINGS : {len(warnings)}")
    for n, message in enumerate(warnings, 1):
        print(f"  WARNING {n}: {message}")

    print("\n" + "=" * 68)
    if errors:
        print("RESULT: VALIDATION FAILED — fix errors before evaluation.")
        return 1

    if warnings:
        print("RESULT: NO CRITICAL ERRORS, BUT REVIEW WARNINGS.")
    else:
        print("RESULT: PASSED — structural checks are clean.")

    print("No source data was modified.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nVALIDATION COULD NOT RUN: {type(exc).__name__}: {exc}")
        sys.exit(2)
