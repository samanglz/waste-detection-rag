import json
from pathlib import Path
from typing import Any


RELEVANCE_FIELDS = (
    "relevant_chunks",
    "relevant_sources",
    "relevant_source",
)


def load_evaluation_queries(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load and validate source-level or chunk-level evaluation queries."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation query file not found: {path}"
        )

    try:
        queries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid evaluation query JSON") from exc

    if not isinstance(queries, list) or not queries:
        raise ValueError(
            "Evaluation queries must be a non-empty list"
        )

    for position, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(
                f"Query at position {position} must be an object"
            )

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Query at position {position} has invalid query text"
            )

        present_fields = [
            field for field in RELEVANCE_FIELDS if field in item
        ]

        if len(present_fields) != 1:
            raise ValueError(
                f"Query at position {position} must contain exactly "
                "one relevance field: relevant_chunks, "
                "relevant_sources, or relevant_source"
            )

        field = present_fields[0]
        relevant_items = item[field]

        if field == "relevant_source":
            if (
                not isinstance(relevant_items, str)
                or not relevant_items.strip()
            ):
                raise ValueError(
                    f"Query at position {position} has invalid "
                    "relevant_source"
                )
            continue

        if not isinstance(relevant_items, list) or not relevant_items:
            raise ValueError(
                f"Query at position {position} must have a non-empty "
                f"{field} list"
            )

        if not all(
            isinstance(value, str) and value.strip()
            for value in relevant_items
        ):
            raise ValueError(
                f"Query at position {position} has invalid values "
                f"in {field}"
            )

        if len(set(relevant_items)) != len(relevant_items):
            raise ValueError(
                f"Query at position {position} has duplicate values "
                f"in {field}"
            )

    return queries
