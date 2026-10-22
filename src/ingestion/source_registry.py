"""Registry utilities for traceable corpus sources."""

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REGISTRY_SCHEMA_VERSION = 1

REQUIRED_FIELDS = {
    "document_id",
    "title",
    "publisher",
    "source_type",
    "url",
    "language",
    "topics",
    "status",
    "license_status",
}

VALID_STATUSES = {
    "candidate",
    "discovered",
    "downloaded",
    "validated",
    "rejected",
}


def validate_source_record(record: dict[str, Any]) -> None:
    """Validate one corpus source record."""

    if not isinstance(record, dict):
        raise ValueError("Source record must be a dictionary")

    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(
            f"Source record is missing fields: {sorted(missing)}"
        )

    for field in (
        "document_id",
        "title",
        "publisher",
        "source_type",
        "url",
        "language",
        "status",
        "license_status",
    ):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    parsed_url = urlparse(record["url"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("url must be a valid HTTP(S) URL")

    if record["language"].lower() != "en":
        raise ValueError("This corpus registry currently requires English sources")

    if record["status"] not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {record['status']}. "
            f"Expected one of {sorted(VALID_STATUSES)}"
        )

    if not isinstance(record["topics"], list) or not record["topics"]:
        raise ValueError("topics must be a non-empty list")

    if not all(
        isinstance(topic, str) and topic.strip()
        for topic in record["topics"]
    ):
        raise ValueError("Every topic must be a non-empty string")

    for optional_field in ("doi", "published_date", "accessed_date"):
        value = record.get(optional_field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{optional_field} must be a string or null")

    content_hash = record.get("content_sha256")
    if content_hash is not None:
        if (
            not isinstance(content_hash, str)
            or len(content_hash) != 64
            or any(c not in "0123456789abcdef" for c in content_hash.lower())
        ):
            raise ValueError("content_sha256 must be a SHA-256 hex digest")

    local_path = record.get("local_path")
    if local_path is not None and not isinstance(local_path, str):
        raise ValueError("local_path must be a string or null")


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load and validate the central corpus registry."""

    registry_path = Path(path)

    if not registry_path.is_file():
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "sources": [],
        }

    try:
        registry = json.loads(
            registry_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Invalid registry JSON: {registry_path}"
        ) from exc

    if not isinstance(registry, dict):
        raise ValueError("Registry must be a JSON object")

    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported registry schema version")

    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Registry sources must be a list")

    seen_ids = set()

    for record in sources:
        validate_source_record(record)

        document_id = record["document_id"]
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document_id: {document_id}")

        seen_ids.add(document_id)

    return registry


def save_registry(
    path: str | Path,
    registry: dict[str, Any],
) -> None:
    """Validate and save the registry as UTF-8 JSON."""

    if not isinstance(registry, dict):
        raise ValueError("Registry must be a dictionary")

    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported registry schema version")

    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Registry sources must be a list")

    seen_ids = set()

    for record in sources:
        validate_source_record(record)

        document_id = record["document_id"]
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document_id: {document_id}")

        seen_ids.add(document_id)

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = registry_path.with_suffix(
        registry_path.suffix + ".tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(registry_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def upsert_source(
    registry: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Insert a new source or replace the record with the same ID."""

    validate_source_record(record)

    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Registry sources must be a list")

    for index, existing in enumerate(sources):
        if existing.get("document_id") == record["document_id"]:
            sources[index] = record.copy()
            return

    sources.append(record.copy())


def compute_sha256(path: str | Path) -> str:
    """Return the SHA-256 hash of a file, reading it in chunks."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()
