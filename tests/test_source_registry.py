import json

import pytest

from src.ingestion.source_registry import (
    compute_sha256,
    load_registry,
    save_registry,
    upsert_source,
    validate_source_record,
)


def make_record(**overrides):
    record = {
        "document_id": "epa_common_recyclables",
        "title": "How Do I Recycle Common Recyclables?",
        "publisher": "U.S. Environmental Protection Agency",
        "source_type": "official_guidance",
        "url": "https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables",
        "doi": None,
        "published_date": None,
        "accessed_date": "2026-10-20",
        "language": "en",
        "topics": ["paper", "glass", "metal", "plastic"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
        "content_sha256": None,
        "local_path": None,
    }
    record.update(overrides)
    return record


def test_valid_source_record():
    validate_source_record(make_record())


def test_rejects_missing_required_field():
    record = make_record()
    del record["publisher"]

    with pytest.raises(ValueError, match="missing fields"):
        validate_source_record(record)


def test_rejects_invalid_url():
    with pytest.raises(ValueError, match="HTTP"):
        validate_source_record(make_record(url="not-a-url"))


def test_rejects_non_english_source():
    with pytest.raises(ValueError, match="English"):
        validate_source_record(make_record(language="fa"))


def test_rejects_invalid_status():
    with pytest.raises(ValueError, match="Invalid status"):
        validate_source_record(make_record(status="unknown"))


def test_upsert_adds_and_updates_record():
    registry = {"schema_version": 1, "sources": []}

    first = make_record()
    upsert_source(registry, first)
    assert len(registry["sources"]) == 1

    updated = make_record(title="Updated title")
    upsert_source(registry, updated)

    assert len(registry["sources"]) == 1
    assert registry["sources"][0]["title"] == "Updated title"


def test_save_and_load_registry(tmp_path):
    path = tmp_path / "corpus_manifest.json"
    registry = {
        "schema_version": 1,
        "sources": [make_record()],
    }

    save_registry(path, registry)
    loaded = load_registry(path)

    assert loaded == registry
    assert json.loads(path.read_text(encoding="utf-8")) == registry


def test_rejects_duplicate_ids_when_saving(tmp_path):
    record = make_record()
    registry = {
        "schema_version": 1,
        "sources": [record, record.copy()],
    }

    with pytest.raises(ValueError, match="Duplicate document_id"):
        save_registry(tmp_path / "registry.json", registry)


def test_missing_registry_returns_empty_registry(tmp_path):
    registry = load_registry(tmp_path / "missing.json")

    assert registry == {"schema_version": 1, "sources": []}


def test_compute_sha256(tmp_path):
    import hashlib

    path = tmp_path / "sample.txt"
    content = b"recycling"
    path.write_bytes(content)

    assert compute_sha256(path) == hashlib.sha256(content).hexdigest()
