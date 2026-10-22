import json
from pathlib import Path

import pytest

from src.ingestion.text_loader import load_text_documents


def test_load_text_documents(tmp_path: Path):
    (tmp_path / "plastic.txt").write_text(
        "PET can be recycled.",
        encoding="utf-8",
    )
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "notes.md").write_text(
        "This file should be ignored.",
        encoding="utf-8",
    )

    documents = load_text_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "plastic.txt"
    assert documents[0]["text"] == "PET can be recycled."
    assert documents[0]["metadata"] == {}


def test_loads_metadata_from_manifest(tmp_path: Path):
    (tmp_path / "plastic.txt").write_text(
        "PET can be recycled.",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "plastic.txt": {
                "title": "Plastic recycling",
                "source_url": "https://example.org/plastic",
                "jurisdiction": "general",
            }
        }),
        encoding="utf-8",
    )

    documents = load_text_documents(tmp_path)

    assert documents[0]["metadata"]["title"] == "Plastic recycling"
    assert documents[0]["metadata"]["jurisdiction"] == "general"


def test_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_text_documents("directory_that_does_not_exist_12345")


def test_rejects_file_instead_of_directory(tmp_path: Path):
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_text_documents(file_path)


def test_rejects_invalid_manifest_json(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid ingestion manifest"):
        load_text_documents(tmp_path)


def test_rejects_non_object_manifest(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(["not", "an", "object"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSON object"):
        load_text_documents(tmp_path)


def test_rejects_invalid_manifest_entry(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"plastic.txt": "metadata must be an object"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata objects"):
        load_text_documents(tmp_path)
