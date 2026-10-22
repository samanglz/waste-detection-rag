import json

import numpy as np
import pytest

from src.indexing.faiss_index import FaissIndex
from src.indexing.index_store import IndexStore


def make_index():
    index = FaissIndex(dimension=2)
    index.add(
        np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
    )
    return index


def make_chunks():
    return [
        {
            "chunk_id": "plastic.txt::chunk_0",
            "source": "plastic.txt",
            "text": "Plastic can be recycled.",
        },
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper can be recycled.",
        },
    ]


def test_save_and_load_store(tmp_path):
    store = IndexStore(tmp_path)
    index = make_index()
    chunks = make_chunks()

    store.save(index, chunks)
    restored_index, restored_chunks = store.load()

    assert restored_index.size == 2
    assert restored_chunks == chunks
    assert (tmp_path / "index.faiss").is_file()
    assert (tmp_path / "chunks.json").is_file()


def test_save_rejects_mismatched_counts(tmp_path):
    store = IndexStore(tmp_path)

    with pytest.raises(ValueError, match="number of chunks"):
        store.save(make_index(), make_chunks()[:1])


def test_save_rejects_empty_chunks(tmp_path):
    store = IndexStore(tmp_path)
    empty_index = FaissIndex(dimension=2)

    with pytest.raises(ValueError, match="must not be empty"):
        store.save(empty_index, [])


def test_load_rejects_missing_index(tmp_path):
    store = IndexStore(tmp_path)
    (tmp_path / "chunks.json").write_text(
        json.dumps(make_chunks()),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        store.load()


def test_load_rejects_missing_metadata(tmp_path):
    store = IndexStore(tmp_path)
    make_index().save(tmp_path / "index.faiss")

    with pytest.raises(
        FileNotFoundError,
        match="Chunk metadata not found",
    ):
        store.load()


def test_load_rejects_invalid_json(tmp_path):
    store = IndexStore(tmp_path)
    make_index().save(tmp_path / "index.faiss")
    (tmp_path / "chunks.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid chunk metadata JSON"):
        store.load()


def test_load_rejects_mismatched_counts(tmp_path):
    store = IndexStore(tmp_path)
    make_index().save(tmp_path / "index.faiss")
    (tmp_path / "chunks.json").write_text(
        json.dumps(make_chunks()[:1]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        store.load()


def test_save_rejects_missing_chunk_fields(tmp_path):
    store = IndexStore(tmp_path)
    chunks = make_chunks()
    del chunks[0]["source"]

    with pytest.raises(ValueError, match="missing required keys"):
        store.save(make_index(), chunks)

def test_save_and_load_preserves_chunk_metadata(tmp_path):
    store = IndexStore(tmp_path)
    index = make_index()
    chunks = make_chunks()
    chunks[0]["metadata"] = {
        "title": "Plastic recycling",
        "source_url": "https://example.org/plastic",
        "topic": "plastic",
    }
    chunks[1]["metadata"] = {"topic": "paper"}

    store.save(index, chunks)
    _, restored_chunks = store.load()

    assert restored_chunks == chunks
    assert restored_chunks[0]["metadata"]["title"] == "Plastic recycling"
    assert restored_chunks[1]["metadata"]["topic"] == "paper"
