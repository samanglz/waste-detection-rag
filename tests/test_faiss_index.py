import numpy as np
import pytest

from src.indexing.faiss_index import FaissIndex


def test_save_and_load_index(tmp_path):
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    original = FaissIndex(dimension=2)
    original.add(embeddings)

    path = tmp_path / "indexes" / "test.index"
    original.save(path)

    restored = FaissIndex.load(path)

    assert restored.size == 2

    scores, indices = restored.search(
        np.array([[1.0, 0.0]], dtype=np.float32),
        top_k=1,
    )

    assert indices[0, 0] == 0
    assert scores[0, 0] == pytest.approx(1.0)


def test_load_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.index"

    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        FaissIndex.load(path)


def test_add_rejects_non_finite_embeddings():
    index = FaissIndex(dimension=2)
    embeddings = np.array([[np.nan, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite values"):
        index.add(embeddings)


def test_search_rejects_non_finite_queries():
    index = FaissIndex(dimension=2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))

    query = np.array([[np.inf, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite values"):
        index.search(query)
