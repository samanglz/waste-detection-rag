import numpy as np
import pytest

from src.indexing.index_builder import build_faiss_index


class FakeEncoder:
    def encode_documents(self, texts):
        vectors = {
            "plastic recycling": [1.0, 0.0],
            "paper recycling": [0.0, 1.0],
        }
        return np.array(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )


class InvalidEncoder:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode_documents(self, texts):
        return self.embeddings


def test_builds_index_from_chunks():
    chunks = [
        {
            "chunk_id": "plastic::0",
            "source": "plastic.txt",
            "text": "plastic recycling",
        },
        {
            "chunk_id": "paper::0",
            "source": "paper.txt",
            "text": "paper recycling",
        },
    ]

    index = build_faiss_index(chunks, FakeEncoder())

    assert index.size == 2

    scores, indices = index.search(
        np.array([[1.0, 0.0]], dtype=np.float32),
        top_k=1,
    )

    assert indices[0, 0] == 0
    assert scores[0, 0] == pytest.approx(1.0)


def test_rejects_empty_chunks():
    with pytest.raises(ValueError, match="chunks must not be empty"):
        build_faiss_index([], FakeEncoder())


def test_rejects_one_dimensional_embeddings():
    chunks = [
        {
            "chunk_id": "plastic::0",
            "source": "plastic.txt",
            "text": "plastic recycling",
        }
    ]

    encoder = InvalidEncoder(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    with pytest.raises(
        ValueError,
        match="one 2D embedding per chunk",
    ):
        build_faiss_index(chunks, encoder)


def test_rejects_wrong_number_of_embeddings():
    chunks = [
        {
            "chunk_id": "plastic::0",
            "source": "plastic.txt",
            "text": "plastic recycling",
        },
        {
            "chunk_id": "paper::0",
            "source": "paper.txt",
            "text": "paper recycling",
        },
    ]

    encoder = InvalidEncoder(
        np.array([[1.0, 0.0]], dtype=np.float32)
    )

    with pytest.raises(
        ValueError,
        match="one 2D embedding per chunk",
    ):
        build_faiss_index(chunks, encoder)


def test_rejects_zero_dimensional_embeddings():
    chunks = [
        {
            "chunk_id": "plastic::0",
            "source": "plastic.txt",
            "text": "plastic recycling",
        }
    ]

    encoder = InvalidEncoder(
        np.empty((1, 0), dtype=np.float32)
    )

    with pytest.raises(
        ValueError,
        match="Embedding dimension must be positive",
    ):
        build_faiss_index(chunks, encoder)

def test_rejects_duplicate_chunk_ids():
    chunks = [
        {
            "chunk_id": "duplicate",
            "source": "plastic.txt",
            "text": "plastic recycling",
        },
        {
            "chunk_id": "duplicate",
            "source": "paper.txt",
            "text": "paper recycling",
        },
    ]

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        build_faiss_index(chunks, FakeEncoder())


def test_rejects_empty_chunk_text_before_encoding():
    chunks = [{
        "chunk_id": "plastic::0",
        "source": "plastic.txt",
        "text": " ",
    }]

    with pytest.raises(ValueError, match="invalid text"):
        build_faiss_index(chunks, FakeEncoder())


def test_rejects_non_dictionary_metadata():
    chunks = [{
        "chunk_id": "plastic::0",
        "source": "plastic.txt",
        "text": "plastic recycling",
        "metadata": ["invalid"],
    }]

    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        build_faiss_index(chunks, FakeEncoder())
