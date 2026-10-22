import numpy as np
import pytest

from src.indexing.faiss_index import FaissIndex
from src.retrieval.dense_retriever import DenseRetriever


class FakeEncoder:
    def encode_queries(self, queries):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def make_retriever():
    chunks = [
        {
            "chunk_id": "plastic.txt::chunk_0",
            "source": "plastic.txt",
            "text": "Plastic recycling.",
        },
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper recycling.",
        },
    ]

    index = FaissIndex(dimension=2)
    index.add(
        np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
    )

    return DenseRetriever(
        chunks=chunks,
        encoder=FakeEncoder(),
        vector_index=index,
    )


def test_dense_retriever_returns_ranked_chunks():
    retriever = make_retriever()

    results = retriever.search("plastic recycling", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "plastic.txt::chunk_0"
    assert results[0]["source"] == "plastic.txt"
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert results[0]["score"] > results[1]["score"]


def test_dense_retriever_rejects_empty_query():
    retriever = make_retriever()

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search("  ")


def test_dense_retriever_rejects_invalid_top_k():
    retriever = make_retriever()

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search("plastic", top_k=0)


def test_dense_retriever_rejects_mismatched_chunk_count():
    index = FaissIndex(dimension=2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))

    chunks = [
        {
            "chunk_id": "plastic.txt::chunk_0",
            "source": "plastic.txt",
            "text": "Plastic recycling.",
        },
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper recycling.",
        },
    ]

    with pytest.raises(
        ValueError,
        match="FAISS index size must match",
    ):
        DenseRetriever(
            chunks=chunks,
            encoder=FakeEncoder(),
            vector_index=index,
        )

def test_dense_retriever_preserves_metadata():
    retriever = make_retriever()
    retriever.chunks[0]["metadata"] = {
        "title": "Plastic recycling",
        "topic": "plastic",
    }

    results = retriever.search("plastic recycling", top_k=1)

    assert results[0]["metadata"] == {
        "title": "Plastic recycling",
        "topic": "plastic",
    }


def test_dense_retriever_metadata_is_independent_of_source_chunk():
    retriever = make_retriever()
    retriever.chunks[0]["metadata"] = {"topic": "plastic"}

    results = retriever.search("plastic recycling", top_k=1)
    results[0]["metadata"]["topic"] = "modified"

    assert retriever.chunks[0]["metadata"]["topic"] == "plastic"


def test_dense_retriever_supports_chunks_without_metadata():
    retriever = make_retriever()

    results = retriever.search("plastic recycling", top_k=1)

    assert results[0]["metadata"] == {}
