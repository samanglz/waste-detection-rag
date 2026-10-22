import numpy as np

from src.indexing.faiss_index import FaissIndex
from src.retrieval.hybrid_retriever import HybridRetriever


class FakeEncoder:
    def encode_queries(self, queries):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeBM25Retriever:
    def search(self, query, top_k=5):
        # Return (chunk_index, lexical_score) pairs.
        return [(0, 1.0), (1, 0.5)][:top_k]


def make_retriever():
    chunks = [
        {
            "chunk_id": "plastic.txt::chunk_0",
            "source": "plastic.txt",
            "text": "Plastic recycling.",
            "metadata": {
                "title": "Plastic recycling",
                "topic": "plastic",
            },
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

    return HybridRetriever(
        chunks=chunks,
        encoder=FakeEncoder(),
        vector_index=index,
        bm25_retriever=FakeBM25Retriever(),
    )


def test_hybrid_retriever_preserves_metadata():
    retriever = make_retriever()

    results = retriever.search("plastic recycling", top_k=2)
    result_by_id = {item["chunk_id"]: item for item in results}

    assert result_by_id["plastic.txt::chunk_0"]["metadata"] == {
        "title": "Plastic recycling",
        "topic": "plastic",
    }


def test_hybrid_retriever_returns_empty_metadata_when_missing():
    retriever = make_retriever()

    results = retriever.search("plastic recycling", top_k=2)
    result_by_id = {item["chunk_id"]: item for item in results}

    assert result_by_id["paper.txt::chunk_0"]["metadata"] == {}


def test_hybrid_retriever_metadata_is_independent_of_source_chunk():
    retriever = make_retriever()

    results = retriever.search("plastic recycling", top_k=2)
    result_by_id = {item["chunk_id"]: item for item in results}

    result_by_id["plastic.txt::chunk_0"]["metadata"]["topic"] = "modified"

    assert (
        retriever.chunks[0]["metadata"]["topic"]
        == "plastic"
    )
