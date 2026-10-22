import pytest

from src.generation.rag_pipeline import retrieve_context


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]


def test_retrieve_context_connects_retrieval_and_context_builder():
    retriever = FakeRetriever([
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper is mixed with water to create pulp.",
            "metadata": {"topic": "paper"},
            "score": 0.9,
            "rank": 1,
        }
    ])

    output = retrieve_context(
        retriever,
        "How is paper recycled?",
        top_k=3,
    )

    assert retriever.last_query == "How is paper recycled?"
    assert retriever.last_top_k == 3
    assert "create pulp" in output["context"]
    assert output["sources"] == [
        {"id": "S1", "source": "paper.txt"}
    ]
    assert len(output["retrieved_chunks"]) == 1


def test_retrieve_context_preserves_retrieved_chunks():
    results = [
        {
            "chunk_id": "glass.txt::chunk_0",
            "source": "glass.txt",
            "text": "Glass can be melted and remade.",
            "metadata": {"topic": "glass"},
            "score": 0.8,
            "rank": 1,
        }
    ]

    output = retrieve_context(FakeRetriever(results), "glass recycling")

    assert output["retrieved_chunks"] == results


def test_retrieve_context_handles_no_results():
    output = retrieve_context(FakeRetriever([]), "unknown material")

    assert output["retrieved_chunks"] == []
    assert output["context"] == ""
    assert output["sources"] == []


def test_retrieve_context_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        retrieve_context(FakeRetriever([]), "   ")


def test_retrieve_context_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be positive"):
        retrieve_context(FakeRetriever([]), "paper", top_k=0)


def test_retrieve_context_respects_context_limit():
    results = [
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper recycling information. " * 100,
        }
    ]

    output = retrieve_context(
        FakeRetriever(results),
        "paper recycling",
        max_chars=100,
    )

    assert len(output["context"]) <= 100
