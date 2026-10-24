from unittest.mock import Mock

import pytest

from src.generation.rag_pipeline import retrieve_context


def make_chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "source": "recycling.txt",
        "text": text,
        "score": 0.5,
    }


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_reranker_is_used_when_injected():
    hybrid = FakeRetriever([
        make_chunk("hybrid-1", "Hybrid result."),
    ])
    reranker = Mock()
    reranker.search.return_value = [
        make_chunk("reranked-1", "Reranked result."),
    ]

    result = retrieve_context(
        retriever=hybrid,
        reranker=reranker,
        query="recycling",
        top_k=5,
    )

    reranker.search.assert_called_once_with("recycling", top_k=5)
    assert hybrid.calls == []
    assert result["retrieved_chunks"][0]["chunk_id"] == "reranked-1"
    assert result["retrieval_method"] == "hybrid+reranker"
    assert result["reranker_error"] is None


def test_original_retriever_is_used_when_reranker_is_not_injected():
    hybrid = FakeRetriever([
        make_chunk("hybrid-1", "Hybrid result."),
    ])

    result = retrieve_context(
        retriever=hybrid,
        query="recycling",
        top_k=1,
    )

    assert hybrid.calls == [("recycling", 1)]
    assert result["retrieved_chunks"][0]["chunk_id"] == "hybrid-1"
    assert result["retrieval_method"] == "hybrid"
    assert result["reranker_error"] is None


def test_reranker_error_falls_back_to_original_retriever_by_default():
    hybrid = FakeRetriever([
        make_chunk("hybrid-1", "Hybrid fallback result."),
    ])
    reranker = Mock()
    reranker.search.side_effect = RuntimeError("model unavailable")

    result = retrieve_context(
        retriever=hybrid,
        reranker=reranker,
        query="recycling",
        top_k=1,
    )

    assert hybrid.calls == [("recycling", 1)]
    assert result["retrieved_chunks"][0]["chunk_id"] == "hybrid-1"
    assert result["retrieval_method"] == "hybrid_fallback"
    assert "model unavailable" in result["reranker_error"]


def test_reranker_error_can_be_configured_to_raise():
    hybrid = FakeRetriever([
        make_chunk("hybrid-1", "Hybrid result."),
    ])
    reranker = Mock()
    reranker.search.side_effect = RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        retrieve_context(
            retriever=hybrid,
            reranker=reranker,
            query="recycling",
            fallback_on_reranker_error=False,
        )

    assert hybrid.calls == []
