import pytest

from src.generation.rag_pipeline import prepare_rag_prompt


class FakeRetriever:
    def search(self, query, top_k=5):
        return [
            {
                "chunk_id": "chunk_1",
                "source": "recycling.txt",
                "text": "Paper and glass can be recycled.",
                "score": 0.95,
            }
        ][:top_k]


def test_prepare_rag_prompt_connects_retrieval_and_prompt_builder():
    result = prepare_rag_prompt(
        retriever=FakeRetriever(),
        query="What materials can be recycled?",
        top_k=3,
        max_chars=1000,
    )

    assert result["query"] == "What materials can be recycled?"
    assert len(result["retrieved_chunks"]) == 1
    assert result["context"]
    assert result["sources"]
    assert result["prompt"]

    assert "What materials can be recycled?" in result["prompt"]
    assert "Paper and glass can be recycled." in result["prompt"]
    assert "[S1] recycling.txt" in result["prompt"]


def test_prepare_rag_prompt_preserves_retrieval_results():
    retriever = FakeRetriever()

    result = prepare_rag_prompt(
        retriever=retriever,
        query="What materials can be recycled?",
    )

    assert result["retrieved_chunks"][0]["chunk_id"] == "chunk_1"
    assert result["retrieved_chunks"][0]["score"] == 0.95


def test_prepare_rag_prompt_rejects_empty_query():
    with pytest.raises(ValueError, match="query must not be empty"):
        prepare_rag_prompt(
            retriever=FakeRetriever(),
            query="   ",
        )


def test_prepare_rag_prompt_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be positive"):
        prepare_rag_prompt(
            retriever=FakeRetriever(),
            query="What materials can be recycled?",
            top_k=0,
        )


def test_prepare_rag_prompt_rejects_empty_retrieved_context():
    class EmptyRetriever:
        def search(self, query, top_k=5):
            return []

    with pytest.raises(ValueError, match="context must not be empty"):
        prepare_rag_prompt(
            retriever=EmptyRetriever(),
            query="What materials can be recycled?",
        )
