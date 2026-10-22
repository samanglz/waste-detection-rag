from unittest.mock import Mock

import pytest

from src.generation.rag_pipeline import (
    INSUFFICIENT_CONTEXT_ANSWER,
    answer_question,
)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def make_chunk(text, source="recycling_guide.txt"):
    return {
        "chunk_id": f"{source}::chunk_0",
        "source": source,
        "text": text,
        "score": 0.95,
    }


def test_empty_query_is_rejected_before_generation():
    retriever = FakeRetriever([])
    generator = Mock()

    with pytest.raises(ValueError, match="query must not be empty"):
        answer_question(retriever, generator, "   ")

    generator.generate.assert_not_called()
    assert retriever.calls == []


def test_empty_retrieved_context_does_not_call_generator():
    retriever = FakeRetriever([])
    generator = Mock()

    result = answer_question(
        retriever=retriever,
        generator=generator,
        query="What can be recycled?",
    )

    generator.generate.assert_not_called()
    assert result["retrieved_chunks"] == []
    assert result["context"] == ""
    assert result["prompt"] is None
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER


def test_nonempty_context_calls_generator_with_prompt():
    retriever = FakeRetriever([
        make_chunk("Glass bottles can be recycled."),
    ])
    generator = Mock()
    generator.generate.return_value = (
        "Glass bottles can be recycled. [S1]"
    )

    result = answer_question(
        retriever=retriever,
        generator=generator,
        query="Can glass bottles be recycled?",
    )

    generator.generate.assert_called_once_with(result["prompt"])
    assert "Can glass bottles be recycled?" in result["prompt"]
    assert "Glass bottles can be recycled." in result["prompt"]
    assert result["context"]
    assert result["query"] == "Can glass bottles be recycled?"
    assert result["answer"] == "Glass bottles can be recycled. [S1]"


def test_top_k_is_passed_to_retriever():
    retriever = FakeRetriever([
        make_chunk("Glass bottles can be recycled."),
    ])
    generator = Mock()
    generator.generate.return_value = "Yes. [S1]"

    answer_question(
        retriever=retriever,
        generator=generator,
        query="Can glass bottles be recycled?",
        top_k=2,
    )

    assert retriever.calls == [
        ("Can glass bottles be recycled?", 2)
    ]
