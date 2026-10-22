from src.generation.rag_pipeline import (
    INSUFFICIENT_CONTEXT_ANSWER,
    answer_question,
)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self, answer="Paper and glass can be recycled. [S1]"):
        self.answer = answer
        self.last_prompt = None
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        return self.answer


def test_answer_question_connects_retrieval_prompt_and_generation():
    retriever = FakeRetriever([
        {
            "chunk_id": "recycling.txt::chunk_0",
            "source": "recycling.txt",
            "text": "Paper and glass can be recycled.",
            "score": 0.95,
        }
    ])
    generator = FakeGenerator()

    result = answer_question(
        retriever=retriever,
        generator=generator,
        query="What materials can be recycled?",
        top_k=3,
        max_chars=1000,
    )

    assert retriever.last_query == "What materials can be recycled?"
    assert retriever.last_top_k == 3
    assert len(result["retrieved_chunks"]) == 1
    assert result["context"]
    assert result["sources"] == [
        {"id": "S1", "source": "recycling.txt"}
    ]
    assert "What materials can be recycled?" in result["prompt"]
    assert "Paper and glass can be recycled." in result["prompt"]
    assert generator.last_prompt == result["prompt"]
    assert generator.calls == 1
    assert result["answer"] == "Paper and glass can be recycled. [S1]"


def test_answer_question_does_not_call_llm_without_context():
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    result = answer_question(
        retriever=retriever,
        generator=generator,
        query="What materials can be recycled?",
    )

    assert result["context"] == ""
    assert result["prompt"] is None
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert generator.calls == 0


def test_answer_question_rejects_empty_query():
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    try:
        answer_question(retriever, generator, "  ")
    except ValueError as exc:
        assert "query must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query")

    assert generator.calls == 0
