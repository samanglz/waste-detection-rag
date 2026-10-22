from src.generation.rag_pipeline import (
    INSUFFICIENT_CONTEXT_ANSWER,
    answer_question,
)


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5):
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return self.answer


def make_chunk(text, source="paper.txt"):
    return {
        "chunk_id": f"{source}::chunk_0",
        "source": source,
        "text": text,
        "score": 0.9,
    }


def test_pipeline_returns_valid_citation_report():
    retriever = FakeRetriever([
        make_chunk("Paper can be recycled."),
    ])
    generator = FakeGenerator("Paper can be recycled. [S1]")

    result = answer_question(
        retriever,
        generator,
        "Can paper be recycled?",
    )

    assert result["answer"] == "Paper can be recycled. [S1]"
    assert result["citation_validation"]["is_valid"] is True
    assert result["citation_validation"]["cited_ids"] == ["S1"]
    assert result["citation_validation"]["invalid_ids"] == []


def test_pipeline_reports_unknown_citation_without_rewriting_answer():
    retriever = FakeRetriever([
        make_chunk("Paper can be recycled."),
    ])
    generator = FakeGenerator("Paper can be recycled. [S9]")

    result = answer_question(
        retriever,
        generator,
        "Can paper be recycled?",
    )

    assert result["answer"] == "Paper can be recycled. [S9]"
    assert result["citation_validation"]["is_valid"] is False
    assert result["citation_validation"]["invalid_ids"] == ["S9"]


def test_pipeline_reports_answer_without_citations():
    retriever = FakeRetriever([
        make_chunk("Paper can be recycled."),
    ])
    generator = FakeGenerator("Paper can be recycled.")

    result = answer_question(
        retriever,
        generator,
        "Can paper be recycled?",
    )

    assert result["citation_validation"]["has_citations"] is False
    assert result["citation_validation"]["is_valid"] is False


def test_empty_context_keeps_existing_behavior():
    retriever = FakeRetriever([])
    generator = FakeGenerator("This should never be returned.")

    result = answer_question(
        retriever,
        generator,
        "What can be recycled?",
    )

    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert result["prompt"] is None
    assert generator.calls == 0
    assert "citation_validation" not in result
