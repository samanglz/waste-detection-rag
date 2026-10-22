
import pytest

from src.evaluation.retrieval_metrics import evaluate_retrieval


class FakeRetriever:
    """A predictable retriever for testing evaluation metrics."""

    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(self, query, top_k=3):
        return self.results_by_query[query][:top_k]


def test_perfect_retrieval():
    retriever = FakeRetriever({
        "plastic?": [
            {"source": "plastic.txt", "rank": 1},
            {"source": "paper.txt", "rank": 2},
        ],
        "paper?": [
            {"source": "paper.txt", "rank": 1},
            {"source": "glass.txt", "rank": 2},
        ],
    })

    queries = [
        {"query": "plastic?", "relevant_source": "plastic.txt"},
        {"query": "paper?", "relevant_source": "paper.txt"},
    ]

    metrics = evaluate_retrieval(retriever, queries, k=2)

    assert metrics["Hit Rate@k"] == pytest.approx(1.0)
    assert metrics["MRR@k"] == pytest.approx(1.0)


def test_relevant_document_at_rank_two():
    retriever = FakeRetriever({
        "plastic?": [
            {"source": "paper.txt", "rank": 1},
            {"source": "plastic.txt", "rank": 2},
        ],
    })

    queries = [
        {"query": "plastic?", "relevant_source": "plastic.txt"},
    ]

    metrics = evaluate_retrieval(retriever, queries, k=2)

    assert metrics["Hit Rate@k"] == pytest.approx(1.0)
    assert metrics["MRR@k"] == pytest.approx(0.5)


def test_no_relevant_document_in_top_k():
    retriever = FakeRetriever({
        "plastic?": [
            {"source": "paper.txt", "rank": 1},
            {"source": "glass.txt", "rank": 2},
        ],
    })

    queries = [
        {"query": "plastic?", "relevant_source": "plastic.txt"},
    ]

    metrics = evaluate_retrieval(retriever, queries, k=2)

    assert metrics["Hit Rate@k"] == pytest.approx(0.0)
    assert metrics["MRR@k"] == pytest.approx(0.0)


def test_invalid_k():
    with pytest.raises(ValueError):
        evaluate_retrieval(FakeRetriever({}), [], k=0)


def test_empty_queries():
    with pytest.raises(ValueError):
        evaluate_retrieval(FakeRetriever({}), [], k=3)
