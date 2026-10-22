import pytest

from src.retrieval.bm25_adapter import BM25SearchAdapter
from src.retrieval.bm25_retriever import BM25Retriever


def make_chunks():
    return [
        {
            "chunk_id": "plastic.txt::chunk_0",
            "source": "plastic.txt",
            "text": "PET plastic bottles can be recycled.",
            "metadata": {
                "title": "Plastic recycling",
                "topic": "plastic",
            },
        },
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper recycling saves trees.",
        },
    ]


def make_adapter(chunks):
    return BM25SearchAdapter(BM25Retriever(chunks), chunks)


def test_bm25_adapter_returns_shared_result_format():
    chunks = make_chunks()
    results = make_adapter(chunks).search(
        "PET plastic bottles",
        top_k=2,
    )

    assert len(results) == 2
    assert {item["chunk_id"] for item in results} == {
        "plastic.txt::chunk_0",
        "paper.txt::chunk_0",
    }

    required_fields = {
        "chunk_id", "source", "text", "metadata", "score", "rank"
    }

    for result in results:
        assert required_fields.issubset(result)
        assert isinstance(result["chunk_id"], str)
        assert isinstance(result["source"], str)
        assert isinstance(result["text"], str)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["score"], float)
        assert isinstance(result["rank"], int)

    assert [item["rank"] for item in results] == [1, 2]


def test_bm25_adapter_preserves_metadata():
    chunks = make_chunks()
    results = make_adapter(chunks).search(
        "PET plastic bottles",
        top_k=2,
    )

    result_by_id = {item["chunk_id"]: item for item in results}

    assert result_by_id["plastic.txt::chunk_0"]["metadata"] == {
        "title": "Plastic recycling",
        "topic": "plastic",
    }


def test_bm25_adapter_returns_empty_metadata_when_missing():
    chunks = make_chunks()
    results = make_adapter(chunks).search("paper recycling", top_k=2)
    result_by_id = {item["chunk_id"]: item for item in results}

    assert result_by_id["paper.txt::chunk_0"]["metadata"] == {}


def test_bm25_adapter_metadata_is_independent_of_source_chunk():
    chunks = make_chunks()
    results = make_adapter(chunks).search(
        "PET plastic bottles",
        top_k=1,
    )

    results[0]["metadata"]["topic"] = "modified"

    assert chunks[0]["metadata"]["topic"] == "plastic"


def test_bm25_adapter_rejects_mismatched_document_count():
    chunks = make_chunks()
    bm25 = BM25Retriever(chunks[:1])

    with pytest.raises(ValueError, match="Chunk count"):
        BM25SearchAdapter(bm25, chunks)
