import pytest

from src.generation.context_builder import build_context


def test_build_context_includes_text_and_source():
    results = [
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper is mixed with water to create pulp.",
        }
    ]

    output = build_context(results)

    assert "[S1]" in output["context"]
    assert "paper.txt" in output["context"]
    assert "create pulp" in output["context"]
    assert output["sources"] == [
        {"id": "S1", "source": "paper.txt"}
    ]


def test_chunks_from_same_source_share_source_id():
    results = [
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "First chunk.",
        },
        {
            "chunk_id": "paper.txt::chunk_1",
            "source": "paper.txt",
            "text": "Second chunk.",
        },
    ]

    output = build_context(results)

    assert output["context"].count("[S1]") == 2
    assert len(output["sources"]) == 1


def test_context_respects_character_limit():
    results = [
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "A" * 1000,
        }
    ]

    output = build_context(results, max_chars=100)

    assert len(output["context"]) <= 100


def test_empty_results_produce_empty_context():
    output = build_context([])

    assert output == {"context": "", "sources": []}


def test_invalid_character_limit_is_rejected():
    with pytest.raises(ValueError, match="max_chars must be positive"):
        build_context([], max_chars=0)


def test_empty_chunks_are_skipped():
    results = [
        {"chunk_id": "empty", "source": "empty.txt", "text": "  "},
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper pulp.",
        },
    ]

    output = build_context(results)

    assert "empty.txt" not in output

def test_sources_only_contains_sources_present_in_context():
    from src.generation.context_builder import build_context

    results = [
        {
            "text": "A sufficiently long piece of content.",
            "source": "source_a.txt",
            "chunk_id": "chunk_1",
        },
        {
            "text": "Another piece of content.",
            "source": "source_b.txt",
            "chunk_id": "chunk_2",
        },
    ]

    result = build_context(results, max_chars=1)

    assert result["context"] == ""
    assert result["sources"] == []
