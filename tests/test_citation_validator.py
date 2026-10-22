import pytest

from src.generation.citation_validator import validate_citations


def test_all_citations_exist_in_source_registry():
    result = validate_citations(
        "Paper [S1]. Glass [S2].",
        [
            {"id": "S1", "source": "paper.txt"},
            {"id": "S2", "source": "glass.txt"},
        ],
    )

    assert result["is_valid"] is True
    assert result["has_citations"] is True
    assert result["all_citations_exist"] is True
    assert result["cited_ids"] == ["S1", "S2"]
    assert result["invalid_ids"] == []
    assert result["unused_source_ids"] == []


def test_unknown_citation_is_reported():
    result = validate_citations(
        "Plastic can be recycled [S3].",
        [{"id": "S1", "source": "recycling.txt"}],
    )

    assert result["is_valid"] is False
    assert result["all_citations_exist"] is False
    assert result["invalid_ids"] == ["S3"]


def test_repeated_citations_are_deduplicated():
    result = validate_citations(
        "Paper [S1]. More paper [S1].",
        [{"id": "S1", "source": "paper.txt"}],
    )

    assert result["cited_ids"] == ["S1"]
    assert result["is_valid"] is True


def test_unused_sources_are_reported():
    result = validate_citations(
        "Paper [S1].",
        [
            {"id": "S1", "source": "paper.txt"},
            {"id": "S2", "source": "glass.txt"},
        ],
    )

    assert result["is_valid"] is True
    assert result["unused_source_ids"] == ["S2"]


def test_no_citations_are_not_marked_as_validated():
    result = validate_citations(
        "The context does not contain enough information.",
        [{"id": "S1", "source": "recycling.txt"}],
    )

    assert result["is_valid"] is False
    assert result["has_citations"] is False
    assert result["all_citations_exist"] is True
    assert result["cited_ids"] == []


def test_citation_with_empty_source_registry_is_invalid():
    result = validate_citations("Paper [S1].", [])

    assert result["is_valid"] is False
    assert result["invalid_ids"] == ["S1"]


def test_invalid_argument_types_are_rejected():
    with pytest.raises(TypeError, match="answer must be a string"):
        validate_citations(None, [])

    with pytest.raises(TypeError, match="sources must be a list"):
        validate_citations("Answer [S1]", None)
