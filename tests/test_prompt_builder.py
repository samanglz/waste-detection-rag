import pytest

from src.generation.prompt_builder import build_prompt


def test_build_prompt_includes_query_context_and_sources():
    query = "What materials can be recycled?"
    context = "[S1] Source: recycling.txt\nContent: Paper and glass can be recycled."
    sources = [
        {"id": "S1", "source": "recycling.txt"},
    ]

    prompt = build_prompt(query, context, sources)

    assert query in prompt
    assert context in prompt
    assert "[S1] recycling.txt" in prompt
    assert "Answer:" in prompt


def test_build_prompt_includes_grounding_instructions():
    prompt = build_prompt(
        query="What can be recycled?",
        context="Paper can be recycled.",
        sources=[],
    )

    assert "using only the provided context" in prompt
    assert "Do not invent facts" in prompt
    assert "clearly say so" in prompt


def test_build_prompt_handles_empty_source_registry():
    prompt = build_prompt(
        query="What can be recycled?",
        context="Paper can be recycled.",
        sources=[],
    )

    assert "No source metadata provided." in prompt


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_build_prompt_rejects_empty_query(query):
    with pytest.raises(ValueError, match="query must not be empty"):
        build_prompt(
            query=query,
            context="Some context.",
            sources=[],
        )


@pytest.mark.parametrize("context", ["", "   ", "\n"])
def test_build_prompt_rejects_empty_context(context):
    with pytest.raises(ValueError, match="context must not be empty"):
        build_prompt(
            query="What can be recycled?",
            context=context,
            sources=[],
        )
