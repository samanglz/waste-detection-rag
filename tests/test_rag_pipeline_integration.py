from pathlib import Path

import pytest

from src.embeddings.encoder import TextEncoder
from src.generation.rag_pipeline import retrieve_context
from src.retrieval.retriever_factory import load_hybrid_retriever


INDEX_DIR = Path("data/processed/retrieval_index_v3")


@pytest.mark.integration
def test_real_retriever_builds_context_from_query():
    if not INDEX_DIR.exists():
        pytest.skip(f"Index directory not found: {INDEX_DIR}")

    encoder = TextEncoder()
    retriever = load_hybrid_retriever(
        index_dir=INDEX_DIR,
        encoder=encoder,
    )

    result = retrieve_context(
        retriever=retriever,
        query="What materials can be recycled?",
        top_k=5,
        max_chars=3000,
    )

    assert result["query"] == "What materials can be recycled?"
    assert len(result["retrieved_chunks"]) > 0
    assert result["context"].strip()
    assert result["sources"]

    for chunk in result["retrieved_chunks"]:
        assert chunk["text"].strip()
        assert chunk["source"].strip()

    for source in result["sources"]:
        assert source["id"].startswith("S")
        assert source["source"].strip()

    assert len(result["context"]) <= 3000
