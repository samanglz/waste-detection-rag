from typing import Any

from src.generation.context_builder import build_context
from src.generation.prompt_builder import build_prompt
from src.generation.citation_validator import validate_citations


INSUFFICIENT_CONTEXT_ANSWER = (
    "I don't have enough information in the retrieved context "
    "to answer this question."
)


def retrieve_context(
    retriever: Any,
    query: str,
    top_k: int = 5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Retrieve relevant chunks and assemble a bounded context."""

    if not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results = retriever.search(query, top_k=top_k)

    context_result = build_context(
        results,
        max_chars=max_chars,
    )

    return {
        "query": query,
        "retrieved_chunks": results,
        "context": context_result["context"],
        "sources": context_result["sources"],
    }


def prepare_rag_prompt(
    retriever: Any,
    query: str,
    top_k: int = 5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Retrieve context and prepare a prompt for a downstream LLM."""

    result = retrieve_context(
        retriever=retriever,
        query=query,
        top_k=top_k,
        max_chars=max_chars,
    )

    prompt = build_prompt(
        query=result["query"],
        context=result["context"],
        sources=result["sources"],
    )

    return {
        **result,
        "prompt": prompt,
    }


def answer_question(
    retriever: Any,
    generator: Any,
    query: str,
    top_k: int = 5,
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Run retrieval, prompt construction and LLM generation.

    Both the retriever and generator are injected by the caller.
    If retrieval produces no usable context, the LLM is not called.
    """

    result = retrieve_context(
        retriever=retriever,
        query=query,
        top_k=top_k,
        max_chars=max_chars,
    )

    if not result["context"].strip():
        return {
            **result,
            "prompt": None,
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
        }

    prompt = build_prompt(
        query=result["query"],
        context=result["context"],
        sources=result["sources"],
    )
    answer = generator.generate(prompt)

    citation_validation = validate_citations(
        answer=answer,
        sources=result["sources"],
    )

    return {
        **result,
        "prompt": prompt,
        "answer": answer,
        "citation_validation": citation_validation,
    }
