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
    reranker: Any = None,
    fallback_on_reranker_error: bool = True,
) -> dict[str, Any]:
    """Retrieve chunks and assemble bounded context.

    If a reranker is injected, it is used for retrieval. If reranking
    fails and fallback is enabled, retrieval falls back to the original
    retriever. The reranker should be constructed once and reused.
    """

    if not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    retrieval_method = "hybrid"
    reranker_error = None

    if reranker is None:
        results = retriever.search(query, top_k=top_k)
    else:
        try:
            results = reranker.search(query, top_k=top_k)
            retrieval_method = "hybrid+reranker"
        except Exception as exc:
            if not fallback_on_reranker_error:
                raise

            reranker_error = f"{type(exc).__name__}: {exc}"
            results = retriever.search(query, top_k=top_k)
            retrieval_method = "hybrid_fallback"

    context_result = build_context(
        results,
        max_chars=max_chars,
    )

    return {
        "query": query,
        "retrieved_chunks": results,
        "context": context_result["context"],
        "sources": context_result["sources"],
        "retrieval_method": retrieval_method,
        "reranker_error": reranker_error,
    }


def prepare_rag_prompt(
    retriever: Any,
    query: str,
    top_k: int = 5,
    max_chars: int = 6000,
    reranker: Any = None,
    fallback_on_reranker_error: bool = True,
) -> dict[str, Any]:
    """Retrieve context and prepare a prompt for a downstream LLM."""

    result = retrieve_context(
        retriever=retriever,
        query=query,
        top_k=top_k,
        max_chars=max_chars,
        reranker=reranker,
        fallback_on_reranker_error=fallback_on_reranker_error,
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
    reranker: Any = None,
    fallback_on_reranker_error: bool = True,
) -> dict[str, Any]:
    """Retrieve, prepare a prompt, and generate a grounded answer.

    Both retriever and generator are injected. The optional reranker
    is also injected and should be reused across requests. If retrieval
    yields no usable context, generation is skipped.
    """

    result = retrieve_context(
        retriever=retriever,
        query=query,
        top_k=top_k,
        max_chars=max_chars,
        reranker=reranker,
        fallback_on_reranker_error=fallback_on_reranker_error,
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
