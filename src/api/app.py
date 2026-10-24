from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.generation.rag_pipeline import answer_question


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    max_chars: int = Field(default=6000, ge=100, le=20000)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    retrieval_method: str
    reranker_error: str | None = None
    citation_validation: Any = None


def _build_components() -> dict[str, Any]:
    """Load heavyweight models and clients once per application process."""

    from src.embeddings.encoder import TextEncoder
    from src.retrieval.retriever_factory import load_hybrid_retriever
    from src.reranking.cross_encoder_reranker import CrossEncoderReranker
    from src.generation.gemini_generator import GeminiAnswerGenerator

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is required. Set it in the environment or .env file."
        )

    index_dir = Path(
        os.getenv(
            "RAG_INDEX_DIR",
            "data/processed/retrieval_index_v4_expanded",
        )
    )
    if not index_dir.is_absolute():
        index_dir = PROJECT_ROOT / index_dir

    if not index_dir.is_dir():
        raise FileNotFoundError(f"RAG index directory not found: {index_dir}")

    encoder = TextEncoder(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "intfloat/multilingual-e5-small",
        )
    )

    retriever = load_hybrid_retriever(
        index_dir=str(index_dir),
        encoder=encoder,
    )

    reranker = CrossEncoderReranker(
        base_retriever=retriever,
        model_name=os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ),
        candidate_k=int(os.getenv("RERANKER_CANDIDATE_K", "20")),
        batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "16")),
    )

    generator = GeminiAnswerGenerator(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    )

    return {
        "retriever": retriever,
        "reranker": reranker,
        "generator": generator,
    }


def create_app(
    components: dict[str, Any] | None = None,
) -> FastAPI:
    """Create the API. Components can be injected for isolated tests."""

    injected = dict(components or {})
    required = {"retriever", "reranker", "generator"}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if required.issubset(injected):
            loaded = injected
        else:
            # Production path: initialize once at startup.
            loaded = _build_components()

        for key in required:
            if key not in loaded or loaded[key] is None:
                raise RuntimeError(f"Missing application component: {key}")

        app.state.retriever = loaded["retriever"]
        app.state.reranker = loaded["reranker"]
        app.state.generator = loaded["generator"]

        yield

    app = FastAPI(
        title="Waste Detection RAG API",
        version="1.0.0",
        description="Grounded recycling knowledge retrieval and question answering.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> dict[str, Any]:
        if not payload.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")

        try:
            result = answer_question(
                retriever=request.app.state.retriever,
                reranker=request.app.state.reranker,
                generator=request.app.state.generator,
                query=payload.query,
                top_k=payload.top_k,
                max_chars=payload.max_chars,
                fallback_on_reranker_error=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            # Do not expose exception details or credentials to API clients.
            raise HTTPException(
                status_code=502,
                detail="RAG processing failed. Check the server logs.",
            ) from exc

        return {
            "query": result["query"],
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieval_method": result["retrieval_method"],
            "reranker_error": result.get("reranker_error"),
            "citation_validation": result.get("citation_validation"),
        }

    return app


app = create_app()
