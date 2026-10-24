"""Cross-encoder reranking wrapper for an existing retriever."""

from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    """Retrieve a larger candidate pool, then reorder it with a cross-encoder.

    The wrapped retriever remains unchanged. This class exposes the same
    search(query, top_k=...) interface used by the RAG pipeline.
    """

    def __init__(
        self,
        base_retriever: Any,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        candidate_k: int = 50,
        batch_size: int = 16,
        model: Any = None,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.base_retriever = base_retriever
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.batch_size = batch_size

        # Inject a fake/custom model in tests; otherwise load the real model.
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoder requires sentence-transformers. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            model = CrossEncoder(model_name)

        self.model = model

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top-k candidates ordered by cross-encoder score."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        candidates = self.base_retriever.search(
            query,
            top_k=max(self.candidate_k, top_k),
        )

        if not candidates:
            return []

        pairs = [
            (query, str(item.get("text", "")))
            for item in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # Support scalar scores as well as ordinary lists/arrays.
        if isinstance(scores, (int, float)):
            scores = [scores]

        if len(scores) != len(candidates):
            raise ValueError(
                "Cross-encoder returned a different number of scores "
                "than the number of candidates."
            )

        reranked = []
        for item, score in zip(candidates, scores):
            result = dict(item)
            result["original_score"] = item.get("score")
            result["rerank_score"] = float(score)
            result["score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        output = []
        for rank, item in enumerate(reranked[:top_k], start=1):
            item["rank"] = rank
            output.append(item)

        return output
