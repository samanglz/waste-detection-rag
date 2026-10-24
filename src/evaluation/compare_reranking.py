"""Compare Hybrid retrieval with Hybrid + Cross-Encoder reranking."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.evaluation.query_loader import load_evaluation_queries
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.retriever_factory import load_hybrid_retriever


def _query_text(item: dict) -> str:
    query = item.get("query")
    if not query:
        query = item.get("question")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Each evaluation item must contain a non-empty 'query' "
            "or 'question' field."
        )
    return query


def _latency_report(retriever: Any, queries: list[dict], top_k: int) -> dict:
    # Warm-up: avoid including initial model setup in the measurements.
    if queries:
        retriever.search(_query_text(queries[0]), top_k=top_k)

    durations_ms = []
    for item in queries:
        query = _query_text(item)
        started = time.perf_counter()
        retriever.search(query, top_k=top_k)
        durations_ms.append((time.perf_counter() - started) * 1000)

    if not durations_ms:
        return {
            "query_count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
        }

    ordered = sorted(durations_ms)
    p95_index = min(
        len(ordered) - 1,
        max(0, int(0.95 * len(ordered) + 0.999999) - 1),
    )

    return {
        "query_count": len(durations_ms),
        "mean_ms": round(statistics.mean(durations_ms), 2),
        "median_ms": round(statistics.median(durations_ms), 2),
        "p95_ms": round(ordered[p95_index], 2),
    }


def compare_reranking(
    index_dir: str,
    queries_path: str,
    k: int = 5,
    candidate_k: int = 50,
    embedding_model: str = "intfloat/multilingual-e5-small",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    batch_size: int = 16,
) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    if candidate_k < k:
        raise ValueError("candidate_k must be >= k")

    queries = load_evaluation_queries(queries_path)
    if not queries:
        raise ValueError("The evaluation dataset is empty.")

    encoder = TextEncoder(model_name=embedding_model)
    hybrid = load_hybrid_retriever(
        index_dir=index_dir,
        encoder=encoder,
    )

    reranked = CrossEncoderReranker(
        base_retriever=hybrid,
        model_name=reranker_model,
        candidate_k=candidate_k,
        batch_size=batch_size,
    )

    # Both systems are evaluated against exactly the same labeled queries.
    hybrid_metrics = evaluate_retrieval(hybrid, queries, k=k)
    reranked_metrics = evaluate_retrieval(reranked, queries, k=k)

    hybrid_latency = _latency_report(hybrid, queries, top_k=k)
    reranked_latency = _latency_report(reranked, queries, top_k=k)

    labeled_count = sum(
        bool(
            item.get("relevant_chunks")
            or item.get("relevant_sources")
            or item.get("relevant_source")
        )
        for item in queries
    )

    return {
        "experiment": "hybrid_vs_hybrid_cross_encoder_reranking",
        "index_dir": index_dir,
        "queries_path": queries_path,
        "query_count": len(queries),
        "labeled_query_count": labeled_count,
        "k": k,
        "candidate_k": candidate_k,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "metrics": {
            "Hybrid": hybrid_metrics,
            "Hybrid+Reranker": reranked_metrics,
        },
        "latency_ms_per_query": {
            "Hybrid": hybrid_latency,
            "Hybrid+Reranker": reranked_latency,
        },
        "latency_note": (
            "End-to-end search latency measured after one warm-up query; "
            "includes retrieval and, for the reranked system, reranking. "
            "Results depend on runtime, hardware, and model caching."
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-dir",
        default="data/processed/retrieval_index_v4_expanded",
    )
    parser.add_argument(
        "--queries",
        default="data/evaluation/queries_v4_expanded.json",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--output",
        default="reports/evaluation/reranking_comparison_v4_expanded_k5.json",
    )
    args = parser.parse_args()

    report = compare_reranking(
        index_dir=args.index_dir,
        queries_path=args.queries,
        k=args.k,
        candidate_k=args.candidate_k,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report: {output_path}")
