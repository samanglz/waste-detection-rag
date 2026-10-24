"""Diagnostic retest of Hybrid retrieval with Cross-Encoder reranking.

This is an experimental entry point. It does not modify the RAG pipeline,
the existing retriever, evaluation labels, or retrieval indexes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from src.embeddings.encoder import TextEncoder
from src.evaluation.query_loader import load_evaluation_queries
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.retrieval.retriever_factory import load_hybrid_retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_latency(retriever, queries, k: int, device: str) -> dict:
    if queries:
        retriever.search(queries[0]["query"], top_k=k)
    _sync(device)

    durations = []
    for item in queries:
        _sync(device)
        start = time.perf_counter()
        retriever.search(item["query"], top_k=k)
        _sync(device)
        durations.append((time.perf_counter() - start) * 1000)

    if not durations:
        return {
            "query_count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
        }

    ordered = sorted(durations)
    p95_index = min(
        len(ordered) - 1,
        max(0, int(0.95 * len(ordered) + 0.999999) - 1),
    )

    return {
        "query_count": len(durations),
        "mean_ms": round(statistics.mean(durations), 2),
        "median_ms": round(statistics.median(durations), 2),
        "p95_ms": round(ordered[p95_index], 2),
        "min_ms": round(min(durations), 2),
        "max_ms": round(max(durations), 2),
    }


def run_experiment(
    index_dir: str,
    queries_path: str,
    k: int,
    candidate_k: int,
    embedding_model: str,
    reranker_model: str,
    batch_size: int,
    device: str,
) -> dict:
    if k < 1:
        raise ValueError("k must be >= 1")
    if candidate_k < k:
        raise ValueError("candidate_k must be >= k")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Enable a GPU runtime."
        )

    queries = load_evaluation_queries(queries_path)
    if not queries:
        raise ValueError("Evaluation dataset is empty.")

    for position, item in enumerate(queries, start=1):
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"Query {position} has an invalid query field.")
        if not isinstance(item.get("relevant_chunks"), list):
            raise ValueError(
                f"Query {position} must contain a relevant_chunks list."
            )
        if not item["relevant_chunks"]:
            raise ValueError(f"Query {position} has no relevant chunk labels.")

    encoder = TextEncoder(model_name=embedding_model)
    hybrid = load_hybrid_retriever(
        index_dir=index_dir,
        encoder=encoder,
    )

    cross_encoder = CrossEncoder(
        reranker_model,
        device=device,
    )
    reranked = CrossEncoderReranker(
        base_retriever=hybrid,
        model_name=reranker_model,
        candidate_k=candidate_k,
        batch_size=batch_size,
        model=cross_encoder,
    )

    # Warm-up the cross-encoder before timing.
    cross_encoder.predict(
        [("recycling", "Recycling recovers useful materials.")],
        batch_size=1,
        show_progress_bar=False,
    )
    _sync(device)

    # Directly check candidate coverage and top-k chunk-ID overlap.
    diagnostic_rows = []
    covered_queries = 0
    hybrid_hit_queries = 0
    reranked_hit_queries = 0

    for position, item in enumerate(queries, start=1):
        query = item["query"]
        relevant = set(item["relevant_chunks"])

        candidates = hybrid.search(query, top_k=candidate_k)
        hybrid_topk = candidates[:k]
        reranked_topk = reranked.search(query, top_k=k)

        candidate_ids = {r.get("chunk_id") for r in candidates}
        hybrid_ids = {r.get("chunk_id") for r in hybrid_topk}
        reranked_ids = {r.get("chunk_id") for r in reranked_topk}

        covered = relevant.intersection(candidate_ids)
        hybrid_matches = relevant.intersection(hybrid_ids)
        reranked_matches = relevant.intersection(reranked_ids)

        if relevant.issubset(candidate_ids):
            covered_queries += 1
        if hybrid_matches:
            hybrid_hit_queries += 1
        if reranked_matches:
            reranked_hit_queries += 1

        diagnostic_rows.append({
            "query_number": position,
            "query": query,
            "relevant_chunk_count": len(relevant),
            "candidate_relevant_chunk_count": len(covered),
            "all_relevant_chunks_in_candidates": relevant.issubset(candidate_ids),
            "hybrid_top_k_chunk_ids": [
                r.get("chunk_id") for r in hybrid_topk
            ],
            "hybrid_matched_chunk_ids": sorted(hybrid_matches),
            "reranked_top_k_chunk_ids": [
                r.get("chunk_id") for r in reranked_topk
            ],
            "reranked_matched_chunk_ids": sorted(reranked_matches),
        })

    # Use the official evaluator for comparable metrics.
    hybrid_metrics = evaluate_retrieval(hybrid, queries, k=k)
    reranked_metrics = evaluate_retrieval(reranked, queries, k=k)

    hybrid_latency = _measure_latency(hybrid, queries, k, device)
    reranked_latency = _measure_latency(reranked, queries, k, device)

    mean_ratio = None
    if hybrid_latency["mean_ms"]:
        mean_ratio = round(
            reranked_latency["mean_ms"] / hybrid_latency["mean_ms"], 2
        )

    return {
        "experiment": "v4_expanded_reranking_retest",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "index_dir": index_dir,
        "queries_path": queries_path,
        "query_count": len(queries),
        "k": k,
        "candidate_k": candidate_k,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "candidate_coverage": {
            "queries_with_all_relevant_chunks_in_candidates": covered_queries,
            "total_queries": len(queries),
        },
        "manual_hit_rate": {
            "Hybrid": hybrid_hit_queries / len(queries),
            "Hybrid+Reranker": reranked_hit_queries / len(queries),
        },
        "metrics": {
            "Hybrid": hybrid_metrics,
            "Hybrid+Reranker": reranked_metrics,
        },
        "latency_ms_per_query": {
            "Hybrid": hybrid_latency,
            "Hybrid+Reranker": reranked_latency,
        },
        "mean_latency_ratio_reranker_over_hybrid": mean_ratio,
        "diagnostics_per_query": diagnostic_rows,
        "latency_note": (
            "Measured after model loading and warm-up. GPU synchronization "
            "is used when CUDA is selected. Latency depends on runtime and "
            "hardware."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a diagnostic Hybrid vs Cross-Encoder retest."
    )
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
        "--embedding-model",
        default="intfloat/multilingual-e5-small",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_experiment(
        index_dir=args.index_dir,
        queries_path=args.queries,
        k=args.k,
        candidate_k=args.candidate_k,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("reports/evaluation") / (
            f"reranking_retest_v4_expanded_k{args.k}_{timestamp}.json"
        )

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== CANDIDATE COVERAGE ===")
    print(json.dumps(report["candidate_coverage"], indent=2))
    print("\n=== MANUAL HIT RATE ===")
    print(json.dumps(report["manual_hit_rate"], indent=2))
    print("\n=== RETRIEVAL METRICS ===")
    print(json.dumps(report["metrics"], indent=2))
    print("\n=== LATENCY ===")
    print(json.dumps(report["latency_ms_per_query"], indent=2))
    print(
        "\nMean latency ratio (reranker / hybrid):",
        report["mean_latency_ratio_reranker_over_hybrid"],
    )
    print("\n=== PER-QUERY DIAGNOSTICS ===")
    for row in report["diagnostics_per_query"]:
        print(
            f"Q{row['query_number']:02d}: "
            f"candidate coverage "
            f"{row['candidate_relevant_chunk_count']}/"
            f"{row['relevant_chunk_count']} | "
            f"Hybrid matches {len(row['hybrid_matched_chunk_ids'])} | "
            f"Reranker matches {len(row['reranked_matched_chunk_ids'])}"
        )
    print("\nSaved report:", output_path)


if __name__ == "__main__":
    main()
