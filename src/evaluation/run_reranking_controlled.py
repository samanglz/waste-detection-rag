from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.retriever_factory import load_hybrid_retriever


class CachedResultsRetriever:
    """Return previously cached retrieval results without searching again."""

    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]]):
        self.results_by_query = results_by_query

    def search(self, query: str, top_k: int = 5):
        if query not in self.results_by_query:
            raise KeyError(f"No cached results for query: {query!r}")
        return self.results_by_query[query][:top_k]


class FrozenCandidatesRetriever:
    """Expose the same cached candidate pool to the cross-encoder."""

    def __init__(self, candidates_by_query: dict[str, list[dict[str, Any]]]):
        self.candidates_by_query = candidates_by_query

    def search(self, query: str, top_k: int = 5):
        if query not in self.candidates_by_query:
            raise KeyError(f"No cached candidates for query: {query!r}")
        return self.candidates_by_query[query][:top_k]


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)

    return {
        "count": len(ordered),
        "mean_ms": statistics.mean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        queries = payload
    elif isinstance(payload, dict):
        queries = payload.get("queries")
        if not isinstance(queries, list):
            raise ValueError(
                "Query JSON must be a list or a dict containing a 'queries' list."
            )
    else:
        raise ValueError("Unsupported query JSON structure.")

    if not queries:
        raise ValueError("The query dataset is empty.")

    seen = set()
    for i, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(f"Query item {i} is not a JSON object.")

        query = item.get("query")
        relevant = item.get("relevant_chunks")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query item {i} has no valid 'query' string.")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Query {query!r} has no non-empty 'relevant_chunks' list."
            )
        if query in seen:
            raise ValueError(
                f"Duplicate query text found. This script keys caches by query: {query!r}"
            )
        seen.add(query)

    return queries


def relevant_ids(item: dict[str, Any]) -> set[str]:
    return {str(value) for value in item["relevant_chunks"]}


def retrieved_ids(results: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["chunk_id"])
        for item in results
        if item.get("chunk_id") is not None
    }


def manual_diagnostics(
    queries: list[dict[str, Any]],
    results_by_query: dict[str, list[dict[str, Any]]],
    k: int,
) -> tuple[float, list[dict[str, Any]]]:
    details = []
    hit_count = 0

    for item in queries:
        query = item["query"]
        relevant = relevant_ids(item)
        results = results_by_query[query][:k]
        retrieved = retrieved_ids(results)
        matches = relevant & retrieved

        if matches:
            hit_count += 1

        details.append({
            "query": query,
            "relevant_count": len(relevant),
            "retrieved_count": len(results),
            "matches_count": len(matches),
            "matched_chunk_ids": sorted(matches),
            "retrieved_chunk_ids": [
                str(result.get("chunk_id"))
                for result in results
            ],
        })

    return hit_count / len(queries), details


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def run(args: argparse.Namespace) -> Path:
    project_root = Path.cwd().resolve()
    index_dir = (project_root / args.index_dir).resolve()
    queries_path = (project_root / args.queries).resolve()

    if not (project_root / "src/embeddings/encoder.py").is_file():
        raise FileNotFoundError(
            "Run this module from the Google Drive project root. "
            f"Current directory: {project_root}"
        )
    if not index_dir.is_dir():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    if not queries_path.is_file():
        raise FileNotFoundError(f"Query file not found: {queries_path}")
    if args.k < 1 or args.candidate_k < args.k:
        raise ValueError("Require k >= 1 and candidate-k >= k.")

    queries = load_queries(queries_path)
    query_texts = [item["query"] for item in queries]

    print(f"Project root: {project_root}")
    print(f"Index: {index_dir}")
    print(f"Queries: {queries_path}")
    print(f"Number of queries: {len(queries)}")
    print("Loading encoder and hybrid retriever...")

    encoder = TextEncoder(model_name=args.embedding_model)
    hybrid = load_hybrid_retriever(
        index_dir=str(index_dir),
        encoder=encoder,
    )

    # Retrieve the candidate pool exactly once per query and cache it.
    candidates_by_query: dict[str, list[dict[str, Any]]] = {}
    candidate_retrieval_ms: list[float] = []

    print(f"\nCaching Hybrid candidate pools (candidate_k={args.candidate_k})...")
    for i, query in enumerate(query_texts, start=1):
        start = time.perf_counter()
        candidates = hybrid.search(query, top_k=args.candidate_k)
        elapsed = (time.perf_counter() - start) * 1000

        candidates_by_query[query] = list(candidates)
        candidate_retrieval_ms.append(elapsed)

        print(
            f"  [{i}/{len(query_texts)}] "
            f"candidates={len(candidates)}; latency={elapsed:.2f} ms"
        )

    # The Hybrid baseline uses the first k results from those same cached pools.
    hybrid_top_k = {
        query: candidates_by_query[query][:args.k]
        for query in query_texts
    }

    # Measure ordinary live Hybrid top-k latency separately.
    hybrid_live_top_k_ms: list[float] = []
    print("\nMeasuring live Hybrid top-k latency...")
    for query in query_texts:
        start = time.perf_counter()
        hybrid.search(query, top_k=args.k)
        hybrid_live_top_k_ms.append((time.perf_counter() - start) * 1000)

    frozen_candidates = FrozenCandidatesRetriever(candidates_by_query)

    print(f"\nLoading cross-encoder: {args.reranker_model}")
    reranker = CrossEncoderReranker(
        base_retriever=frozen_candidates,
        model_name=args.reranker_model,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
    )

    # Warm up the cross-encoder before collecting per-query timings.
    print("Warming up the cross-encoder...")
    reranker.search(query_texts[0], top_k=args.k)
    sync_cuda()

    reranked_by_query: dict[str, list[dict[str, Any]]] = {}
    reranker_ms: list[float] = []

    print("\nReranking cached candidates...")
    for i, query in enumerate(query_texts, start=1):
        sync_cuda()
        start = time.perf_counter()
        results = reranker.search(query, top_k=args.candidate_k)
        sync_cuda()
        elapsed = (time.perf_counter() - start) * 1000

        reranked_by_query[query] = list(results)
        reranker_ms.append(elapsed)

        print(
            f"  [{i}/{len(query_texts)}] "
            f"reranked={len(results)}; latency={elapsed:.2f} ms"
        )

    # Evaluate the cached results, not fresh retrieval calls.
    hybrid_metrics = evaluate_retrieval(
        CachedResultsRetriever(hybrid_top_k),
        queries,
        k=args.k,
    )
    reranker_metrics = evaluate_retrieval(
        CachedResultsRetriever(reranked_by_query),
        queries,
        k=args.k,
    )

    hybrid_manual_hit, hybrid_details = manual_diagnostics(
        queries, hybrid_top_k, args.k
    )
    reranker_manual_hit, reranker_details = manual_diagnostics(
        queries, reranked_by_query, args.k
    )

    # Ensure evaluator and manual hit-rate calculations use the same results.
    if abs(float(hybrid_metrics["Hit Rate@k"]) - hybrid_manual_hit) > 1e-9:
        raise AssertionError(
            "Hybrid Hit Rate mismatch between evaluator and cached diagnostics."
        )
    if abs(float(reranker_metrics["Hit Rate@k"]) - reranker_manual_hit) > 1e-9:
        raise AssertionError(
            "Reranker Hit Rate mismatch between evaluator and cached diagnostics."
        )

    candidate_hits = 0
    for item in queries:
        query = item["query"]
        candidate_ids = retrieved_ids(candidates_by_query[query])
        if relevant_ids(item) & candidate_ids:
            candidate_hits += 1

    candidate_coverage = candidate_hits / len(queries)

    hybrid_candidate_mean = statistics.mean(candidate_retrieval_ms)
    reranker_mean = statistics.mean(reranker_ms)

    report = {
        "experiment": "controlled_cached_candidates_reranking",
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(project_root),
        "index_dir": str(index_dir),
        "queries_path": str(queries_path),
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "query_count": len(queries),
        "k": args.k,
        "candidate_k": args.candidate_k,
        "batch_size": args.batch_size,
        "candidate_coverage": {
            "hits": candidate_hits,
            "total_queries": len(queries),
            "rate": candidate_coverage,
        },
        "metrics": {
            "Hybrid": json_safe(hybrid_metrics),
            "Hybrid+Reranker": json_safe(reranker_metrics),
        },
        "manual_hit_rate": {
            "Hybrid": hybrid_manual_hit,
            "Hybrid+Reranker": reranker_manual_hit,
        },
        "latency": {
            "Hybrid_live_top_k": latency_summary(hybrid_live_top_k_ms),
            "Hybrid_candidate_pool": latency_summary(candidate_retrieval_ms),
            "Reranker_on_cached_candidates": latency_summary(reranker_ms),
            "estimated_full_hybrid_plus_reranker_mean_ms": (
                hybrid_candidate_mean + reranker_mean
            ),
            "note": (
                "Reranker_on_cached_candidates excludes live candidate retrieval. "
                "The estimated full pipeline mean adds the mean candidate-retrieval "
                "latency to the mean reranking latency."
            ),
        },
        "per_query": {
            "Hybrid": hybrid_details,
            "Hybrid+Reranker": reranker_details,
        },
        # Persist all candidates and their reranker scores for offline analysis.
        "cached_rankings": {
            "candidate_pools": {
                query: json_safe(candidates_by_query[query])
                for query in query_texts
            },
            "reranked_pools": {
                query: json_safe(reranked_by_query[query])
                for query in query_texts
            },
        },
    }

    if args.output:
        output_path = (project_root / args.output).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            project_root
            / "reports/evaluation"
            / f"reranking_controlled_v4_expanded_k{args.k}_{stamp}.json"
        )

    # Prevent accidental writes outside the Google Drive project root.
    if not output_path.is_relative_to(project_root):
        raise ValueError(f"Report path must be inside the project: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("METRICS FROM CACHED RESULTS")
    print("=" * 72)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print("\nManual Hit Rate:")
    print(json.dumps(report["manual_hit_rate"], ensure_ascii=False, indent=2))
    print("\nCandidate coverage:")
    print(json.dumps(report["candidate_coverage"], ensure_ascii=False, indent=2))
    print("\nLatency:")
    print(json.dumps(report["latency"], ensure_ascii=False, indent=2))
    print("\nPer-query diagnostics:")
    for method, details in report["per_query"].items():
        print(f"\n{method}")
        for detail in details:
            print(
                f"{detail['query']}: "
                f"matches={detail['matches_count']}, "
                f"matched_ids={detail['matched_chunk_ids']}"
            )

    print(f"\nReport saved to Google Drive: {output_path}")
    print(f"Report exists: {output_path.is_file()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled reranker evaluation using cached retrieval results."
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-small",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
