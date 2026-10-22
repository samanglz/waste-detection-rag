import argparse
import json
from pathlib import Path
from typing import Any

from src.embeddings.encoder import TextEncoder
from src.evaluation.query_loader import load_evaluation_queries
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.indexing.index_store import IndexStore
from src.retrieval.bm25_adapter import BM25SearchAdapter
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def get_ground_truth(
    item: dict[str, Any],
) -> tuple[str, list[str]]:
    """Return the result identifier key and relevant identifiers."""
    if "relevant_chunks" in item:
        return "chunk_id", item["relevant_chunks"]

    if "relevant_sources" in item:
        return "source", item["relevant_sources"]

    if "relevant_source" in item:
        return "source", [item["relevant_source"]]

    raise ValueError(
        "Each query must contain relevant_chunks, "
        "relevant_sources, or relevant_source"
    )


def build_retrievers(
    index_dir: str | Path,
    model_name: str = "intfloat/multilingual-e5-small",
):
    """Build all retrievers using the same chunks and vector index."""
    vector_index, chunks = IndexStore(index_dir).load()
    encoder = TextEncoder(model_name=model_name)
    bm25 = BM25Retriever(chunks)

    retrievers = {
        "Dense": DenseRetriever(chunks, encoder, vector_index),
        "BM25": BM25SearchAdapter(bm25, chunks),
        "Hybrid": HybridRetriever(
            chunks=chunks,
            encoder=encoder,
            vector_index=vector_index,
            bm25_retriever=bm25,
        ),
    }

    return retrievers, chunks


def generate_diagnostics(
    index_dir: str | Path,
    queries_path: str | Path,
    k: int = 3,
    model_name: str = "intfloat/multilingual-e5-small",
) -> dict:
    """Create chunk/source-level rankings and aggregate metrics."""
    if k <= 0:
        raise ValueError("k must be positive")

    queries = load_evaluation_queries(queries_path)

    units = {
        "chunk" if "relevant_chunks" in item else "source"
        for item in queries
    }
    if len(units) != 1:
        raise ValueError(
            "All queries must use the same evaluation unit: "
            "either relevant_chunks or source-level ground truth."
        )

    evaluation_unit = units.pop()
    identifier_key = (
        "chunk_id" if evaluation_unit == "chunk" else "source"
    )

    retrievers, chunks = build_retrievers(index_dir, model_name)

    available_ids = (
        {chunk["chunk_id"] for chunk in chunks}
        if evaluation_unit == "chunk"
        else {chunk["source"] for chunk in chunks}
    )

    for query_id, item in enumerate(queries, start=1):
        _, relevant_ids = get_ground_truth(item)
        unknown = set(relevant_ids) - available_ids
        if unknown:
            raise ValueError(
                f"Query {query_id} has unknown {identifier_key} "
                f"values: {sorted(unknown)}"
            )

    diagnostics = []

    for query_id, item in enumerate(queries, start=1):
        _, relevant_ids = get_ground_truth(item)
        relevant_set = set(relevant_ids)
        method_results = {}

        for method_name, retriever in retrievers.items():
            results = retriever.search(item["query"], top_k=k)
            ranked_results = []
            retrieved_relevant = set()

            for position, result in enumerate(results, start=1):
                identifier = result.get(identifier_key)
                is_relevant = identifier in relevant_set

                if is_relevant:
                    retrieved_relevant.add(identifier)

                ranked_results.append({
                    "rank": result.get("rank", position),
                    "chunk_id": result["chunk_id"],
                    "source": result["source"],
                    "score": float(result["score"]),
                    "relevant": is_relevant,
                })

            relevant_ranks = [
                result["rank"]
                for result in ranked_results
                if result["relevant"]
            ]

            first_relevant_rank = (
                min(relevant_ranks) if relevant_ranks else None
            )

            method_results[method_name] = {
                "hit": first_relevant_rank is not None,
                "first_relevant_rank": first_relevant_rank,
                "retrieved_relevant_ids": sorted(retrieved_relevant),
                "missed_relevant_ids": sorted(
                    relevant_set - retrieved_relevant
                ),
                "results": ranked_results,
            }

        diagnostics.append({
            "query_id": query_id,
            "query": item["query"],
            "evaluation_unit": evaluation_unit,
            "relevant_ids": sorted(relevant_set),
            "methods": method_results,
        })

    metrics = {
        name: evaluate_retrieval(retriever, queries, k=k)
        for name, retriever in retrievers.items()
    }

    return {
        "k": k,
        "query_count": len(queries),
        "chunk_count": len(chunks),
        "evaluation_unit": evaluation_unit,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect per-query Dense, BM25, and Hybrid rankings."
    )
    parser.add_argument(
        "--index-dir",
        default="data/processed/retrieval_index_v3",
    )
    parser.add_argument(
        "--queries",
        default="data/evaluation/queries_v3.json",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
    )
    parser.add_argument(
        "--output",
        default="reports/retrieval_diagnostics_v3.json",
    )
    args = parser.parse_args()

    report = generate_diagnostics(
        index_dir=args.index_dir,
        queries_path=args.queries,
        k=args.k,
        model_name=args.model,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Queries evaluated: {report['query_count']}")
    print(f"Chunks in index: {report['chunk_count']}")
    print(f"Evaluation unit: {report['evaluation_unit']}")
    print(f"Results per query: up to {report['k']}")

    print("\nAggregate metrics:")
    for method, values in report["metrics"].items():
        print(
            f"{method:>7} | "
            f"Hit Rate@{report['k']}: {values['Hit Rate@k']:.3f} | "
            f"Recall@{report['k']}: {values['Recall@k']:.3f} | "
            f"MRR@{report['k']}: {values['MRR@k']:.3f}"
        )

    print(f"\nDetailed report saved to: {output_path}")

    print("\nQueries with missed relevant chunks:")
    for item in report["diagnostics"]:
        methods_with_misses = {
            method: data["missed_relevant_ids"]
            for method, data in item["methods"].items()
            if data["missed_relevant_ids"]
        }

        if methods_with_misses:
            print(f"\nQ{item['query_id']}: {item['query']}")
            print(f"  Expected IDs: {item['relevant_ids']}")

            for method, missed_ids in methods_with_misses.items():
                print(f"  {method} missed: {missed_ids}")

    print("\nDetailed rankings:")
    for item in report["diagnostics"]:
        print(f"\nQ{item['query_id']}: {item['query']}")
        print(f"  Relevant IDs: {item['relevant_ids']}")

        for method, data in item["methods"].items():
            first_rank = data["first_relevant_rank"]
            rank_label = (
                str(first_rank) if first_rank is not None else "MISS"
            )
            print(f"  {method} | first relevant rank: {rank_label}")

            for result in data["results"]:
                marker = "✓" if result["relevant"] else " "
                print(
                    f"    {marker} rank={result['rank']} "
                    f"chunk={result['chunk_id']} "
                    f"source={result['source']} "
                    f"score={result['score']:.4f}"
                )


if __name__ == "__main__":
    main()
