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


def _get_evaluation_unit(
    queries: list[dict[str, Any]],
) -> str:
    units = set()

    for item in queries:
        if "relevant_chunks" in item:
            units.add("chunk")
        elif "relevant_sources" in item or "relevant_source" in item:
            units.add("source")

    if len(units) == 1:
        return units.pop()

    return "mixed"


def compare_retrievers(
    index_dir: str | Path,
    queries_path: str | Path,
    k: int = 3,
    model_name: str = "intfloat/multilingual-e5-small",
) -> dict[str, dict[str, float]]:
    """Evaluate Dense, BM25, and Hybrid on the same ground truth."""
    if k <= 0:
        raise ValueError("k must be positive")

    queries = load_evaluation_queries(queries_path)

    evaluation_unit = _get_evaluation_unit(queries)
    if evaluation_unit == "mixed":
        raise ValueError(
            "Cannot compare mixed evaluation units. Use either "
            "relevant_chunks or source-level ground truth consistently."
        )

    vector_index, chunks = IndexStore(index_dir).load()

    relevant_chunk_ids = {
        chunk["chunk_id"] for chunk in chunks
    }
    relevant_sources = {
        chunk["source"] for chunk in chunks
    }

    for position, item in enumerate(queries):
        if "relevant_chunks" in item:
            unknown = set(item["relevant_chunks"]) - relevant_chunk_ids
            if unknown:
                raise ValueError(
                    f"Query at position {position} references unknown "
                    f"chunk IDs: {sorted(unknown)}"
                )
        else:
            sources = (
                item["relevant_sources"]
                if "relevant_sources" in item
                else [item["relevant_source"]]
            )
            unknown = set(sources) - relevant_sources
            if unknown:
                raise ValueError(
                    f"Query at position {position} references unknown "
                    f"sources: {sorted(unknown)}"
                )

    encoder = TextEncoder(model_name=model_name)
    bm25 = BM25Retriever(chunks)

    retrievers = {
        "Dense": DenseRetriever(
            chunks=chunks,
            encoder=encoder,
            vector_index=vector_index,
        ),
        "BM25": BM25SearchAdapter(
            retriever=bm25,
            chunks=chunks,
        ),
        "Hybrid": HybridRetriever(
            chunks=chunks,
            encoder=encoder,
            vector_index=vector_index,
            bm25_retriever=bm25,
        ),
    }

    return {
        name: evaluate_retrieval(
            retriever=retriever,
            queries=queries,
            k=k,
        )
        for name, retriever in retrievers.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Dense, BM25, and Hybrid retrieval."
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
        default="reports/retrieval_comparison_v3.json",
    )
    args = parser.parse_args()

    queries = load_evaluation_queries(args.queries)
    results = compare_retrievers(
        index_dir=args.index_dir,
        queries_path=args.queries,
        k=args.k,
        model_name=args.model,
    )

    report = {
        "k": args.k,
        "query_count": len(queries),
        "evaluation_unit": _get_evaluation_unit(queries),
        "metrics": results,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
