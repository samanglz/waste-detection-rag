from collections.abc import Mapping
from math import log2


def evaluate_retrieval(
    retriever,
    queries: list[dict],
    k: int = 3,
) -> dict[str, float]:
    """Evaluate retrieval using Hit Rate, Recall, MRR, and nDCG at k.

    Ground truth supports relevant_chunks, relevant_sources, and
    relevant_source. Relevance is binary.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    if not queries:
        raise ValueError("queries must not be empty")

    hits = 0
    reciprocal_ranks = []
    recalls = []
    ndcgs = []

    for query_position, item in enumerate(queries):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Query at position {query_position} must be a mapping"
            )

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Query at position {query_position} must contain "
                "a non-empty query string"
            )

        if "relevant_chunks" in item:
            identifier_key = "chunk_id"
            relevant_items = item["relevant_chunks"]
        elif "relevant_sources" in item:
            identifier_key = "source"
            relevant_items = item["relevant_sources"]
        elif "relevant_source" in item:
            identifier_key = "source"
            relevant_item = item["relevant_source"]
            relevant_items = (
                [relevant_item] if isinstance(relevant_item, str) else []
            )
        else:
            raise ValueError(
                f"Query at position {query_position} has no relevance labels"
            )

        if (
            not isinstance(relevant_items, list)
            or not relevant_items
            or not all(
                isinstance(value, str) and value.strip()
                for value in relevant_items
            )
        ):
            raise ValueError(
                f"Query at position {query_position} has invalid relevance labels"
            )

        relevant_set = set(relevant_items)
        results = retriever.search(query, top_k=k)

        if not isinstance(results, list):
            raise TypeError("retriever.search() must return a list")

        retrieved_relevant = set()
        first_relevant_rank = None
        dcg = 0.0
        seen_identifiers = set()

        for rank, result in enumerate(results[:k], start=1):
            if not isinstance(result, Mapping):
                raise TypeError("Each retrieval result must be a mapping")

            identifier = result.get(identifier_key)

            # Count a chunk/document only once.
            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)

            if identifier not in relevant_set:
                continue

            retrieved_relevant.add(identifier)

            if first_relevant_rank is None:
                first_relevant_rank = rank

            dcg += 1.0 / log2(rank + 1)

        if retrieved_relevant:
            hits += 1

        reciprocal_ranks.append(
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        recalls.append(len(retrieved_relevant) / len(relevant_set))

        ideal_relevant_count = min(len(relevant_set), k)
        ideal_dcg = sum(
            1.0 / log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)

    query_count = len(queries)

    return {
        "Hit Rate@k": hits / query_count,
        "Recall@k": sum(recalls) / query_count,
        "MRR@k": sum(reciprocal_ranks) / query_count,
        "nDCG@k": sum(ndcgs) / query_count,
    }
