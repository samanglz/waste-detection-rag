from src.retrieval.bm25_retriever import BM25Retriever


class BM25SearchAdapter:
    """Adapt BM25 tuple results to the shared retrieval result format."""

    def __init__(
        self,
        retriever: BM25Retriever,
        chunks: list[dict],
    ):
        if not chunks:
            raise ValueError("chunks must not be empty")

        if len(chunks) != len(retriever.documents):
            raise ValueError(
                "Chunk count must match BM25 document count"
            )

        self.retriever = retriever
        self.chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return ranked BM25 results in the shared retrieval format."""

        results = self.retriever.search(query, top_k=top_k)
        normalized_results = []

        for rank, (index, score) in enumerate(results, start=1):
            chunk = self.chunks[index]

            normalized_results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}).copy(),
                    "score": float(score),
                    "rank": rank,
                }
            )

        return normalized_results
