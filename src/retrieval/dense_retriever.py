from src.embeddings.encoder import TextEncoder
from src.indexing.faiss_index import FaissIndex


class DenseRetriever:
    """Semantic retrieval using text embeddings and a FAISS index."""

    def __init__(
        self,
        chunks: list[dict[str, str]],
        encoder: TextEncoder,
        vector_index: FaissIndex,
    ):
        if not chunks:
            raise ValueError("chunks must not be empty")

        if vector_index.size != len(chunks):
            raise ValueError(
                "FAISS index size must match the number of chunks"
            )

        self.chunks = chunks
        self.encoder = encoder
        self.vector_index = vector_index

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, str | int | float]]:
        """Return the most semantically relevant chunks."""

        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.encoder.encode_queries([query])

        scores, indices = self.vector_index.search(
            query_embedding,
            top_k=top_k,
        )

        results = []

        for rank, (score, index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            idx = int(index)
            chunk = self.chunks[idx]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}).copy(),
                    "score": float(score),
                    "rank": rank,
                }
            )

        return results
