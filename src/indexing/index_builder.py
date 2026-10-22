from typing import TYPE_CHECKING

from src.indexing.faiss_index import FaissIndex

if TYPE_CHECKING:
    from src.embeddings.encoder import TextEncoder


def build_faiss_index(
    chunks: list[dict[str, str]],
    encoder: "TextEncoder",
) -> FaissIndex:
    """Build a FAISS index from text chunks."""

    if not chunks:
        raise ValueError("chunks must not be empty")

    required_keys = {"chunk_id", "source", "text"}
    seen_ids = set()

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk at position {position} must be a dictionary"
            )

        if not required_keys.issubset(chunk):
            raise ValueError(
                f"Chunk at position {position} is missing required keys"
            )

        for key in required_keys:
            if not isinstance(chunk[key], str) or not chunk[key].strip():
                raise ValueError(
                    f"Chunk at position {position} has invalid {key}"
                )

        if chunk["chunk_id"] in seen_ids:
            raise ValueError(
                f"Duplicate chunk_id: {chunk['chunk_id']}"
            )
        seen_ids.add(chunk["chunk_id"])

        if "metadata" in chunk and not isinstance(chunk["metadata"], dict):
            raise ValueError(
                f"Chunk at position {position} metadata must be a dictionary"
            )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = encoder.encode_documents(texts)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Encoder must return one 2D embedding per chunk"
        )

    if embeddings.shape[1] == 0:
        raise ValueError("Embedding dimension must be positive")

    index = FaissIndex(dimension=embeddings.shape[1])
    index.add(embeddings)

    return index
