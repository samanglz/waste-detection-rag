from pathlib import Path

from src.chunking.text_chunker import chunk_documents
from src.embeddings.encoder import TextEncoder
from src.indexing.index_builder import build_faiss_index
from src.indexing.index_store import IndexStore
from src.ingestion.text_loader import load_text_documents


def build_index_pipeline(
    data_dir: str | Path,
    index_dir: str | Path,
    encoder: TextEncoder,
    chunk_size: int = 120,
    overlap: int = 15,
) -> dict[str, int]:
    """Load documents, build chunks and embeddings, then persist the index.

    The encoder is injected by the caller, so this function does not
    load or manage the embedding model.
    """

    documents = load_text_documents(data_dir)

    if not documents:
        raise ValueError(f"No non-empty text documents found in: {data_dir}")

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("Chunking produced no chunks")

    index = build_faiss_index(
        chunks=chunks,
        encoder=encoder,
    )

    store = IndexStore(index_dir)
    store.save(index, chunks)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "vectors": index.size,
    }
