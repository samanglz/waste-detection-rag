from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.indexing.index_store import IndexStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


def load_hybrid_retriever(
    index_dir: str | Path,
    encoder: TextEncoder,
) -> HybridRetriever:
    """Load a persisted index and prepare hybrid retrieval.

    The encoder is injected by the caller. Document embeddings
    are not recomputed when the stored index is loaded.
    """

    store = IndexStore(index_dir)
    vector_index, chunks = store.load()

    bm25_retriever = BM25Retriever(chunks)

    return HybridRetriever(
        chunks=chunks,
        encoder=encoder,
        vector_index=vector_index,
        bm25_retriever=bm25_retriever,
    )
