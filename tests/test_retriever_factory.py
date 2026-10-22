import numpy as np

from src.indexing.faiss_index import FaissIndex
from src.indexing.index_store import IndexStore
from src.retrieval.retriever_factory import load_hybrid_retriever


class FakeEncoder:
    def encode_documents(self, texts):
        raise AssertionError(
            "Loading a stored index must not re-encode documents"
        )

    def encode_queries(self, queries):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_load_hybrid_retriever_and_search(tmp_path):
    chunks = [
        {
            "chunk_id": "plastic.txt::chunk_0",
            "source": "plastic.txt",
            "text": "PET plastic bottles can be recycled.",
        },
        {
            "chunk_id": "paper.txt::chunk_0",
            "source": "paper.txt",
            "text": "Paper recycling saves trees and resources.",
        },
    ]

    index = FaissIndex(dimension=2)
    index.add(
        np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
    )

    store = IndexStore(tmp_path)
    store.save(index, chunks)

    retriever = load_hybrid_retriever(
        index_dir=tmp_path,
        encoder=FakeEncoder(),
    )

    results = retriever.search(
        "PET plastic bottles",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["source"] == "plastic.txt"
    assert results[0]["chunk_id"] == "plastic.txt::chunk_0"
    assert results[0]["rank"] == 1
