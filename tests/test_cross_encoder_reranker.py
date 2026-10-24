from src.reranking.cross_encoder_reranker import CrossEncoderReranker


class FakeRetriever:
    def __init__(self):
        self.last_top_k = None

    def search(self, query, top_k=5):
        self.last_top_k = top_k
        return [
            {
                "chunk_id": "c1",
                "source": "doc.txt",
                "text": "first candidate",
                "score": 0.9,
                "rank": 1,
            },
            {
                "chunk_id": "c2",
                "source": "doc.txt",
                "text": "second candidate",
                "score": 0.8,
                "rank": 2,
            },
            {
                "chunk_id": "c3",
                "source": "doc.txt",
                "text": "third candidate",
                "score": 0.7,
                "rank": 3,
            },
        ][:top_k]


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        # Deliberately reverse the first and last candidates.
        mapping = {
            "first candidate": 0.1,
            "second candidate": 0.4,
            "third candidate": 0.9,
        }
        return [mapping[text] for _, text in pairs]


def test_reranker_reorders_and_preserves_original_score():
    base = FakeRetriever()
    reranker = CrossEncoderReranker(
        base_retriever=base,
        candidate_k=3,
        model=FakeCrossEncoder(),
    )

    results = reranker.search("test query", top_k=2)

    assert [item["chunk_id"] for item in results] == ["c3", "c2"]
    assert results[0]["score"] == 0.9
    assert results[0]["rerank_score"] == 0.9
    assert results[0]["original_score"] == 0.7
    assert [item["rank"] for item in results] == [1, 2]
    assert base.last_top_k == 3


def test_candidate_pool_is_at_least_requested_top_k():
    base = FakeRetriever()
    reranker = CrossEncoderReranker(
        base_retriever=base,
        candidate_k=2,
        model=FakeCrossEncoder(),
    )

    reranker.search("test query", top_k=3)

    assert base.last_top_k == 3


def test_empty_candidate_list_returns_empty_results():
    class EmptyRetriever:
        def search(self, query, top_k=5):
            return []

    reranker = CrossEncoderReranker(
        base_retriever=EmptyRetriever(),
        model=FakeCrossEncoder(),
    )

    assert reranker.search("test query", top_k=5) == []
