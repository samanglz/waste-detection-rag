from fastapi.testclient import TestClient

from src.api.app import create_app


class FakeRetriever:
    def search(self, query, top_k=5):
        return [
            {
                "chunk_id": "fake.txt::chunk_0",
                "source": "fake.txt",
                "text": "Plastic should be sorted before recycling.",
                "score": 0.9,
                "rank": 1,
            }
        ][:top_k]


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def search(self, query, top_k=5):
        self.calls += 1
        return [
            {
                "chunk_id": "fake.txt::chunk_0",
                "source": "fake.txt",
                "text": "Plastic should be sorted before recycling.",
                "score": 0.95,
                "rank": 1,
            }
        ][:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        assert "Plastic should be sorted before recycling." in prompt
        return "Plastic should be sorted before recycling. [S1]"


def make_client():
    retriever = FakeRetriever()
    reranker = FakeReranker()
    generator = FakeGenerator()
    app = create_app({
        "retriever": retriever,
        "reranker": reranker,
        "generator": generator,
    })
    return TestClient(app), retriever, reranker, generator


def test_health_endpoint():
    client, *_ = make_client()
    with client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_uses_reranker_and_generator():
    client, _, reranker, generator = make_client()

    with client:
        response = client.post(
            "/ask",
            json={"query": "How should plastic be recycled?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_method"] == "hybrid+reranker"
    assert body["reranker_error"] is None
    assert body["answer"]
    assert body["sources"][0]["id"] == "S1"
    assert reranker.calls == 1
    assert generator.calls == 1


def test_ask_rejects_blank_query():
    client, *_ = make_client()
    with client:
        response = client.post("/ask", json={"query": "   "})
    assert response.status_code == 422


def test_ask_rejects_invalid_top_k():
    client, *_ = make_client()
    with client:
        response = client.post(
            "/ask",
            json={"query": "recycling", "top_k": 0},
        )
    assert response.status_code == 422


def test_components_are_reused_across_requests():
    client, _, reranker, generator = make_client()
    with client:
        first = client.post("/ask", json={"query": "Question one"})
        second = client.post("/ask", json={"query": "Question two"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert reranker.calls == 2
    assert generator.calls == 2
