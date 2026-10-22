from types import SimpleNamespace

import pytest

from src.generation.gemini_generator import GeminiAnswerGenerator


class FakeModels:
    def __init__(self, response_text="  Test answer.  "):
        self.response_text = response_text
        self.last_request = None

    def generate_content(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(text=self.response_text)


class FakeGeminiClient:
    def __init__(self, response_text="  Test answer.  "):
        self.models = FakeModels(response_text)


def test_generator_uses_injected_client_and_model():
    client = FakeGeminiClient()
    generator = GeminiAnswerGenerator(
        model="test-model",
        client=client,
    )

    answer = generator.generate("Test prompt")

    assert answer == "Test answer."
    assert client.models.last_request == {
        "model": "test-model",
        "contents": "Test prompt",
    }


def test_generator_uses_environment_model(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "env-test-model")
    generator = GeminiAnswerGenerator(client=FakeGeminiClient())

    assert generator.model == "env-test-model"


@pytest.mark.parametrize("prompt", ["", " ", "\n"])
def test_generator_rejects_empty_prompt(prompt):
    generator = GeminiAnswerGenerator(client=FakeGeminiClient())

    with pytest.raises(ValueError, match="prompt must not be empty"):
        generator.generate(prompt)


def test_generator_rejects_empty_model():
    with pytest.raises(ValueError, match="model must not be empty"):
        GeminiAnswerGenerator(model=" ", client=FakeGeminiClient())


def test_generator_rejects_empty_response():
    generator = GeminiAnswerGenerator(
        client=FakeGeminiClient(response_text="  ")
    )

    with pytest.raises(RuntimeError, match="empty text response"):
        generator.generate("Test prompt")
