from types import SimpleNamespace

import pytest

from src.generation.llm_generator import OpenAIAnswerGenerator


class FakeResponses:
    def __init__(self, output_text=" Generated answer. "):
        self.output_text = output_text
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAIClient:
    def __init__(self, output_text=" Generated answer. "):
        self.responses = FakeResponses(output_text=output_text)


def test_generator_uses_injected_client_and_configured_model():
    client = FakeOpenAIClient()
    generator = OpenAIAnswerGenerator(
        model="test-model",
        client=client,
    )

    answer = generator.generate("Answer using this context.")

    assert answer == "Generated answer."
    assert client.responses.last_request == {
        "model": "test-model",
        "input": "Answer using this context.",
    }


@pytest.mark.parametrize("prompt", ["", " ", "\n"])
def test_generator_rejects_empty_prompt(prompt):
    generator = OpenAIAnswerGenerator(
        model="test-model",
        client=FakeOpenAIClient(),
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        generator.generate(prompt)


def test_generator_rejects_empty_model():
    with pytest.raises(ValueError, match="model must not be empty"):
        OpenAIAnswerGenerator(model=" ", client=FakeOpenAIClient())


def test_generator_rejects_empty_model_response():
    generator = OpenAIAnswerGenerator(
        model="test-model",
        client=FakeOpenAIClient(output_text="  "),
    )

    with pytest.raises(RuntimeError, match="empty text response"):
        generator.generate("A valid prompt.")
