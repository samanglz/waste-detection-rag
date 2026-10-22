import os
from typing import Any


class OpenAIAnswerGenerator:
    """Generate text with OpenAI's Responses API.

    The OpenAI client can be injected for testing or customized usage.
    When omitted, the SDK reads OPENAI_API_KEY from the environment.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        if not self.model.strip():
            raise ValueError("model must not be empty")

        if client is None:
            from openai import OpenAI

            client = OpenAI()

        self.client = client

    def generate(self, prompt: str) -> str:
        """Generate an answer from a complete prompt."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        answer = (getattr(response, "output_text", None) or "").strip()
        if not answer:
            raise RuntimeError("The LLM returned an empty text response")

        return answer
