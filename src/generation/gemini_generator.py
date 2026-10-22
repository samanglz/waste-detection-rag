import os
from typing import Any


class GeminiAnswerGenerator:
    """Generate answers using the Gemini Developer API.

    The client is injectable so unit tests do not make real API requests.
    If no client is provided, the official Google Gen AI SDK is initialized
    using GEMINI_API_KEY from the environment.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        if not self.model.strip():
            raise ValueError("model must not be empty")

        if client is None:
            from google import genai

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

        self.client = client

    def generate(self, prompt: str) -> str:
        """Generate a text answer from a complete prompt."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            raise RuntimeError(
                "Gemini returned an empty text response"
            )

        return answer
