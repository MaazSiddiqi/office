import requests
from typing import Generator
import json

API_URL = "http://localhost:11434/api/generate"


class LLM:
    def __init__(self, model: str, max_tokens: int, temperature: float):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_response_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate a response using the Ollama API.

        Args:
            prompt (str): The input prompt to generate a response for

        Returns:
            Generator[str, None, None]: A generator of response chunks
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            print(f"Error generating response: {e}")
            yield ""
