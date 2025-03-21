import requests

API_URL = "http://localhost:11434/api/generate"


class LLM:
    def __init__(self, model: str, max_tokens: int, temperature: float):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Ollama API.

        Args:
            prompt (str): The input prompt to generate a response for

        Returns:
            str: The generated response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "max_tokens": self.max_tokens},
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            result = ""
            for line in response.text.splitlines():
                if not line:
                    continue

                data = response.json()
                if "response" in data:
                    result += data["response"]

                if data.get("done", False):
                    break

            return result

        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
