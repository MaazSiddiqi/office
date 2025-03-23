import requests
from typing import Generator
import json
from logger import Logger

API_URL = "http://localhost:11434/api/generate"


class LLM:
    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def generate_response_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate a streaming response using the Ollama API.

        Args:
            prompt (str): The input prompt to generate a response for

        Returns:
            Generator[str, None, None]: A generator of response chunks
        """
        # Add the user prompt to the conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        # Build the full prompt with conversation history
        full_prompt = ""
        for message in self.conversation_history:
            role_prefix = "User: " if message["role"] == "user" else "Assistant: "
            full_prompt += f"{role_prefix}{message['content']}\n"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,  # Ensure streaming is explicitly enabled
        }

        # Let exceptions propagate to the caller
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()

        full_response = ""

        buffer = ""
        in_op = False
        op_buffer = ""

        for line in response.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line)
                chunk = data["response"] if "response" in data else ""

                for char in chunk:
                    if not in_op and "[CONNECT]" in buffer:
                        print(f"\n\n[CONNECT_STATUS] Connection request initiated.")
                        in_op = True
                        op_buffer = ""

                    if in_op:
                        if char == "\n":
                            print(
                                f"[CONNECT_STATUS] Successfully connected to agent {op_buffer.strip()}"
                            )

                            full_response += f"Hello! How may I help you today?"

                            for response in self.generate_response_stream(
                                full_prompt + full_response
                            ):
                                full_response += response
                                yield response
                            break
                        op_buffer += char
                    else:
                        buffer += char
                        if char == " ":
                            full_response += buffer
                            yield buffer
                            buffer = ""

            except json.JSONDecodeError:
                continue

        if not in_op:
            yield buffer

        # After generating the complete response, add it to conversation history
        self.conversation_history.append(
            {"role": "assistant", "content": full_response}
        )

    def log(self, message: str):
        Logger.print_system("LLM", message)
