from agent_config import AgentConfig
from llm import LLM
from typing import Generator


class Agent:
    def __init__(self, config: AgentConfig):
        self.llm = LLM(
            config.model,
            config.system_prompt,
            2048,
            config.temperature,
        )

    def ask(self, prompt: str) -> Generator[str, None, None]:
        return self.llm.generate_response_stream(prompt)
