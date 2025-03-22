from agent import Agent
from agent_config import AgentConfig, AgentStatus


class ExecutiveAssistant(Agent):
    def __init__(self, prompt_path: str, args: dict):
        with open(prompt_path, "r") as f:
            self.ea_prompt = f.read()

        self.ea_prompt = self.ea_prompt.format(**args)

        self.config = AgentConfig(
            name="executive_assistant",
            display_name="Executive Assistant",
            description="Handles all tasks and requests for the user",
            system_prompt=self.ea_prompt,
            model="llama3.1:latest",
            temperature=0.7,
            status=AgentStatus.ACTIVE,
        )

        super().__init__(self.config)
