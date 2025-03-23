from agent import Agent
from agent_config import AgentConfig, AgentStatus

EA_MODEL = "gemma3:latest"


class ExecutiveAssistant(Agent):
    def __init__(self, prompt_path: str, args: dict):
        with open(prompt_path, "r") as f:
            self.ea_prompt = f.read()

        for key, value in args.items():
            self.ea_prompt = self.ea_prompt.replace(f"{{{key}}}", value)

        self.config = AgentConfig(
            name="executive_assistant",
            display_name="Executive Assistant",
            description="Handles all tasks and requests for the user",
            system_prompt=self.ea_prompt,
            model=EA_MODEL,
            temperature=0.7,
            status=AgentStatus.ACTIVE,
        )

        super().__init__(self.config)
