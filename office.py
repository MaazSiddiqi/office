from agent_registry import AgentRegistry


class Office:
    def __init__(self):
        self.agent_registry = AgentRegistry()

    def load(self):
        self.agent_registry.load()

    def run(self):
        pass
