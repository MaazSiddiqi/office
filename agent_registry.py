#!/usr/bin/env python3

import os
import json
from typing import Dict
from logger import Logger
from agent_config import AgentConfig
from agent import Agent


class AgentRegistry:
    """Manages the registration and spawning of AI agents."""

    def __init__(self, registry_dir: str = "registry"):
        self.registry_dir = registry_dir
        self.agents: Dict[str, AgentConfig] = {}

    def load(self):
        self._load_all_configs()

    def get_agent(self, agent_name: str) -> AgentConfig:
        return self.agents.get(agent_name)

    def spawn_agent(self, agent_name: str) -> Agent:
        return Agent(self.get_agent(agent_name))

    def list_agents(self) -> Dict[str, AgentConfig]:
        return self.agents

    def _load_config(self, config_path: str) -> AgentConfig:
        self.log(f"Loading agent config from {config_path}")
        with open(config_path, "r") as f:
            return AgentConfig(**json.load(f))

    def _load_all_configs(self):
        self.log("Loading all agents from registry")
        for filename in os.listdir(self.registry_dir):
            if not filename.endswith(".json"):
                continue
            config = self._load_config(os.path.join(self.registry_dir, filename))
            self.agents[config.name] = config
        self.log(f"Loaded {len(self.agents)} agents")

    def __str__(self) -> str:
        return (
            "\n".join(
                [
                    f"{name} - Name: {agent.display_name}, Description: {agent.description}"
                    for name, agent in self.agents.items()
                ]
            )
            if self.agents
            else "No agents available"
        )

    def log(self, message: str):
        Logger.print_system("AgentRegistry", message)
