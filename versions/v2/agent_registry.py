#!/usr/bin/env python3

"""
AI Office v2 - Agent Registry
============================

This module manages the registration and lifecycle of specialized AI agents.
"""

import os
import json
import time
import signal
import threading
import requests
from multiprocessing import Process, Pipe
from typing import Dict, List, Optional, Any
from logger import Logger
from agent_process import AgentProcess, AgentConfig

# Configuration
AGENTS_DIR = "agents"  # Directory containing agent configuration files
MODEL_NAME = "mistral"  # Using local LLM
API_URL = "http://localhost:11434/api/generate"  # Local Ollama API


class AgentRegistry:
    """Manages the registration and lifecycle of AI agents."""

    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self.processes: Dict[str, AgentProcess] = {}
        self.pipes: Dict[str, Pipe] = {}
        self._load_agents()

    def _load_agents(self):
        """Load agent configurations from the agents directory."""
        if not os.path.exists(AGENTS_DIR):
            Logger.print_warning(f"Agents directory '{AGENTS_DIR}' not found")
            return

        for filename in os.listdir(AGENTS_DIR):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(AGENTS_DIR, filename)
            try:
                with open(filepath, "r") as f:
                    agent_data = json.load(f)
                    agent = AgentConfig(**agent_data)
                    self.agents[agent.name] = agent
                    Logger.print_info(f"Loaded agent: {agent.display_name}")
            except Exception as e:
                Logger.print_error(f"Error loading agent from '{filename}': {e}")

    def start_agent_processes(self):
        """Start all agent processes."""
        for agent_name, agent_config in self.agents.items():
            try:
                # Create pipe for communication
                parent_pipe, child_pipe = Pipe()
                self.pipes[agent_name] = parent_pipe

                # Create and start process
                process = AgentProcess(agent_name, agent_config, child_pipe)
                process.start()
                self.processes[agent_name] = process
                Logger.print_success(
                    f"Started agent process: {agent_config.display_name}"
                )
            except Exception as e:
                Logger.print_error(f"Error starting agent '{agent_name}': {e}")

    def shutdown_agents(self):
        """Shutdown all agent processes."""
        for agent_name, process in self.processes.items():
            try:
                # Send shutdown message
                self.pipes[agent_name].send({"type": "shutdown"})
                # Wait for process to terminate
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                Logger.print_info(f"Shutdown agent: {agent_name}")
            except Exception as e:
                Logger.print_error(f"Error shutting down agent '{agent_name}': {e}")

    def send_request_to_agent(self, agent_name: str, query: str) -> Optional[str]:
        """Send a request to an agent and get its response."""
        if agent_name not in self.pipes:
            Logger.print_error(f"Agent '{agent_name}' not found")
            return None

        try:
            # Send request
            self.pipes[agent_name].send({"type": "request", "query": query})

            # Collect response
            full_response = ""
            while True:
                if self.pipes[agent_name].poll(timeout=None):
                    message = self.pipes[agent_name].recv()

                    if message["type"] == "response":
                        if message["is_error"]:
                            Logger.print_error(f"Agent error: {message['response']}")
                            return None

                        if message.get("full_response"):
                            full_response = message["full_response"]

                        if message["is_final"]:
                            return full_response

                        # Print streaming response
                        print(message["response"], end="", flush=True)

        except Exception as e:
            Logger.print_error(f"Error communicating with agent '{agent_name}': {e}")
            return None

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name)

    def list_agents(self) -> Dict[str, AgentConfig]:
        """List all registered agents."""
        return self.agents

    def list_available_agents(self) -> List[str]:
        """List all available agent names."""
        return list(self.agents.keys())

    def get_running_agents(self) -> List[str]:
        """Get a list of currently running agent names."""
        return [name for name, process in self.processes.items() if process.is_alive()]
