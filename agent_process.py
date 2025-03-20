#!/usr/bin/env python3

"""
AI Office v2 - Agent Process
=========================

This module handles the process that runs an AI agent.
"""

import json
import requests
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Dict, Any, Optional
from logger import Logger
from agent_config import AgentConfig

# Configuration
API_URL = "http://localhost:11434/api/generate"


class AgentProcess(Process):
    """Process that runs an AI agent."""

    def __init__(self, agent_name: str, agent_config: AgentConfig, pipe: Connection):
        super().__init__()
        self.agent_name = agent_name
        self.config = agent_config
        self.pipe = pipe
        self.daemon = True

    def run(self):
        """Main process loop that handles agent requests."""
        while True:
            try:
                if self.pipe.poll(timeout=None):
                    message = self.pipe.recv()

                    if message["type"] == "request":
                        self._handle_request(message["query"])
                    elif message["type"] == "shutdown":
                        break

            except Exception as e:
                Logger.print_error(f"Error in agent '{self.agent_name}': {e}")
                break

    def _handle_request(self, query: str):
        """Handle an incoming request."""
        try:
            self.pipe.send(
                {
                    "type": "status",
                    "agent": self.agent_name,
                    "status": "starting",
                    "message": f"Processing request...",
                }
            )

            self._call_llm_api(query)

            self.pipe.send(
                {
                    "type": "status",
                    "agent": self.agent_name,
                    "status": "completed",
                    "message": "Request completed.",
                }
            )

        except Exception as e:
            self.pipe.send(
                {
                    "type": "response",
                    "agent": self.agent_name,
                    "response": f"Error: {str(e)}",
                    "is_error": True,
                    "is_final": True,
                }
            )

    def _call_llm_api(self, query: str):
        """Call the LLM API with streaming enabled."""
        try:
            payload = {
                "model": self.config.model,
                "prompt": query,
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "max_tokens": 500,
                    "num_ctx": 2048,
                    "num_thread": 4,
                },
                "system": self.config.system_prompt,
            }

            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if "response" in data:
                        chunk = data["response"]
                        full_response += chunk
                        self.pipe.send(
                            {
                                "type": "response",
                                "agent": self.agent_name,
                                "response": chunk,
                                "is_error": False,
                                "is_final": False,
                            }
                        )

                    if data.get("done", False):
                        break

                except json.JSONDecodeError:
                    continue

            self.pipe.send(
                {
                    "type": "response",
                    "agent": self.agent_name,
                    "response": "",
                    "full_response": full_response,
                    "is_error": False,
                    "is_final": True,
                }
            )

        except Exception as e:
            self.pipe.send(
                {
                    "type": "response",
                    "agent": self.agent_name,
                    "response": f"Error calling API: {str(e)}",
                    "is_error": True,
                    "is_final": True,
                }
            )
