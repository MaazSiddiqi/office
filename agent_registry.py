#!/usr/bin/env python3

"""
AI Office v2 - Agent Registry
=============================

This module defines the AgentRegistry class, which manages the registration and
lifecycle of specialized AI agents in the AI Office system.
"""

import os
import json
import signal
import time
import multiprocessing
from multiprocessing import Process, Pipe
import requests
from output_manager import SUBTLE_COLOR, RESET

# Default directories and file paths
AGENTS_DIR = "agents"
DEFAULT_MODEL = "llama3.1:latest"
API_URL = "http://localhost:11434/api/generate"


class AgentProcess(Process):
    """
    Represents an agent running in a separate process.

    This class handles the communication between the main process and the agent,
    as well as the agent's lifecycle.
    """

    def __init__(self, agent_name, agent_config, pipe=None):
        """
        Initialize the agent process.

        Args:
            agent_name (str): Name of the agent
            agent_config (dict): Configuration for the agent
            pipe (Connection): Communication pipe to the main process
        """
        super().__init__()
        self.agent_name = agent_name
        self.system_prompt = agent_config["system_prompt"]
        self.model_name = agent_config.get("model", DEFAULT_MODEL)
        self.temperature = agent_config.get("temperature", 0.7)
        self.pipe = pipe
        self.daemon = True  # Process will exit when main process exits
        # Add verbose flag
        self.verbose = False

    def run(self):
        """Main process loop that waits for and handles incoming requests."""
        # Only print this in verbose mode
        if self.verbose:
            print(f"Agent '{self.agent_name}' started in process {os.getpid()}")

        while True:
            try:
                if self.pipe.poll(timeout=None):  # Wait for messages
                    message = self.pipe.recv()

                    if message["type"] == "request":
                        # Process the request
                        query = message["query"]
                        # Only print in verbose mode
                        if self.verbose:
                            print(
                                f"Agent '{self.agent_name}' received request: {query[:30]}..."
                            )

                        try:
                            # Notify main process that we're starting
                            self.pipe.send(
                                {
                                    "type": "status",
                                    "agent": self.agent_name,
                                    "status": "starting",
                                    "message": f"Agent '{self.agent_name}' is processing your request...",
                                }
                            )

                            # Call the LLM API with streaming
                            if self.verbose:
                                print(
                                    f"Agent '{self.agent_name}' calling LLM API with streaming..."
                                )
                            self.call_llm_api_streaming(query)

                            # Notify main process that we're done
                            self.pipe.send(
                                {
                                    "type": "status",
                                    "agent": self.agent_name,
                                    "status": "completed",
                                    "message": f"Agent '{self.agent_name}' completed processing your request.",
                                }
                            )

                        except Exception as e:
                            print(
                                f"Agent '{self.agent_name}' error processing request: {e}"
                            )
                            # Send error response back
                            self.pipe.send(
                                {
                                    "type": "response",
                                    "agent": self.agent_name,
                                    "response": f"Error in agent '{self.agent_name}': {str(e)}",
                                    "is_error": True,
                                    "is_final": True,
                                }
                            )

                    elif message["type"] == "shutdown":
                        if self.verbose:
                            print(f"Agent '{self.agent_name}' shutting down...")
                        break

            except Exception as e:
                print(f"Error in agent '{self.agent_name}' main loop: {e}")
                break

        if self.verbose:
            print(f"Agent '{self.agent_name}' process exiting.")

    def call_llm_api_streaming(self, query):
        """
        Call the LLM API and stream the response back to the main process.

        Args:
            query (str): The query to send to the LLM
        """
        try:
            # Create the payload for the LLM API with streaming enabled
            payload = {
                "model": self.model_name,
                "prompt": query,
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": self.temperature,
                    "max_tokens": 500,
                    # Add performance options
                    "num_ctx": 2048,  # Reduce context window for faster processing
                    "num_thread": 4,  # Use multiple threads
                },
                "system": self.system_prompt,
            }

            # Log API call attempt
            if self.verbose:
                print(
                    f"Agent '{self.agent_name}' streaming from API with model: {self.model_name}"
                )

            # Call the LLM API with streaming
            start_time = time.time()
            full_response = ""

            # Make the API call with streaming
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            # Process the streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    if "response" in data:
                        chunk = data["response"]
                        full_response += chunk

                        # Send the chunk back to the main process
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
                    if self.verbose:
                        print(
                            f"Agent '{self.agent_name}' received invalid JSON: {line}"
                        )

            # Send final complete response
            elapsed_time = time.time() - start_time
            if self.verbose:
                print(
                    f"Agent '{self.agent_name}' LLM API streaming completed in {elapsed_time:.2f}s"
                )

            # Send final message to indicate completion
            self.pipe.send(
                {
                    "type": "response",
                    "agent": self.agent_name,
                    "response": "",  # Empty string as we've already sent the content
                    "full_response": full_response,  # Include full response for reference
                    "is_error": False,
                    "is_final": True,
                }
            )

        except Exception as e:
            if self.verbose:
                print(
                    f"Agent '{self.agent_name}' exception in call_llm_api_streaming: {str(e)}"
                )
            self.pipe.send(
                {
                    "type": "response",
                    "agent": self.agent_name,
                    "response": f"Error calling {self.agent_name}: {str(e)}",
                    "is_error": True,
                    "is_final": True,
                }
            )


class AgentRegistry:
    """
    Registry that manages the registration and lifecycle of specialized AI agents.

    This class is responsible for:
    1. Loading agent configurations from files
    2. Starting agent processes
    3. Communicating with agents
    4. Shutting down agent processes cleanly
    """

    def __init__(self):
        """Initialize the Agent Registry."""
        self.agents = {}
        self.agent_processes = {}
        self.agent_pipes = {}
        # Add debug log
        self.debug_log = []
        self.load_agents()

    def load_agents(self):
        """Load agent configurations from the agents directory."""
        if not os.path.exists(AGENTS_DIR):
            self.debug_log.append(f"Warning: Agents directory '{AGENTS_DIR}' not found")
            return

        # Load all agent configuration files
        for filename in os.listdir(AGENTS_DIR):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(AGENTS_DIR, filename)
                    with open(filepath, "r") as f:
                        agent_config = json.load(f)

                    agent_name = agent_config.get("name")
                    if agent_name:
                        self.agents[agent_name] = agent_config
                        # Store in debug log instead of printing
                        self.debug_log.append(
                            f"Loaded agent configuration: {agent_name}"
                        )
                    else:
                        self.debug_log.append(
                            f"Warning: Agent configuration missing name: {filepath}"
                        )

                except Exception as e:
                    self.debug_log.append(
                        f"Error loading agent configuration '{filename}': {e}"
                    )

    def test_agent_availability(self, agent_name):
        """
        Test if an agent's LLM API is available by making a small test request.

        Args:
            agent_name (str): Name of the agent to test

        Returns:
            bool: True if the agent's LLM API is available, False otherwise
        """
        if agent_name not in self.agents:
            # No print, handled by animation
            return False

        model = self.agents[agent_name].get("model", DEFAULT_MODEL)
        try:
            # Create a simple payload for testing
            payload = {
                "model": model,
                "prompt": "Hello",
                "stream": False,
                "options": {"temperature": 0.7, "max_tokens": 10},
            }

            # Make the request with a short timeout
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()

            # Check for valid response
            response_data = response.json()
            if "response" in response_data:
                return True
            else:
                return False

        except Exception as e:
            # Record error in debug log but don't print to console
            if hasattr(self, "debug_log"):
                self.debug_log.append(f"Error testing API for '{agent_name}': {e}")
            return False

    def start_agent_processes(self, use_animation=True):
        """
        Start processes for all agents in the registry.

        Args:
            use_animation (bool): Whether to use animation while starting agents
                                 (Note: animation is now disabled)
        """
        from output_manager import OutputManager

        # Count agents to start
        agent_count = len(self.agents)
        if agent_count == 0:
            OutputManager.print_warning("No agents configured")
            return

        # Display initial message
        OutputManager.print_info(f"Initializing {agent_count} agents...")

        # Track successful starts
        successful_starts = []
        failed_starts = []

        for agent_name, agent_config in self.agents.items():
            # Test LLM API availability first
            if not self.test_agent_availability(agent_name):
                # Log the failure to debug log
                self.debug_log.append(f"API test failed for {agent_name}")
                failed_starts.append(agent_name)
                continue

            try:
                # Create a pipe for communication
                parent_conn, child_conn = Pipe()

                # Create and start the agent process
                agent_process = AgentProcess(
                    agent_name=agent_name, agent_config=agent_config, pipe=child_conn
                )

                # Store the process and pipe
                self.agent_processes[agent_name] = agent_process
                self.agent_pipes[agent_name] = parent_conn

                # Start the process
                agent_process.start()
                successful_starts.append(agent_name)

                # Log success to debug log
                self.debug_log.append(f"Started agent {agent_name}")

            except Exception as e:
                self.debug_log.append(f"Error starting agent '{agent_name}': {e}")
                failed_starts.append(agent_name)

        # Print summary
        if successful_starts:
            OutputManager.print_success(
                f"Started {len(successful_starts)} agents: {', '.join(successful_starts)}"
            )
        if failed_starts:
            OutputManager.print_warning(
                f"Failed to start {len(failed_starts)} agents: {', '.join(failed_starts)}"
            )

        # Add empty line for spacing
        print()

    def send_request_to_agent(self, agent_name, query, response_callback=None):
        """
        Send a request to an agent and handle streaming responses.

        Args:
            agent_name (str): Name of the agent to query
            query (str): The query to send to the agent
            response_callback (callable, optional): Callback function that will be called
                                                   with each response chunk

        Returns:
            str: The agent's final response or error message
        """
        if agent_name not in self.agent_processes:
            return f"Error: Agent '{agent_name}' not found or not running"

        try:
            # Send the query to the agent (make subtle)
            print(f"{SUBTLE_COLOR}Sending request to agent '{agent_name}'{RESET}")
            self.agent_pipes[agent_name].send({"type": "request", "query": query})

            # Process streaming responses (make subtle)
            print(
                f"{SUBTLE_COLOR}Starting to receive streaming responses from agent '{agent_name}'{RESET}"
            )

            full_response = ""
            is_complete = False

            # We'll keep polling the pipe until we get the final response
            while not is_complete:
                # Check for a response with a short timeout to not block the main thread
                if self.agent_pipes[agent_name].poll(timeout=0.1):
                    # Read the message
                    message = self.agent_pipes[agent_name].recv()

                    # Handle different message types
                    if message["type"] == "response":
                        # Check if it's the final message
                        if message.get("is_final", False):
                            is_complete = True
                            # If there's a full_response, use it (for final message)
                            if "full_response" in message and message["full_response"]:
                                full_response = message["full_response"]

                        # Add to response if not empty
                        response_chunk = message.get("response", "")
                        if response_chunk:
                            full_response += response_chunk

                        # If it's an error, mark as complete
                        if message.get("is_error", False):
                            is_complete = True

                        # Call the callback if provided
                        if response_callback and callable(response_callback):
                            try:
                                response_callback(message)
                            except Exception as e:
                                print(
                                    f"{SUBTLE_COLOR}Exception in response callback: {e}{RESET}"
                                )

                    elif message["type"] == "status":
                        # Status updates can be handled by the callback
                        if response_callback and callable(response_callback):
                            try:
                                response_callback(message)
                            except Exception as e:
                                print(
                                    f"{SUBTLE_COLOR}Exception in status callback: {e}{RESET}"
                                )

                # Check if the process is still alive
                if not self.agent_processes[agent_name].is_alive():
                    if not is_complete:
                        return f"Error: Agent '{agent_name}' process died unexpectedly"

                # Short sleep to prevent busy waiting
                time.sleep(0.01)

            return full_response

        except Exception as e:
            print(
                f"{SUBTLE_COLOR}Exception communicating with agent '{agent_name}': {e}{RESET}"
            )
            return f"Error communicating with agent '{agent_name}': {e}"

    def shutdown_agents(self):
        """Shutdown all agent processes cleanly."""
        print("Shutting down all agent processes...")

        for agent_name, agent_process in self.agent_processes.items():
            try:
                if agent_process.is_alive():
                    # Send shutdown message
                    self.agent_pipes[agent_name].send({"type": "shutdown"})

                    # Wait for agent to exit gracefully (with timeout)
                    agent_process.join(timeout=2)

                    # Force terminate if still running
                    if agent_process.is_alive():
                        print(f"Force terminating agent '{agent_name}'")
                        agent_process.terminate()

            except Exception as e:
                print(f"Error shutting down agent '{agent_name}': {e}")

        print("All agent processes shut down")

    def list_available_agents(self):
        """
        Get a list of all available agents.

        Returns:
            list: List of agent names
        """
        return list(self.agents.keys())

    def get_agent_info(self, agent_name):
        """
        Get information about an agent.

        Args:
            agent_name (str): Name of the agent

        Returns:
            dict: Agent configuration or None if not found
        """
        return self.agents.get(agent_name)


# Singleton instance
_registry_instance = None


def get_registry():
    """
    Get the singleton registry instance.

    Returns:
        AgentRegistry: The singleton registry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = AgentRegistry()
    return _registry_instance
