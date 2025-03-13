#!/usr/bin/env python3

"""
AI Office v2 - Executive Assistant Module
=========================================

This module defines the ExecutiveAssistant class, which serves as the primary
interface for user interaction in the AI Office system.
"""

import json
import requests
import time
import datetime
import sys
import threading
from output_manager import (
    OutputManager,
    SUBTLE_COLOR,
    RESET,
    EA_COLOR,
    ARROW,
    ERROR_COLOR,
)
from agent_registry import get_registry
from query_router import QueryRouter, ROUTER_MODEL

# Configuration
MODEL_NAME = "llama3.1:latest"  # Using local LLM
API_URL = "http://localhost:11434/api/generate"  # Local Ollama API


class ExecutiveAssistant:
    """
    Executive Assistant (EA) that serves as the primary interface for the user.
    This is a simplified version focusing only on basic conversation.
    """

    def __init__(self):
        """Initialize the Executive Assistant."""
        self.conversation_history = []
        self.registry = get_registry()
        self.router = QueryRouter(
            self.registry, enabled=True, verbose=False, speed_mode="fast"
        )
        self.auto_delegation_enabled = True
        self.request_in_progress = False

        # Define the EA system prompt with personality and role definition
        self.system_prompt = """You are an Executive Assistant (EA) in an AI Office environment, serving as the central coordinator of a team of specialized AI agents.

### Your Role and Identity
You are the primary point of contact for the user (the "CEO") and the orchestrator of all agent activities. Think of yourself as the Chief of Staff in an executive office - you don't handle every specialized task yourself, but you ensure everything runs smoothly and coordinate all activities.

### Core Responsibilities
1. **Central Communication Hub**: All user interactions flow exclusively through you. Users never interact directly with specialized agents.
2. **Task Delegation**: Assess user requests and delegate appropriate tasks to specialized agents based on their expertise.
3. **Conversation Management**: Conduct conversations with specialized agents on the user's behalf to gather information or complete tasks.
4. **Information Synthesis**: Compile and synthesize information from multiple agents into cohesive, unified responses for the user.
5. **Knowledge Management**: Control access to the centralized memory system, providing relevant context to agents only when necessary.
6. **Task Tracking**: Monitor the progress of all ongoing tasks and provide status updates to the user.
7. **Continuous Improvement**: Analyze performance and user feedback to improve agent capabilities over time.

### Agent Ecosystem You Will Coordinate
- **Research Assistant**: Finds, analyzes, and summarizes information from trusted sources
- **Calendar Manager**: Handles scheduling, appointments, and time management
- **Project Manager**: Tracks project milestones, deliverables, and coordinates teamwork
- **Creative Director**: Generates creative content, designs, and creative problem-solving
- **Technical Specialist**: Provides technical expertise, code snippets, and technical troubleshooting
- **Data Analyst**: Processes, analyzes, and visualizes data for insights
- **Communications Expert**: Drafts communications, emails, and helps with outreach
- **Personal Assistant**: Manages personal tasks, reminders, and lifestyle requests

### Your Communication Style
- Professional yet personable and approachable
- Clear, concise, and structured
- Proactive in anticipating needs and following up
- Transparent about capabilities and limitations
- Contextually appropriate formality level with the user

### Current Limitations (Be Transparent About These)
- You're currently in early development with limited functionality
- Specialized agents and advanced features are not fully implemented yet
- You can only engage in conversation at this stage
- Your memory is limited to the current session

### How to Handle Requests
1. For general questions and simple tasks: Handle directly
2. For domain-specific requests (when future capabilities exist): Explain that you would normally delegate this to a specialized agent, but this functionality is still in development
3. For complex requests requiring multiple agents: Explain how you would coordinate between specialized agents to accomplish this in the future

### Future Vision (What You Will Be Able to Do)
- Seamlessly coordinate multiple specialized agents on complex tasks
- Maintain long-term memory of user preferences and important information
- Run multiple tasks in parallel while continuing to engage with the user
- Provide a comprehensive feedback loop for continuous improvement
- Adapt agent behaviors based on user satisfaction and preferences

Always maintain a helpful, efficient, and professional demeanor. Your purpose is to make the user's experience as productive and pleasant as possible.
"""

    def delegate_to_agent(self, agent_name, query):
        """
        Delegate a query to a specific agent.

        Args:
            agent_name (str): Name of the agent to query
            query (str): The query to send to the agent

        Returns:
            str: The agent's response
        """
        # Check if the agent is available in registry
        if agent_name not in self.registry.list_available_agents():
            return f"Agent '{agent_name}' is not configured in the system."

        # Check if the agent process is running
        if agent_name not in self.registry.agent_processes:
            return f"Agent '{agent_name}' is not currently running or available."

        # Mark request as in progress
        self.request_in_progress = True

        # Print the agent response prefix
        timestamp = OutputManager.format_timestamp()
        print(
            f"{SUBTLE_COLOR}[{timestamp}] {EA_COLOR}Agent '{agent_name}' {ARROW} {RESET}",
            end="",
            flush=True,
        )

        # Track whether we've started printing the response
        is_printing = False
        last_chunk_newline = False
        full_response = ""

        # Define a callback to handle streaming responses
        def handle_response_chunk(message):
            nonlocal is_printing, last_chunk_newline, full_response

            if message["type"] == "status":
                if message["status"] == "starting":
                    # Status updates are printed in subtle style
                    print(f"{SUBTLE_COLOR}{message['message']}{RESET}")

            elif message["type"] == "response":
                # Regular responses are printed normally
                chunk = message.get("response", "")
                full_response += chunk

                if chunk:
                    # Print the chunk
                    is_printing = True
                    print(chunk, end="", flush=True)
                    last_chunk_newline = chunk.endswith("\n")

                # If this is the final message, add a newline if needed
                if (
                    message.get("is_final", False)
                    and is_printing
                    and not last_chunk_newline
                ):
                    print()  # Add a newline at the end if needed

            return False  # Continue receiving chunks

        try:
            # Send the query to the agent and get streaming responses
            response = self.registry.send_request_to_agent(
                agent_name, query, handle_response_chunk
            )

            # Add a newline if we didn't get any output
            if not is_printing:
                print()

            # Print a divider to separate responses
            OutputManager.print_divider()

            return response
        finally:
            # End the request
            self.request_in_progress = False

    def handle_auto_delegation(self, user_input, timestamp):
        """
        Use the QueryRouter to determine if the query should be delegated to an agent.

        Args:
            user_input (str): The user's query
            timestamp (str): Timestamp for the query

        Returns:
            tuple: (delegated (bool), response (str))
        """
        # Check if auto delegation is enabled
        if not self.auto_delegation_enabled:
            return False, None

        # First check if there are any running agents
        running_agents = list(self.registry.agent_processes.keys())
        if not running_agents:
            return False, None

        # Mark request as in progress
        self.request_in_progress = True

        try:
            # Use the router to analyze the query
            start_time = time.time()
            routing_decision = self.router.route_query(
                user_input, self.conversation_history
            )
            routing_time = time.time() - start_time

            # Log the routing decision
            if hasattr(self.router, "debug_log") and self.router.debug_log:
                # When routing fails, show more diagnostic information
                if routing_decision.get("confidence", 0) <= 0.1:
                    print(
                        f"{SUBTLE_COLOR}Router: Failed to analyze query. Handling with EA.{RESET}"
                    )
                    # Add more detailed error info
                    for log_entry in self.router.debug_log[
                        -3:
                    ]:  # Show last few entries for debugging
                        if (
                            "error" in log_entry.lower()
                            or "raw response" in log_entry.lower()
                        ):
                            print(f"{SUBTLE_COLOR}Router Debug: {log_entry}{RESET}")
                else:
                    for log_entry in self.router.debug_log:
                        print(f"{SUBTLE_COLOR}Router: {log_entry}{RESET}")
                    # Add routing time metric
                    print(
                        f"{SUBTLE_COLOR}Router: Total routing time: {routing_time:.2f}s{RESET}"
                    )
                self.router.debug_log = []  # Clear the debug log

            # If the router suggests delegation with sufficient confidence
            if (
                routing_decision.get("delegate", False)
                and routing_decision.get("confidence", 0) > 0.7
            ):
                agent_name = routing_decision.get("agent")
                explanation = routing_decision.get("explanation", "")

                # Print delegation message
                print(
                    f"{SUBTLE_COLOR}[{timestamp}] Auto-delegating to {agent_name}: {explanation}{RESET}"
                )

                # Delegate to the agent
                response = self.delegate_to_agent(agent_name, user_input)

                # Add to conversation history
                self.conversation_history.append(
                    {"role": "user", "content": user_input, "timestamp": timestamp}
                )
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "timestamp": OutputManager.format_timestamp(),
                    }
                )

                return True, response

            return False, None
        finally:
            # End the request
            self.request_in_progress = False

    def generate_response(self, user_input):
        """
        Generate a response from the EA based on user input.

        Args:
            user_input (str): The user's message

        Returns:
            str: The EA's response
        """
        # Check for auto-delegation toggle command
        if user_input.lower() in ["/auto on", "/auto enable"]:
            self.auto_delegation_enabled = True
            self.router.enabled = True
            response = "Automatic agent delegation has been enabled."
            OutputManager.print_ea_response_prefix()
            print(response)
            OutputManager.print_divider()
            return response

        elif user_input.lower() in ["/auto off", "/auto disable"]:
            self.auto_delegation_enabled = False
            self.router.enabled = False
            response = "Automatic agent delegation has been disabled. You can still use /ask commands."
            OutputManager.print_ea_response_prefix()
            print(response)
            OutputManager.print_divider()
            return response

        # Router mode commands
        elif user_input.lower() in ["/router verbose", "/router debug"]:
            self.router.verbose = True
            response = "Router verbose mode enabled. System prompts will include detailed instructions."
            OutputManager.print_ea_response_prefix()
            print(response)
            OutputManager.print_divider()
            return response

        elif user_input.lower() in ["/router simple", "/router fast"]:
            self.router.verbose = False
            self.router.speed_mode = "fast"
            response = "Router fast mode enabled. Using balanced speed and accuracy for routing."
            OutputManager.print_ea_response_prefix()
            print(response)
            OutputManager.print_divider()
            return response

        elif user_input.lower() in ["/router fastest", "/router keyword"]:
            self.router.verbose = False
            self.router.speed_mode = "fastest"
            response = "Router fastest mode enabled. Using keyword matching for instantaneous routing when possible."
            OutputManager.print_ea_response_prefix()
            print(response)
            OutputManager.print_divider()
            return response

        elif user_input.lower() in ["/router accurate", "/router precise"]:
            self.router.verbose = True
            self.router.speed_mode = "accurate"
            response = "Router accurate mode enabled. Using more thorough analysis for better routing decisions."
            OutputManager.print_ea_response_prefix()
            print(response)
            OutputManager.print_divider()
            return response

        elif user_input.lower() == "/router":
            # Display router configuration
            router_status = "enabled" if self.router.enabled else "disabled"
            verbose_status = "verbose" if self.router.verbose else "concise"
            speed_mode = self.router.speed_mode
            model_info = f"using {ROUTER_MODEL}"
            response = f"Router status: {router_status}, verbosity: {verbose_status}, mode: {speed_mode}, {model_info}."
            OutputManager.print_ea_response_prefix()
            print(response)
            OutputManager.print_divider()
            return response

        # Check for agent delegation commands
        if user_input.startswith("/ask "):
            # Format: /ask agent_name query
            parts = user_input.split(" ", 2)
            if len(parts) >= 3:
                agent_name = parts[1]
                query = parts[2]

                timestamp = OutputManager.format_timestamp()
                print(
                    f"{SUBTLE_COLOR}[{timestamp}] Delegating to {agent_name}...{RESET}"
                )

                # Get list of running agents for better error messages
                available_agents = list(self.registry.agent_processes.keys())

                # Process the query with error handling
                if agent_name not in self.registry.list_available_agents():
                    response = f"I'm sorry, but '{agent_name}' is not a configured agent. Available agents are: {', '.join(self.registry.list_available_agents())}"
                    OutputManager.print_ea_response_prefix()
                    print(response)
                    OutputManager.print_divider()
                elif agent_name not in available_agents:
                    response = f"I'm sorry, but the '{agent_name}' agent is not currently running. Running agents are: {', '.join(available_agents)}"
                    OutputManager.print_ea_response_prefix()
                    print(response)
                    OutputManager.print_divider()
                else:
                    # The delegate_to_agent method will handle printing the response
                    response = self.delegate_to_agent(agent_name, query)

                # Add the interaction to conversation history
                self.conversation_history.append(
                    {"role": "user", "content": user_input, "timestamp": timestamp}
                )
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "timestamp": OutputManager.format_timestamp(),
                    }
                )

                return response

        # Get timestamp for this interaction
        timestamp = OutputManager.format_timestamp()

        # Try automatic delegation first (unless this is a system command)
        if not user_input.startswith("/"):
            delegated, response = self.handle_auto_delegation(user_input, timestamp)
            if delegated:
                return response

        # If we get here, either auto-delegation was not triggered or this is a system command
        # Add user input to conversation history with timestamp
        self.conversation_history.append(
            {"role": "user", "content": user_input, "timestamp": timestamp}
        )

        # Prepare conversation context
        conversation_context = ""
        if len(self.conversation_history) > 1:
            conversation_context = "Previous conversation:\n"
            # Include up to the last 10 exchanges for context
            for entry in self.conversation_history[-10:]:
                role = "User" if entry["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {entry['content']}\n"

        # Create prompt for the LLM
        prompt = f"{conversation_context}\n\nUser: {user_input}"

        # Create the payload for the LLM API
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.7, "max_tokens": 500},
            "system": self.system_prompt,
        }

        # Display timestamp and thinking animation
        print(f"{SUBTLE_COLOR}[{timestamp}]{RESET}")
        OutputManager.display_thinking_animation()

        # Mark request as in progress
        self.request_in_progress = True

        try:
            # Call the LLM API with streaming to show output as it's generated
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            full_response = ""

            # Display EA label with timestamp
            OutputManager.print_ea_response_prefix()

            # Stream the response chunks as they come in
            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line)
                if "response" in data:
                    chunk = data["response"]
                    full_response += chunk
                    print(chunk, end="", flush=True)

                if data.get("done", False):
                    break

            print()  # Add a newline after the response

            # Add response to conversation history
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": OutputManager.format_timestamp(),
                }
            )

            # Add a subtle divider
            OutputManager.print_divider()

            return full_response

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            OutputManager.print_error(error_msg)
            OutputManager.print_divider()
            return error_msg
        finally:
            # End the request
            self.request_in_progress = False
