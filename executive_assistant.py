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
from memory_manager import get_memory_manager
from enum import Enum

# Configuration
MODEL_NAME = "llama3.1:latest"  # Using local LLM
API_URL = "http://localhost:11434/api/generate"  # Local Ollama API


# Router speed mode options
class RouterSpeedMode(str, Enum):
    FASTEST = "fastest"  # Keyword-based, no LLM
    FAST = "fast"  # FastLLM for routing
    ACCURATE = "accurate"  # More capable but slower LLM


class ExecutiveAssistant:
    """
    Executive Assistant (EA) that serves as the primary interface for the user.
    This is a simplified version focusing only on basic conversation.
    """

    def __init__(self):
        """Initialize the Executive Assistant."""
        self.conversation_history = []
        self.registry = get_registry()

        # Initialize the router with the new parameter structure
        self.router = QueryRouter(speed_mode=RouterSpeedMode.FAST.value)

        self.auto_delegation_enabled = True
        self.request_in_progress = False
        self.router_verbose = False
        self.memory_manager = get_memory_manager()

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

    def _get_agent_by_command(self, command):
        """Get the agent name from a command like '/ask agent_name query'."""
        parts = command.split(" ", 2)
        return parts[1].lower() if len(parts) > 1 else None

    def _get_query_from_command(self, command):
        """Extract the query from a command like '/ask agent_name query'."""
        parts = command.split(" ", 2)
        return parts[2] if len(parts) > 2 else ""

    def _handle_router_command(self, command):
        """Handle router control commands."""
        parts = command.split()

        if len(parts) == 1:
            # Just /router - show status
            mode = self.router.speed_mode
            verbose = "verbose" if self.router_verbose else "concise"
            auto = "enabled" if self.auto_delegation_enabled else "disabled"
            OutputManager.print_info(
                f"Router status: {mode} mode, {verbose} output, auto-delegation {auto}"
            )
            return True

        if len(parts) == 2:
            subcommand = parts[1].lower()

            # Handle verbosity toggle
            if subcommand == "verbose":
                self.router_verbose = True
                OutputManager.print_info("Router set to verbose mode")
                return True
            elif subcommand == "concise":
                self.router_verbose = False
                OutputManager.print_info("Router set to concise mode")
                return True

            # Handle auto-delegation toggle
            elif subcommand == "auto":
                self.auto_delegation_enabled = True
                OutputManager.print_info("Auto-delegation enabled")
                return True
            elif subcommand == "manual":
                self.auto_delegation_enabled = False
                OutputManager.print_info(
                    "Auto-delegation disabled (will ask before delegating)"
                )
                return True

            # Handle speed modes
            elif subcommand in [mode.value for mode in RouterSpeedMode]:
                self.router.speed_mode = subcommand
                OutputManager.print_info(f"Router set to {subcommand} mode")
                return True

        return False

    def _handle_memory_command(self, command):
        """Handle memory-related commands."""
        parts = command.split(maxsplit=1)

        if len(parts) == 1:
            # Just /memory - show summary
            summary = self.memory_manager.get_memory_summary()
            OutputManager.print_info(summary)
            return True

        if len(parts) == 2:
            category = parts[1].lower()
            summary = self.memory_manager.get_memory_summary(category)
            OutputManager.print_info(summary)
            return True

        return False

    def handle_directly(self, query, memory_context=None):
        """Handle a query directly with the Executive Assistant."""
        start_time = time.time()

        # Prepare the query with memory context if available
        if memory_context:
            enhanced_query = f"{query}\n\n{memory_context}"
        else:
            enhanced_query = query

        # Display thinking message
        OutputManager.print_info("Handling directly...")

        # Create payload for the LLM API
        payload = {
            "model": MODEL_NAME,
            "prompt": enhanced_query,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "max_tokens": 500,
            },
            "system": self.system_prompt,
        }

        try:
            # Stream the response to show output as it's generated
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            full_response = ""

            # Display EA response start
            OutputManager.print_ea_response_prefix()

            # Process the streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line)
                if "response" in data:
                    chunk = data["response"]
                    full_response += chunk
                    OutputManager.print_response(chunk, end="")

                if data.get("done", False):
                    break

            # End the response line
            OutputManager.print_response("", end="\n")

            # Calculate response time
            response_time = time.time() - start_time
            OutputManager.print_info(
                f"Response completed in {response_time:.2f} seconds"
            )

            # Store the EA response in memory
            self.memory_manager.add_observation(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": OutputManager.format_timestamp(),
                }
            )

            return full_response

        except Exception as e:
            OutputManager.print_error(f"Error generating response: {e}")
            return None

    def generate_response(self, user_query):
        """Generate a response to a user query."""
        # Check for command to ask a specific agent
        if user_query.lower().startswith("/ask "):
            agent_name = self._get_agent_by_command(user_query)
            query = self._get_query_from_command(user_query)

            if not agent_name or not query:
                OutputManager.print_error("Usage: /ask <agent_name> <query>")
                return

            # Check if agent exists
            if agent_name not in self.registry.agent_processes:
                available = list(self.registry.agent_processes.keys())
                OutputManager.print_error(
                    f"Agent '{agent_name}' not found. Available agents: {', '.join(available)}"
                )
                return

            # Directly delegate to the specified agent
            self.delegate_to_agent(agent_name, query)

            # Store the response in memory
            self.memory_manager.add_observation(
                {
                    "role": "assistant",
                    "content": f"Delegated query to {agent_name} agent",
                    "timestamp": OutputManager.format_timestamp(),
                }
            )

            return

        # Check for router commands
        if user_query.lower().startswith("/router"):
            self._handle_router_command(user_query.lower())
            return

        # Check for memory commands
        if user_query.lower().startswith("/memory"):
            self._handle_memory_command(user_query.lower())
            return

        # For non-command queries, use auto-delegation or respond directly
        if self.auto_delegation_enabled:
            self.handle_auto_delegation(user_query)
        else:
            # Ask user if they want to delegate
            OutputManager.print_info(
                "Would you like me to route this query to a specialist agent? (y/n)"
            )
            response = input().lower()
            if response.startswith("y"):
                self.handle_auto_delegation(user_query)
            else:
                # Handle directly with the EA
                memory_context = self.memory_manager.get_memory_for_prompt(user_query)
                self.handle_directly(user_query, memory_context)

    def handle_auto_delegation(self, user_query):
        """Handle automatic delegation of queries to specialist agents."""
        start_time = time.time()

        # Get memory context that might be relevant to this query
        memory_context = self.memory_manager.get_memory_for_prompt(user_query)

        # Route the query to determine which agent should handle it
        OutputManager.print_info("Analyzing query...")

        result = self.router.route_query(
            user_prompt=user_query, memory_context=memory_context
        )

        # Extract the routed agent from the result
        if isinstance(result, dict) and "agent" in result:
            agent = result["agent"].lower()
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "No reasoning provided")

            routing_time = time.time() - start_time

            # Print debugging information if verbose
            if self.router_verbose:
                OutputManager.print_info(
                    f"Router selected: {agent} (confidence: {confidence:.2f})"
                )
                OutputManager.print_info(f"Reasoning: {reasoning}")
                OutputManager.print_info(f"Routing took {routing_time:.2f} seconds")

            # Special case for 'none' agent (no agents available)
            if agent == "none":
                OutputManager.print_info(
                    "No specialized agents available. Handling directly."
                )
                self.handle_directly(user_query, memory_context)
                return

            # Check if the agent exists
            if agent in self.registry.agent_processes:
                self.delegate_to_agent(agent, user_query, memory_context)
            else:
                OutputManager.print_error(
                    f"Selected agent '{agent}' not available. Handling directly."
                )
                # Fall back to EA handling the query directly
                self.handle_directly(user_query, memory_context)
        else:
            OutputManager.print_error("Failed to route query. Handling directly.")
            self.handle_directly(user_query, memory_context)

    def delegate_to_agent(self, agent_name, query, memory_context=None):
        """Delegate a query to a specific agent."""
        # Prepare response time tracking
        start_time = time.time()

        # Connect memory context if available
        if memory_context:
            # Add memory context to the query
            enhanced_query = f"{query}\n\n{memory_context}"
        else:
            enhanced_query = query

        # Display thinking animation
        OutputManager.print_info(f"Delegating to {agent_name}...")

        # Print the EA response prefix for the agent's response
        OutputManager.print_ea_response_prefix()

        # Callback for processing the streaming response
        def process_response(message):
            if message["type"] == "response":
                # Print the response chunk if there is one
                if "response" in message and message["response"]:
                    response_chunk = message["response"]
                    # Print the response without a newline
                    OutputManager.print_response(response_chunk, end="")

                    # Store the response in memory if it's not an error
                    if not message.get("is_error", False):
                        self.memory_manager.add_observation(
                            {
                                "role": "assistant",
                                "content": response_chunk,
                                "agent": agent_name,
                                "timestamp": OutputManager.format_timestamp(),
                            }
                        )

                # Add a newline after the final message
                if message.get("is_final", False):
                    OutputManager.print_response("", end="\n")

            elif message["type"] == "status":
                # Handle status updates
                if message.get("status") == "starting":
                    OutputManager.print_info(f"Agent {agent_name} is processing...")
                elif message.get("status") == "completed":
                    # This is handled by the response time output below
                    pass

        # Send the query to the agent
        response = self.registry.send_request_to_agent(
            agent_name, enhanced_query, process_response
        )

        # Calculate and display response time
        response_time = time.time() - start_time
        OutputManager.print_info(f"Response completed in {response_time:.2f} seconds")
