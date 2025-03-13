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
from output_manager import OutputManager, SUBTLE_COLOR, RESET, EA_COLOR, ARROW
from agent_registry import get_registry

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

        # Define a callback to handle streaming responses
        def handle_response_chunk(message):
            nonlocal is_printing, last_chunk_newline

            if message["type"] == "status":
                if message["status"] == "starting":
                    # Status updates are printed in subtle style
                    print(f"{SUBTLE_COLOR}{message['message']}{RESET}")

            elif message["type"] == "response":
                # Regular responses are printed normally
                chunk = message.get("response", "")

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

    def generate_response(self, user_input):
        """
        Generate a response from the EA based on user input.

        Args:
            user_input (str): The user's message

        Returns:
            str: The EA's response
        """
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

        # Add user input to conversation history with timestamp
        timestamp = OutputManager.format_timestamp()
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
