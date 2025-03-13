#!/usr/bin/env python3

"""
AI Office v2 - Executive Assistant
=================================

Simple terminal interface for interacting with the Executive Assistant.
"""

import os
import json
import requests
import time
import datetime
import colorama
import sys
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Configuration
MODEL_NAME = "llama3.1:latest"  # Using local LLM
API_URL = "http://localhost:11434/api/generate"  # Local Ollama API

# Terminal UI colors
SYSTEM_COLOR = Fore.CYAN
USER_COLOR = Fore.BLUE
EA_COLOR = Fore.GREEN
ERROR_COLOR = Fore.RED
WARNING_COLOR = Fore.YELLOW
HIGHLIGHT_COLOR = Fore.MAGENTA
SUBTLE_COLOR = Fore.WHITE + Style.DIM
RESET = Style.RESET_ALL

# Symbols for better visual cues (using ASCII compatible versions)
BULLET = "*"
ARROW = "->"
DIVIDER = "-"
CHECK = "+"
ERROR = "!"
THINKING = "..."
PROMPT = ">"

class ExecutiveAssistant:
    """
    Executive Assistant (EA) that serves as the primary interface for the user.
    This is a simplified version focusing only on basic conversation.
    """

    def __init__(self):
        """Initialize the Executive Assistant."""
        self.conversation_history = []

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

    def display_thinking_animation(self, duration=1.0):
        """Display a simple thinking animation."""
        start_time = time.time()
        i = 0
        thinking_states = ["/", "-", "\\", "|"]  # Simpler animation that works in all terminals

        print(f"{SUBTLE_COLOR}", end="")
        sys.stdout.write("Thinking ")

        while (time.time() - start_time) < duration:
            sys.stdout.write(f"\b{thinking_states[i % len(thinking_states)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

        sys.stdout.write("\b \b" * 9)  # Clear the "Thinking X" text
        sys.stdout.flush()
        print(f"{RESET}", end="")

    def generate_response(self, user_input):
        """
        Generate a response from the EA based on user input.

        Args:
            user_input (str): The user's message

        Returns:
            str: The EA's response
        """
        # Add user input to conversation history with timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        # Prepare conversation context
        conversation_context = ""
        if len(self.conversation_history) > 1:
            conversation_context = "Previous conversation:\n"
            # Include up to the last 5 exchanges for context
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
        self.display_thinking_animation()

        try:
            # Call the LLM API with streaming to show output as it's generated
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            full_response = ""

            # Display EA label with timestamp
            ea_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"{SUBTLE_COLOR}[{ea_timestamp}] {EA_COLOR}EA {ARROW} {RESET}", end="", flush=True)

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
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": ea_timestamp
            })

            # Add a subtle divider
            print(f"{SUBTLE_COLOR}{DIVIDER * 60}{RESET}")

            return full_response

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"{ERROR_COLOR}{ERROR} {error_msg}{RESET}")
            print(f"{SUBTLE_COLOR}{DIVIDER * 60}{RESET}")
            return error_msg

def print_welcome():
    """Print a nicely formatted welcome message."""
    terminal_width = os.get_terminal_size().columns
    width = min(80, terminal_width)

    # Title banner
    print(f"{SYSTEM_COLOR}{DIVIDER * width}")
    print(f"{HIGHLIGHT_COLOR}AI Office v2{RESET}{SYSTEM_COLOR} - Executive Assistant")
    print(f"{DIVIDER * width}{RESET}")

    # App description
    print(f"\n{SYSTEM_COLOR}Welcome to your AI Office environment!{RESET}")
    print("Your Executive Assistant (EA) is ready to help you.")

    # Current capabilities
    print(f"\n{HIGHLIGHT_COLOR}Current Capabilities:{RESET}")
    print(f"{SYSTEM_COLOR}{BULLET} Basic conversation with the EA{RESET}")
    print(f"{SYSTEM_COLOR}{BULLET} Conversation history tracking{RESET}")

    # Future capabilities (not yet implemented)
    print(f"\n{SUBTLE_COLOR}Coming Soon:{RESET}")
    print(f"{SUBTLE_COLOR}{BULLET} Delegating tasks to specialized agents{RESET}")
    print(f"{SUBTLE_COLOR}{BULLET} Parallel task execution{RESET}")
    print(f"{SUBTLE_COLOR}{BULLET} Centralized memory system{RESET}")
    print(f"{SUBTLE_COLOR}{BULLET} Feedback and improvement loops{RESET}")

    # Commands
    print(f"\n{HIGHLIGHT_COLOR}Commands:{RESET}")
    print(f"{SYSTEM_COLOR}{BULLET} exit, quit, /exit {ARROW} End the session{RESET}")

    # Divider
    print(f"\n{SYSTEM_COLOR}{DIVIDER * width}{RESET}\n")

def main():
    """Main entry point for the AI Office application."""
    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Print welcome message
    print_welcome()

    ea = ExecutiveAssistant()

    while True:
        # Get user input with timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        user_input = input(f"{SUBTLE_COLOR}[{timestamp}] {USER_COLOR}You {PROMPT} {RESET}")

        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "/exit"]:
            print(f"\n{WARNING_COLOR}Ending session. Goodbye!{RESET}")
            break

        # Generate and print response
        ea.generate_response(user_input)

if __name__ == "__main__":
    main()
