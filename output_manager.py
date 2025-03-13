#!/usr/bin/env python3

"""
AI Office v2 - Output Manager
============================

This module centralizes all terminal output formatting and display functionality.
It handles consistent styling, colors, and display patterns throughout the application.
"""

import sys
import time
import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init()

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


class OutputManager:
    """
    Handles all terminal output formatting and display for the AI Office system.
    Provides consistent styling and visual cues across the application.
    """

    @staticmethod
    def print_divider(width=60):
        """Print a divider line with specified width."""
        print(f"{SUBTLE_COLOR}{DIVIDER * width}{RESET}")

    @staticmethod
    def print_system_message(message):
        """Print a system message with system color."""
        print(f"{SYSTEM_COLOR}{message}{RESET}")

    @staticmethod
    def print_info(message):
        """Print an informational message."""
        print(f"{SYSTEM_COLOR}{BULLET} {message}{RESET}")

    @staticmethod
    def print_highlight(message):
        """Print a highlighted message."""
        print(f"{HIGHLIGHT_COLOR}{message}{RESET}")

    @staticmethod
    def print_warning(message):
        """Print a warning message."""
        print(f"{WARNING_COLOR}{message}{RESET}")

    @staticmethod
    def print_error(message):
        """Print an error message."""
        print(f"{ERROR_COLOR}{ERROR} {message}{RESET}")

    @staticmethod
    def print_success(message):
        """Print a success message."""
        print(f"{EA_COLOR}{CHECK} {message}{RESET}")

    @staticmethod
    def print_subtle(message):
        """Print a subtle/dimmed message."""
        print(f"{SUBTLE_COLOR}{message}{RESET}")

    @staticmethod
    def print_response(chunk, end=None):
        """Print a response chunk from an agent or EA with appropriate styling."""
        # If end is specified, use it, otherwise default to newline
        if end is not None:
            print(f"{EA_COLOR}{chunk}{RESET}", end=end, flush=True)
        else:
            print(f"{EA_COLOR}{chunk}{RESET}", flush=True)

    @staticmethod
    def format_timestamp():
        """Return a formatted timestamp string."""
        return datetime.datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def display_thinking_animation(duration=1.0):
        """Display a simple thinking animation."""
        start_time = time.time()
        i = 0
        thinking_states = [
            "/",
            "-",
            "\\",
            "|",
        ]  # Simpler animation that works in all terminals

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

    @staticmethod
    def display_agent_startup_animation(stop_event, agent_names, status_dict=None):
        """
        Display an animation while agents are starting up.

        Args:
            stop_event (threading.Event): Event to signal when to stop the animation
            agent_names (list): List of agent names that are starting
            status_dict (dict): Dictionary to track the current status of each agent
        """
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        # Fallback for terminals that don't support unicode
        fallback_chars = ["-", "\\", "|", "/"]

        # Initialize status dictionary if not provided
        if status_dict is None:
            status_dict = {}

        # Set up status tracking
        agent_statuses = {}
        for name in agent_names:
            # Default status is "pending"
            agent_statuses[name] = {
                "status": status_dict.get(name, {}).get("status", "pending"),
                "message": status_dict.get(name, {}).get("message", ""),
            }

        try:
            chars = spinner_chars
            # Test if unicode works
            sys.stdout.write(chars[0])
            sys.stdout.flush()
            sys.stdout.write("\b")
        except:
            chars = fallback_chars

        i = 0
        last_status_length = 0

        while not stop_event.is_set():
            # Format status line
            status_line = f"{SUBTLE_COLOR}Starting agents: "

            # Update status from shared dictionary if provided
            if status_dict:
                for name in agent_names:
                    if name in status_dict:
                        agent_statuses[name]["status"] = status_dict[name].get(
                            "status", agent_statuses[name]["status"]
                        )
                        agent_statuses[name]["message"] = status_dict[name].get(
                            "message", agent_statuses[name]["message"]
                        )

            # Add agent status indicators
            for name in agent_names:
                status = agent_statuses[name]["status"]
                if status == "pending":
                    indicator = chars[i % len(chars)]
                elif status == "testing":
                    indicator = "🔍" if chars == spinner_chars else "T"
                elif status == "success":
                    indicator = "✓" if chars == spinner_chars else "+"
                elif status == "failed":
                    indicator = "✗" if chars == spinner_chars else "X"
                else:
                    indicator = "?"

                # Add the indicator and name
                status_line += f"{name}: {indicator}  "

            # Add current action if applicable
            current_action = None
            for name in agent_names:
                if (
                    agent_statuses[name]["message"]
                    and agent_statuses[name]["status"] != "success"
                    and agent_statuses[name]["status"] != "failed"
                ):
                    current_action = agent_statuses[name]["message"]
                    break

            if current_action:
                status_line += f"| {current_action}"

            # Clear the line completely
            sys.stdout.write("\r")
            sys.stdout.write(" " * last_status_length)
            sys.stdout.write("\r")

            # Print status line and save its length
            sys.stdout.write(status_line + RESET)
            sys.stdout.flush()
            last_status_length = len(status_line) + len(RESET)

            # Update counter and sleep
            i += 1
            time.sleep(0.1)

        # Clear the line when done
        OutputManager.clear_line()

    @staticmethod
    def clear_line():
        """Clear the current line in the terminal."""
        # ANSI escape sequence to clear the line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    @staticmethod
    def print_user_input_prompt():
        """Print the user input prompt with timestamp."""
        timestamp = OutputManager.format_timestamp()
        return f"{SUBTLE_COLOR}[{timestamp}] {USER_COLOR}You {ARROW} {RESET}"

    @staticmethod
    def print_ea_response_prefix():
        """Print the EA response prefix with timestamp."""
        timestamp = OutputManager.format_timestamp()
        print(
            f"{SUBTLE_COLOR}[{timestamp}] {EA_COLOR}EA {ARROW} {RESET}",
            end="",
            flush=True,
        )

    @staticmethod
    def print_agent_response_prefix(agent_name):
        """Print an agent response prefix with timestamp."""
        timestamp = OutputManager.format_timestamp()
        print(
            f"{SUBTLE_COLOR}[{timestamp}] {EA_COLOR}EA ({agent_name}) {ARROW} {RESET}",
            end="",
            flush=True,
        )

    @staticmethod
    def print_welcome(width=80):
        """Print a nicely formatted welcome message."""
        # Title banner
        OutputManager.print_divider(width)
        print(
            f"{HIGHLIGHT_COLOR}AI Office v2{RESET}{SYSTEM_COLOR} - Executive Assistant"
        )
        OutputManager.print_divider(width)

        # App description
        print()
        OutputManager.print_system_message("Welcome to your AI Office environment!")
        print("Your Executive Assistant (EA) is ready to help you.")

        # Current capabilities
        print()
        OutputManager.print_highlight("Current Capabilities:")
        OutputManager.print_system_message(f"{BULLET} Basic conversation with the EA")
        OutputManager.print_system_message(f"{BULLET} Conversation history tracking")
        OutputManager.print_system_message(f"{BULLET} Interact with specialized agents")
        OutputManager.print_system_message(
            f"{BULLET} Fast query routing to specialized agents"
        )
        OutputManager.print_system_message(
            f"{BULLET} 'Sticky' agent delegation (stay with an agent until topic changes)"
        )
        OutputManager.print_system_message(
            f"{BULLET} Long-term memory system for user preferences and information"
        )

        # Future capabilities (not yet implemented)
        print()
        OutputManager.print_subtle("Coming Soon:")
        OutputManager.print_subtle(f"{BULLET} Parallel task execution")
        OutputManager.print_subtle(f"{BULLET} Feedback and improvement loops")

        # Commands
        print()
        OutputManager.print_highlight("Commands:")
        OutputManager.print_system_message(
            f"{BULLET} /ask <agent> <query> {ARROW} Ask a specialized agent"
        )
        OutputManager.print_system_message(
            f"{BULLET} /reset or /ea {ARROW} Return control to the Executive Assistant"
        )
        OutputManager.print_system_message(
            f"{BULLET} /agents {ARROW} List available agents"
        )
        OutputManager.print_system_message(
            f"{BULLET} /status {ARROW} Show detailed agent status"
        )
        OutputManager.print_system_message(
            f"{BULLET} /auto on|off {ARROW} Enable/disable automatic agent delegation"
        )
        OutputManager.print_system_message(
            f"{BULLET} /router {ARROW} Show router status and configuration"
        )
        OutputManager.print_system_message(
            f"{BULLET} /router verbose|fast {ARROW} Set router verbosity mode"
        )
        OutputManager.print_system_message(
            f"{BULLET} /router fastest|fast|accurate {ARROW} Set router speed/accuracy mode"
        )
        OutputManager.print_system_message(
            f"{BULLET} /memory {ARROW} View stored memory about you"
        )
        OutputManager.print_system_message(
            f"{BULLET} exit, quit, /exit {ARROW} End the session"
        )

        # Divider
        print()
        OutputManager.print_divider(width)
        print()
