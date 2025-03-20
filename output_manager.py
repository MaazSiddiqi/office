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
        try:
            # If end is specified, use it, otherwise default to newline
            if end is not None:
                print(f"{EA_COLOR}{chunk}{RESET}", end=end, flush=True)
                # Force flush again to ensure output is displayed immediately
                sys.stdout.flush()
            else:
                print(f"{EA_COLOR}{chunk}{RESET}", flush=True)
                sys.stdout.flush()
        except Exception as e:
            # If there's an error, try a more direct approach
            sys.stdout.write(f"{EA_COLOR}{chunk}{RESET}")
            if end is None:
                sys.stdout.write("\n")
            sys.stdout.flush()

    @staticmethod
    def debug_stream():
        """Test if streaming output is working."""
        print("Testing streaming output... ", end="", flush=True)
        for char in "This text should appear character by character.":
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.05)
        print(" Done!")

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
        """Print the agent response prefix with timestamp."""
        timestamp = OutputManager.format_timestamp()
        print(
            f"{SUBTLE_COLOR}[{timestamp}] {EA_COLOR}EA ({agent_name}) {ARROW} {RESET}",
            end="",
            flush=True,
        )

    @staticmethod
    def print_welcome(width=80):
        """Print the welcome message for the AI Office."""
        OutputManager.print_divider(width)
        OutputManager.print_system_message("AI Office v2 - Executive Assistant")
        OutputManager.print_divider(width)

        # Multi-line welcome message with core information
        welcome_message = """Welcome to your AI Office environment!
Your Executive Assistant (EA) is ready to help you.

Current Capabilities:
* Basic conversation with the EA
* Conversation history tracking
* Collaborative specialist consultations - watch the EA & specialists work together in real-time
* Intelligent query routing for specialist knowledge
* 'Sticky' context (stay with a specialist topic until it changes)
* Long-term memory system for user preferences and information

Coming Soon:
* Parallel task execution
* Feedback and improvement loops

Commands:
* /ask <agent> <query> -> Request the EA to collaborate with a specific specialist
* /reset or /ea -> Return to general EA conversation
* /agents -> List available specialists
* /status -> Show detailed system status
* /auto on|off -> Enable/disable automatic specialist consultations
* /router -> Show router status and configuration
* /router verbose|fast -> Set verbosity mode
* /router fastest|fast|accurate -> Set speed/accuracy mode
* /memory -> View stored memory about you
* /debug stream -> Test if streaming output is working properly
* exit, quit, /exit -> End the session"""

        print(welcome_message)
        OutputManager.print_divider(width)

    @staticmethod
    def print_internal_consultation_start(agent_name):
        """Print the start of an internal consultation between EA and an agent."""
        OutputManager.print_divider(80)
        print(
            f"{HIGHLIGHT_COLOR}┌─ COLLABORATIVE CONSULTATION WITH {agent_name.upper()} ───{RESET}"
        )
        OutputManager.print_divider(80)

    @staticmethod
    def print_internal_consultation_end():
        """Print the end of an internal consultation."""
        OutputManager.print_divider(80)
        print(f"{HIGHLIGHT_COLOR}└─ END OF COLLABORATIVE CONSULTATION ───{RESET}")
        OutputManager.print_divider(80)

    @staticmethod
    def print_consultation_message(sender, message):
        """Print a message in the internal consultation with proper formatting."""
        # Clean up the message for display
        clean_message = message.replace("[INTERNAL CONSULTATION]", "").strip()

        # Split into lines for better display
        lines = clean_message.split("\n")

        # Print the sender with a clear prefix
        if sender == "EA":
            print(f"\n{SUBTLE_COLOR}┌── EA:{RESET}")
        else:
            print(f"\n{HIGHLIGHT_COLOR}┌── {sender}:{RESET}")

        # Print each line of the message with proper indentation
        for line in lines:
            if sender == "EA":
                print(f"{SUBTLE_COLOR}│ {line}{RESET}")
            else:
                print(f"{HIGHLIGHT_COLOR}│ {line}{RESET}")

        # Add a closing line
        if sender == "EA":
            print(f"{SUBTLE_COLOR}└───────{RESET}")
        else:
            print(f"{HIGHLIGHT_COLOR}└───────{RESET}")

    @staticmethod
    def print_collaborative_summary(agent_name):
        """Print that the EA is incorporating insights from the agent."""
        OutputManager.print_divider(80)
        print(
            f"{SYSTEM_COLOR}* Incorporating insights from collaboration with {agent_name}...{RESET}"
        )
        OutputManager.print_divider(40)
