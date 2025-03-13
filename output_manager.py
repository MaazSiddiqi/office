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
        """Print a system message with appropriate styling."""
        print(f"{SYSTEM_COLOR}{message}{RESET}")

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

        # Future capabilities (not yet implemented)
        print()
        OutputManager.print_subtle("Coming Soon:")
        OutputManager.print_subtle(f"{BULLET} Delegating tasks to specialized agents")
        OutputManager.print_subtle(f"{BULLET} Parallel task execution")
        OutputManager.print_subtle(f"{BULLET} Centralized memory system")
        OutputManager.print_subtle(f"{BULLET} Feedback and improvement loops")

        # Commands
        print()
        OutputManager.print_highlight("Commands:")
        OutputManager.print_system_message(
            f"{BULLET} exit, quit, /exit {ARROW} End the session"
        )

        # Divider
        print()
        OutputManager.print_divider(width)
        print()
