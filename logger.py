#!/usr/bin/env python3

"""
AI Office - Logger
===============

This module provides centralized logging functionality.
"""

import sys
from colorama import init, Fore, Style
import datetime

# Initialize colorama for colored terminal output
init()


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


class Logger:
    """Provides centralized logging functionality."""

    @staticmethod
    def clear_line():
        """Clear the current line in the terminal."""
        # ANSI escape sequence to clear the line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    @staticmethod
    def format_timestamp():
        """Return a formatted timestamp string."""
        return datetime.datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def print_divider(width=60):
        """Print a divider line with specified width."""
        print(f"{SUBTLE_COLOR}{DIVIDER * width}{RESET}")

    @staticmethod
    def print_system(caller: str, message: str):
        """Print a system message with system color."""
        print(f"{SYSTEM_COLOR}[{caller}] {message}{RESET}")

    @staticmethod
    def print_system_message(message: str):
        """Print a system message with system color."""
        print(f"{SYSTEM_COLOR}{message}{RESET}")

    @staticmethod
    def print_llm_stream(message: str):
        """Print a stream message with system color."""
        print(f"{SYSTEM_COLOR}{message}{RESET}", end="", flush=True)

    @staticmethod
    def print_info(message: str):
        """Print an informational message."""
        print(f"{SYSTEM_COLOR}{BULLET} {message}{RESET}")

    @staticmethod
    def print_highlight(message: str):
        """Print a highlighted message."""
        print(f"{HIGHLIGHT_COLOR}{message}{RESET}")

    @staticmethod
    def print_warning(message: str):
        """Print a warning message."""
        print(f"{WARNING_COLOR}{message}{RESET}")

    @staticmethod
    def print_error(message: str):
        """Print an error message."""
        print(f"{ERROR_COLOR}{ERROR} {message}{RESET}")

    @staticmethod
    def print_success(message: str):
        """Print a success message."""
        print(f"{EA_COLOR}{CHECK} {message}{RESET}")

    @staticmethod
    def print_subtle(message: str):
        """Print a subtle/dimmed message."""
        print(f"{SUBTLE_COLOR}{message}{RESET}")

    @staticmethod
    def print_welcome(width=80):
        """Print a nicely formatted welcome message."""
        # Title banner
        Logger.print_divider(width)
        print(
            f"{HIGHLIGHT_COLOR}AI Office v2{RESET}{SYSTEM_COLOR} - Executive Assistant"
        )
        Logger.print_divider(width)

        # App description
        print()
        Logger.print_system_message("Welcome to your AI Office environment!")
        print("Your Executive Assistant (EA) is ready to help you.")

        # Current capabilities
        print()
        Logger.print_highlight("Current Capabilities:")
        Logger.print_system_message(f"{BULLET} Basic conversation with the EA")
        Logger.print_system_message(f"{BULLET} Conversation history tracking")
        Logger.print_system_message(f"{BULLET} Interact with specialized agents")
        Logger.print_system_message(
            f"{BULLET} Fast query routing to specialized agents"
        )
        Logger.print_system_message(
            f"{BULLET} 'Sticky' agent delegation (stay with an agent until topic changes)"
        )
        Logger.print_system_message(
            f"{BULLET} Long-term memory system for user preferences and information"
        )

        # Future capabilities (not yet implemented)
        print()
        Logger.print_subtle("Coming Soon:")
        Logger.print_subtle(f"{BULLET} Parallel task execution")
        Logger.print_subtle(f"{BULLET} Feedback and improvement loops")

        # Commands
        print()
        Logger.print_highlight("Commands:")
        Logger.print_system_message(
            f"{BULLET} /ask <agent> <query> {ARROW} Ask a specialized agent"
        )
        Logger.print_system_message(
            f"{BULLET} /reset or /ea {ARROW} Return control to the Executive Assistant"
        )
        Logger.print_system_message(f"{BULLET} /agents {ARROW} List available agents")
        Logger.print_system_message(
            f"{BULLET} /status {ARROW} Show detailed agent status"
        )
        Logger.print_system_message(
            f"{BULLET} /auto on|off {ARROW} Enable/disable automatic agent delegation"
        )
        Logger.print_system_message(
            f"{BULLET} /router {ARROW} Show router status and configuration"
        )
        Logger.print_system_message(
            f"{BULLET} /router verbose|fast {ARROW} Set router verbosity mode"
        )
        Logger.print_system_message(
            f"{BULLET} /router fastest|fast|accurate {ARROW} Set router speed/accuracy mode"
        )
        Logger.print_system_message(
            f"{BULLET} /memory {ARROW} View stored memory about you"
        )
        Logger.print_system_message(
            f"{BULLET} exit, quit, /exit {ARROW} End the session"
        )

        # Divider
        print()
        Logger.print_divider(width)
        print()
