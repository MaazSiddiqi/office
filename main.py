#!/usr/bin/env python3

"""
AI Office v2 - Main Application
===============================

Main entry point for the AI Office application.
"""

import os
import datetime
from executive_assistant import ExecutiveAssistant
from output_manager import OutputManager


def main():
    """Main entry point for the AI Office application."""
    # Clear the terminal
    os.system("cls" if os.name == "nt" else "clear")

    # Print welcome message
    OutputManager.print_welcome(min(80, os.get_terminal_size().columns))

    # Initialize the Executive Assistant
    ea = ExecutiveAssistant()

    # Main interaction loop
    while True:
        # Get user input with timestamp
        user_input = input(OutputManager.print_user_input_prompt())

        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "/exit"]:
            OutputManager.print_warning("\nEnding session. Goodbye!")
            break

        # Generate and print response
        ea.generate_response(user_input)


if __name__ == "__main__":
    main()
