#!/usr/bin/env python3

"""
AI Office - Logger
===============

This module provides centralized logging functionality.
"""

import sys
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()


class Logger:
    """Provides centralized logging functionality."""

    @staticmethod
    def print_info(message: str):
        """Print an informational message."""
        print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {message}")

    @staticmethod
    def print_warning(message: str):
        """Print a warning message."""
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}")

    @staticmethod
    def print_error(message: str):
        """Print an error message."""
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}", file=sys.stderr)

    @staticmethod
    def print_success(message: str):
        """Print a success message."""
        print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {message}")
