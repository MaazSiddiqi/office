#!/usr/bin/env python3

"""
AI Office v2 - Logger
====================

This module provides centralized logging functionality for the AI Office system.
"""

import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()


class Logger:
    """Centralized logging functionality for the AI Office system."""

    # Terminal UI colors and symbols
    COLORS = {
        "system": Fore.CYAN,
        "info": Fore.BLUE,
        "warning": Fore.YELLOW,
        "error": Fore.RED,
        "success": Fore.GREEN,
        "agent": Fore.MAGENTA,
        "ea": Fore.CYAN,
        "subtle": Fore.WHITE,
        "highlight": Fore.GREEN,
        "reset": Style.RESET_ALL,
    }

    SYMBOLS = {
        "arrow": "→",
        "error": "✗",
        "success": "✓",
        "warning": "⚠",
        "info": "ℹ",
        "agent": "🤖",
        "ea": "👨‍💼",
        "thinking": "💭",
        "divider": "─",
    }

    @classmethod
    def _print(
        cls, message: str, color: str, symbol: Optional[str] = None, end: str = "\n"
    ):
        """Internal method to print formatted messages."""
        prefix = f"{cls.SYMBOLS[symbol]} " if symbol else ""
        print(f"{cls.COLORS[color]}{prefix}{message}{cls.COLORS['reset']}", end=end)

    @classmethod
    def print_system(cls, message: str, end: str = "\n"):
        """Print system-level messages."""
        cls._print(message, "system", "info", end)

    @classmethod
    def print_info(cls, message: str, end: str = "\n"):
        """Print informational messages."""
        cls._print(message, "info", "info", end)

    @classmethod
    def print_warning(cls, message: str, end: str = "\n"):
        """Print warning messages."""
        cls._print(message, "warning", "warning", end)

    @classmethod
    def print_error(cls, message: str, end: str = "\n"):
        """Print error messages."""
        cls._print(message, "error", "error", end)

    @classmethod
    def print_success(cls, message: str, end: str = "\n"):
        """Print success messages."""
        cls._print(message, "success", "success", end)

    @classmethod
    def print_agent(cls, message: str, end: str = "\n"):
        """Print agent-specific messages."""
        cls._print(message, "agent", "agent", end)

    @classmethod
    def print_ea(cls, message: str, end: str = "\n"):
        """Print Executive Assistant messages."""
        cls._print(message, "ea", "ea", end)

    @classmethod
    def print_divider(cls, length: int = 40):
        """Print a visual divider line."""
        print(
            cls.COLORS["subtle"] + cls.SYMBOLS["divider"] * length + cls.COLORS["reset"]
        )

    @classmethod
    def format_timestamp(cls) -> str:
        """Format current timestamp for logging."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def display_agent_startup_animation(
        cls, stop_event: Any, agent_names: list, status_dict: Dict[str, Dict[str, str]]
    ):
        """Display an animated startup sequence for agents."""
        import time
        import sys

        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_idx = 0
        start_time = time.time()

        while not stop_event.is_set():
            sys.stdout.write("\033[2J\033[H")  # Clear screen
            print(
                f"\n{cls.COLORS['system']}Initializing AI Office Agents...{cls.COLORS['reset']}\n"
            )

            for agent_name in agent_names:
                status = status_dict.get(
                    agent_name, {"status": "waiting", "message": ""}
                )
                frame = frames[frame_idx]

                if status["status"] == "success":
                    print(
                        f"{cls.COLORS['success']}✓ {agent_name}: Ready{cls.COLORS['reset']}"
                    )
                elif status["status"] == "failed":
                    print(
                        f"{cls.COLORS['error']}✗ {agent_name}: Failed{cls.COLORS['reset']}"
                    )
                else:
                    print(
                        f"{cls.COLORS['info']}{frame} {agent_name}: {status['message']}{cls.COLORS['reset']}"
                    )

            frame_idx = (frame_idx + 1) % len(frames)
            time.sleep(0.1)

        # Final status display
        sys.stdout.write("\033[2J\033[H")
        print(
            f"\n{cls.COLORS['system']}AI Office Agents Initialized{cls.COLORS['reset']}\n"
        )

        for agent_name in agent_names:
            status = status_dict.get(agent_name, {"status": "unknown", "message": ""})
            if status["status"] == "success":
                print(
                    f"{cls.COLORS['success']}✓ {agent_name}: Ready{cls.COLORS['reset']}"
                )
            elif status["status"] == "failed":
                print(
                    f"{cls.COLORS['error']}✗ {agent_name}: Failed{cls.COLORS['reset']}"
                )
            else:
                print(
                    f"{cls.COLORS['warning']}⚠ {agent_name}: Unknown Status{cls.COLORS['reset']}"
                )

        print(f"\nInitialization completed in {time.time() - start_time:.1f} seconds")

    @staticmethod
    def print_welcome():
        """Print the welcome message."""
        print(f"\n{Fore.BLUE}Welcome to AI Office v2!{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Type 'exit' to quit{Style.RESET_ALL}\n")
