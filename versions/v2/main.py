#!/usr/bin/env python3

"""
AI Office v2 - Main Application
=============================

This is the main entry point for the AI Office system.
"""

import os
import sys
import signal
import time
from typing import Optional
from logger import Logger
from agent_registry import AgentRegistry
from query_router import QueryRouter
from memory_manager import get_memory_manager


def handle_shutdown(signum: int, frame):
    """Handle shutdown signals gracefully."""
    Logger.print_warning("\nShutting down AI Office...")
    sys.exit(0)


def main():
    """Main entry point for the AI Office system."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Initialize components
    try:
        # Create registry first as it's a dependency for other components
        registry = AgentRegistry()

        # Initialize other components with dependencies
        router = QueryRouter(registry)
        memory_manager = get_memory_manager()
    except Exception as e:
        Logger.print_error(f"Failed to initialize components: {e}")
        sys.exit(1)

    # Start agent processes
    try:
        registry.start_agent_processes()
    except Exception as e:
        Logger.print_error(f"Failed to start agent processes: {e}")
        sys.exit(1)

    # Main interaction loop
    Logger.print_welcome()
    while True:
        try:
            # Get user input
            user_input = input(f"{Logger.format_timestamp()} > ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                Logger.print_warning("Shutting down AI Office...")
                break

            # Route query to appropriate agent
            agent_name = router.route_query(user_input)
            if not agent_name:
                Logger.print_warning("No suitable agent found for your query")
                continue

            # Get agent response
            response = registry.send_request_to_agent(agent_name, user_input)
            if response:
                Logger.print_success(response)
            else:
                Logger.print_error("Failed to get response from agent")

        except KeyboardInterrupt:
            Logger.print_warning("\nShutting down AI Office...")
            break
        except Exception as e:
            Logger.print_error(f"Error in main loop: {e}")

    # Cleanup
    try:
        registry.shutdown_agents()
    except Exception as e:
        Logger.print_error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    main()
