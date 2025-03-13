#!/usr/bin/env python3

"""
AI Office v2 - Main Application
===============================

Main entry point for the AI Office application.
"""

import os
import datetime
import atexit
import signal
import sys
from executive_assistant import ExecutiveAssistant
from output_manager import OutputManager
from agent_registry import get_registry


def shutdown_handler():
    """Clean up resources on program exit."""
    OutputManager.print_warning("\nCleaning up resources...")

    # Shutdown all agent processes
    registry = get_registry()
    registry.shutdown_agents()


def main():
    """Main entry point for the AI Office application."""
    # Clear the terminal
    os.system("cls" if os.name == "nt" else "clear")

    # Print welcome message
    OutputManager.print_welcome(min(80, os.get_terminal_size().columns))

    # Register the shutdown handler
    atexit.register(shutdown_handler)

    # Custom signal handler for cleaner exit
    def handle_interrupt(sig, frame):
        print("\nReceived interrupt signal. Shutting down...")
        shutdown_handler()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # Initialize the Executive Assistant
    ea = ExecutiveAssistant()

    # Initialize and start agent processes with animation
    registry = get_registry()
    registry.start_agent_processes(use_animation=True)

    # Display available agents
    agents = registry.list_available_agents()
    running_agents = list(registry.agent_processes.keys())

    # Add command info for agent interaction if there are agents
    if running_agents:
        # Display routing information
        print("Router commands:")
        print("* /router → Show router status")
        print("* /router verbose | fast → Set verbosity mode")
        print("* /router fastest | fast | accurate → Set speed/accuracy mode")
        print()
    elif agents:
        OutputManager.print_warning("No agents are currently running")

    print()  # Add a blank line for spacing

    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input(f"[{OutputManager.format_timestamp()}] You -> ")

            # Handle special case: empty input
            if not user_input.strip():
                continue

            # Handle exit commands
            if user_input.lower() in ["exit", "quit", "/exit"]:
                print("Ending session. Goodbye!")
                break

            # Check for agent listing command
            elif user_input.lower() == "/agents":
                running_agents = list(registry.agent_processes.keys())
                if running_agents:
                    OutputManager.print_success(
                        f"Running agents: {', '.join(running_agents)}"
                    )
                else:
                    OutputManager.print_warning("No agents are currently running")
                continue

            # Check for agent status command
            elif user_input.lower() == "/status":
                OutputManager.print_info("Agent Status:")

                # Show configured agents
                all_agents = registry.list_available_agents()
                running_agents = list(registry.agent_processes.keys())

                for agent in all_agents:
                    if agent in running_agents:
                        agent_process = registry.agent_processes[agent]
                        pid = agent_process.pid
                        alive = "✓ RUNNING" if agent_process.is_alive() else "✗ STOPPED"
                        model = registry.agents[agent].get("model", "unknown")
                        OutputManager.print_success(
                            f"{agent} (PID {pid}): {alive}, Model: {model}"
                        )
                    else:
                        OutputManager.print_warning(f"{agent}: ✗ NOT RUNNING")

                # Show agent command help
                print()
                OutputManager.print_system_message("Agent Commands:")
                OutputManager.print_system_message(
                    "/ask <agent> <query> → Ask an agent a question"
                )
                OutputManager.print_system_message("/agents → List running agents")
                OutputManager.print_system_message(
                    "/status → Show detailed agent status"
                )
                continue

            # Process the input
            ea.generate_response(user_input)
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt.")
            continue_session = input("Do you want to exit? (y/n): ").lower()
            if continue_session.startswith("y"):
                print("Ending session. Goodbye!")
                break
            print("Continuing session...")
        except Exception as e:
            OutputManager.print_error(f"Error processing input: {e}")

    # Clean exit will trigger the shutdown_handler via atexit


if __name__ == "__main__":
    main()
