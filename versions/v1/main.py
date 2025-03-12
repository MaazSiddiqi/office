#!/usr/bin/env python3

"""
AI Office - Main Entry Point
This script can either run the original single-process AI Office or the new multi-process version.
"""

import os
import sys
import time
import select


def main():
    # Check if we want to run the multi-process version (default)
    use_multiprocess = os.environ.get("AI_OFFICE_MULTIPROCESS", "1") == "1"

    if use_multiprocess:
        print("Starting AI Office in multi-process mode...")
        # Import and run the multi-process version
        try:
            from multi_agent_office import MultiAgentOffice

            office = MultiAgentOffice()

            print("\nAvailable Agents:")
            for agent_name in office.registry.keys():
                print(f"- {agent_name}")

            # Print information about conversation sessions
            print("\nConversation Session Commands:")
            print("- /who - Shows which agent you're currently talking to")
            print(
                "- /switch_to <agent> - Explicitly switch to talking with a specific agent"
            )
            print("- /ea - Switch back to talking with the Executive Assistant")
            print(
                "Note: The system will automatically maintain conversation continuity with agents"
            )

            try:
                while True:
                    # Simple input approach
                    user_prompt = input("\n> ")

                    if user_prompt.lower() in ["quit", "exit", "q"]:
                        break

                    # Check for memory updates before processing
                    office.check_memory_updates()

                    # Process the user's query
                    response = office.process_query(user_prompt)
                    print(response)

                    # Check for memory updates after processing
                    office.check_memory_updates()

                    # Also periodically check while waiting for next input
                    for _ in range(3):
                        time.sleep(0.5)
                        office.check_memory_updates()

            finally:
                # Ensure we shut down all agent processes when exiting
                print("\nShutting down agent processes...")
                office.shutdown_agent_processes()
                print("Goodbye!")

        except ImportError as e:
            print(f"Error importing multi-process module: {e}")
            print("Falling back to single-process mode.")
            run_single_process()
    else:
        # Run the original single-process version
        run_single_process()


def run_single_process():
    print("Starting AI Office in single-process mode...")
    # Import and run the original single-process version
    from legacy_main import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    main()
