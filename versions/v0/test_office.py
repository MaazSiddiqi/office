#!/usr/bin/env python3

import time
from multi_agent_office import MultiAgentOffice


def main():
    print("Starting MultiAgentOffice test...")

    # Create office
    office = MultiAgentOffice()
    print("Office created")

    # Wait for initialization
    time.sleep(2)

    # Manually add a fact to memory
    print("\nManually adding facts to memory...")
    task_id = "test-456"
    office.memory_manager.task_queue.put(
        {
            "type": "extract",
            "query": "I have three meetings tomorrow at 9am, 11am, and 2pm.",
            "response": "I'll make note of your three meetings tomorrow at 9am, 11am, and 2pm.",
            "agent": "test_agent",
            "task_id": task_id,
            "facts": [
                "The user has three meetings tomorrow.",
                "The meetings are scheduled at 9am, 11am, and 2pm.",
            ],
        }
    )

    # Wait for extraction to complete
    print("Waiting for extraction to complete...")
    time.sleep(3)

    # Check for memory updates
    print("\nChecking for memory updates...")
    office.check_memory_updates()

    # Wait a bit more and check again
    print("\nWaiting and checking again...")
    time.sleep(2)
    office.check_memory_updates()

    # Shutdown
    print("\nShutting down office...")
    office.shutdown_agent_processes()
    print("Test complete")


if __name__ == "__main__":
    main()
