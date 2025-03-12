#!/usr/bin/env python3

import time
from memory_manager import MemoryManagerProcess


def main():
    print("Starting memory manager test...")

    # Create memory manager
    memory_manager = MemoryManagerProcess()
    memory_manager.start()
    print(f"Memory manager started in process {memory_manager.pid}")

    # Wait for memory manager to initialize
    time.sleep(1)

    # Send a test extraction task with pre-extracted facts
    print("Sending test extraction task with pre-extracted facts...")
    memory_manager.task_queue.put(
        {
            "type": "extract",
            "query": "I have three meetings tomorrow at 9am, 11am, and 2pm.",
            "response": "I'll make note of your three meetings tomorrow at 9am, 11am, and 2pm.",
            "agent": "test_agent",
            "task_id": "test-123",
            "facts": [
                "The user has three meetings tomorrow.",
                "The meetings are scheduled at 9am, 11am, and 2pm.",
            ],
        }
    )

    # Wait for extraction to complete
    print("Waiting for extraction to complete...")
    time.sleep(5)

    # Check for results
    print("Checking for results...")
    results = []
    while not memory_manager.result_queue.empty():
        result = memory_manager.result_queue.get()
        results.append(result)
        print(f"Got result: {result}")

    if not results:
        print("No results received!")

    # Shutdown memory manager
    print("Shutting down memory manager...")
    memory_manager.task_queue.put({"type": "shutdown"})
    memory_manager.join(timeout=5)
    print("Memory manager test complete")


if __name__ == "__main__":
    main()
