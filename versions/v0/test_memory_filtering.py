#!/usr/bin/env python3

import time
from memory_manager import MemoryManagerProcess


def test_memory_filtering():
    """Test the improved memory filtering to ensure it only records important information."""
    print("Starting memory filtering test...")

    # Create memory manager
    memory_manager = MemoryManagerProcess()
    memory_manager.start()
    print(f"Memory manager started in process {memory_manager.pid}")

    # Wait for memory manager to initialize
    time.sleep(2)

    # Clear any existing memory
    print("Clearing memory...")
    memory_manager.task_queue.put({"type": "clear", "task_id": "clear-test"})

    # Wait for clear operation to complete
    time.sleep(2)
    while not memory_manager.result_queue.empty():
        result = memory_manager.result_queue.get()
        if result.get("task_id") == "clear-test":
            print("Memory cleared successfully")

    # Test cases for different types of conversations
    test_cases = [
        # Case 1: Important meeting information (should be stored)
        {
            "name": "Meeting details",
            "query": "I need to schedule a meeting with the team for tomorrow at 2pm to discuss the quarterly report.",
            "response": "I've noted your meeting with the team tomorrow at 2pm about the quarterly report.",
            "should_store": True,
        },
        # Case 2: Simple greeting (should be ignored)
        {
            "name": "Simple greeting",
            "query": "Hello, how are you today?",
            "response": "I'm doing well! How can I help you today?",
            "should_store": False,
        },
        # Case 3: User preference (important, should be stored)
        {
            "name": "User preference",
            "query": "I prefer to have meetings in the morning rather than afternoon.",
            "response": "I'll make note of your preference for morning meetings instead of afternoon ones.",
            "should_store": True,
        },
        # Case 4: General knowledge question (should be ignored)
        {
            "name": "General knowledge",
            "query": "What's the capital of France?",
            "response": "The capital of France is Paris. It's known as the City of Light and is famous for the Eiffel Tower, Louvre Museum, and its culinary traditions.",
            "should_store": False,
        },
        # Case 5: Contact information (important, should be stored)
        {
            "name": "Contact information",
            "query": "My new work email is john.smith@company.com and my phone is 555-123-4567.",
            "response": "I've recorded your new work email (john.smith@company.com) and phone number (555-123-4567).",
            "should_store": True,
        },
        # Case 6: Task with deadline (important, should be stored)
        {
            "name": "Task with deadline",
            "query": "Remember that I need to submit the project proposal by Friday at 5pm.",
            "response": "I'll remind you about the project proposal deadline this Friday at 5pm.",
            "should_store": True,
        },
        # Case 7: General chat (should be ignored)
        {
            "name": "General chat",
            "query": "I'm feeling tired today because I didn't sleep well.",
            "response": "I'm sorry to hear you didn't sleep well. It's important to get good rest. Perhaps you could try to get to bed earlier tonight.",
            "should_store": False,
        },
    ]

    # Process each test case
    for i, test in enumerate(test_cases, 1):
        print(f"\n----- Test {i}: {test['name']} -----")
        print(f"Query: {test['query']}")
        print(f"Should store facts: {'Yes' if test['should_store'] else 'No'}")

        # Send extraction task
        task_id = f"test-{i}"
        memory_manager.task_queue.put(
            {
                "type": "extract",
                "query": test["query"],
                "response": test["response"],
                "agent": "test_agent",
                "task_id": task_id,
            }
        )

        # Wait for extraction to complete
        time.sleep(5)  # Increase wait time

        # Drain all results and look for our task_id
        found_results = False
        facts = []

        while not memory_manager.result_queue.empty():
            result = memory_manager.result_queue.get()
            # Print all results for debugging
            print(f"Received result: {result}")

            if result.get("task_id") == task_id:
                facts = result.get("facts", [])
                found_results = True

                # Print extracted facts
                if facts:
                    print(f"Extracted {len(facts)} facts:")
                    for j, fact in enumerate(facts, 1):
                        print(f"  {j}. {fact}")
                else:
                    print("No facts extracted.")

                # Check if result matches expectation
                if bool(facts) == test["should_store"]:
                    print("✅ Result matches expectation")
                else:
                    print("❌ Result does NOT match expectation")

        if not found_results:
            print("❌ No response received from memory manager")

    # Wait for any pending extractions to complete
    print("\nWaiting for any pending extractions to complete...")
    time.sleep(5)

    # Drain any remaining results
    while not memory_manager.result_queue.empty():
        result = memory_manager.result_queue.get()
        print(f"Pending result: {result}")

    # Request current memory contents to verify overall storage
    print("\nRequesting final memory contents...")
    memory_manager.task_queue.put({"type": "read", "task_id": "final-read"})
    time.sleep(3)  # Increase wait time

    print("\n===== Final Memory Contents =====")
    found_memory = False

    while not memory_manager.result_queue.empty():
        result = memory_manager.result_queue.get()
        if result.get("task_id") == "final-read":
            memory = result.get("memory", {})
            facts = memory.get("facts", [])
            found_memory = True

            print(f"Total facts stored: {len(facts)}")
            for i, fact in enumerate(facts, 1):
                print(f"  {i}. {fact}")

    if not found_memory:
        print("Could not retrieve final memory contents")

    # Shutdown memory manager
    print("\nShutting down memory manager...")
    memory_manager.task_queue.put({"type": "shutdown"})
    memory_manager.join(timeout=5)
    print("Memory filtering test complete")


if __name__ == "__main__":
    test_memory_filtering()
