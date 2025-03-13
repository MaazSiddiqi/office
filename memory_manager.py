#!/usr/bin/env python3

"""
Memory Manager for AI Office
============================

This module provides a memory management system that:
1. Runs in its own process
2. Eavesdrops on conversations
3. Extracts key information about user (preferences, characteristics)
4. Writes to a file on disk asynchronously
5. Loads memory from this file on startup
"""

import os
import json
import time
import datetime
from pathlib import Path
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Any, Union
from output_manager import OutputManager

# Configuration
MEMORY_DIR = Path("./memory")
MEMORY_FILE = MEMORY_DIR / "user_memory.json"
MEMORY_CATEGORIES = [
    "preferences",
    "characteristics",
    "background",
    "work",
    "projects",
    "schedule",
    "relationships",
    "technical_knowledge",
]

# Global instance
_memory_manager_instance = None


def get_memory_manager():
    """Get or create the global MemoryManager instance."""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance


class MemoryProcess(Process):
    """Process for managing memory asynchronously."""

    def __init__(self, memory_file: Path, input_queue: Queue):
        super().__init__(daemon=True)
        self.memory_file = memory_file
        self.input_queue = input_queue
        self.memories = {}  # Initialize empty, will load in run()

    def _load_memories_from_file(self) -> Dict:
        """Load memories from disk."""
        if not self.memory_file.exists():
            # Initialize with empty categories
            return {category: [] for category in MEMORY_CATEGORIES}

        try:
            with open(self.memory_file, "r") as f:
                memories = json.load(f)

            # Ensure all categories exist
            for category in MEMORY_CATEGORIES:
                if category not in memories:
                    memories[category] = []

            return memories
        except Exception as e:
            OutputManager.print_error(f"Error loading memory file: {e}")
            return {category: [] for category in MEMORY_CATEGORIES}

    def _save_memories_to_file(self):
        """Save memories to disk."""
        # Ensure directory exists
        self.memory_file.parent.mkdir(exist_ok=True)

        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            OutputManager.print_error(f"Error saving memory file: {e}")

    def _extract_information(self, observation: Dict) -> Dict[str, List[str]]:
        """Extract information from an observation using simple keyword matching.

        This method is a simplified version that doesn't rely on the QueryRouter,
        avoiding the pickling issues with thread locks.
        """
        content = observation.get("content", "")

        # Skip extraction for empty or command messages
        if not content or content.startswith("/"):
            return {}

        # Very simple keyword-based extraction
        extracted_info = {category: [] for category in MEMORY_CATEGORIES}

        # Convert to lowercase for better matching
        content_lower = content.lower()

        # Name detection patterns
        name_patterns = [
            "my name is ",
            "i am called ",
            "i go by ",
            "call me ",
            "name's ",
        ]

        for pattern in name_patterns:
            if pattern in content_lower:
                idx = content_lower.find(pattern) + len(pattern)
                # Extract the name - look for the end of the sentence or a delimiter
                end_markers = [".", ",", ";", "!", "?", " and ", " but "]
                end_idx = len(content)
                for marker in end_markers:
                    marker_idx = content_lower[idx:].find(marker)
                    if marker_idx != -1 and marker_idx < end_idx:
                        end_idx = idx + marker_idx

                name = content[idx:end_idx].strip()
                if name and len(name) > 1 and len(name) < 30:  # Basic validation
                    extracted_info["characteristics"].append(
                        f"The user's name is {name}"
                    )
                break  # Found a name, no need to check other patterns

        # Preference patterns
        if (
            "i like" in content_lower
            or "i love" in content_lower
            or "i enjoy" in content_lower
        ):
            for pref_pattern in ["i like ", "i love ", "i enjoy "]:
                if pref_pattern in content_lower:
                    idx = content_lower.find(pref_pattern) + len(pref_pattern)
                    end_idx = len(content)
                    for marker in [".", ",", ";", "!", "?", " but ", " and "]:
                        marker_idx = content_lower[idx:].find(marker)
                        if marker_idx != -1 and marker_idx < end_idx - idx:
                            end_idx = idx + marker_idx

                    preference = content[idx:end_idx].strip()
                    if preference and len(preference) > 1:
                        extracted_info["preferences"].append(
                            f"The user likes {preference}"
                        )

        # Dislike patterns
        if (
            "i don't like" in content_lower
            or "i hate" in content_lower
            or "i dislike" in content_lower
        ):
            for pref_pattern in [
                "i don't like ",
                "i do not like ",
                "i hate ",
                "i dislike ",
            ]:
                if pref_pattern in content_lower:
                    idx = content_lower.find(pref_pattern) + len(pref_pattern)
                    end_idx = len(content)
                    for marker in [".", ",", ";", "!", "?", " but ", " and "]:
                        marker_idx = content_lower[idx:].find(marker)
                        if marker_idx != -1 and marker_idx < end_idx - idx:
                            end_idx = idx + marker_idx

                    dislike = content[idx:end_idx].strip()
                    if dislike and len(dislike) > 1:
                        extracted_info["preferences"].append(
                            f"The user dislikes {dislike}"
                        )

        # Characteristics patterns
        characteristic_patterns = [
            ("i am ", ""),
            ("i'm ", ""),
            ("i consider myself ", ""),
            ("i tend to be ", ""),
        ]

        for pattern, prefix in characteristic_patterns:
            if pattern in content_lower:
                idx = content_lower.find(pattern) + len(pattern)
                end_idx = len(content)
                for marker in [".", ",", ";", "!", "?", " but ", " and "]:
                    marker_idx = content_lower[idx:].find(marker)
                    if marker_idx != -1 and marker_idx < end_idx - idx:
                        end_idx = idx + marker_idx

                trait = content[idx:end_idx].strip()
                if (
                    trait
                    and len(trait) > 1
                    and not any(
                        ignore in trait.lower()
                        for ignore in ["years old", "from", "living in"]
                    )
                ):
                    extracted_info["characteristics"].append(
                        f"The user is {prefix}{trait}"
                    )

        # Location/background patterns
        location_patterns = [
            ("i live in ", "lives in"),
            ("i'm from ", "is from"),
            ("i am from ", "is from"),
            ("i grew up in ", "grew up in"),
            ("i was born in ", "was born in"),
        ]

        for pattern, result_format in location_patterns:
            if pattern in content_lower:
                idx = content_lower.find(pattern) + len(pattern)
                end_idx = len(content)
                for marker in [".", ",", ";", "!", "?", " but ", " and "]:
                    marker_idx = content_lower[idx:].find(marker)
                    if marker_idx != -1 and marker_idx < end_idx - idx:
                        end_idx = idx + marker_idx

                location = content[idx:end_idx].strip()
                if location and len(location) > 1:
                    extracted_info["background"].append(
                        f"The user {result_format} {location}"
                    )

        # Age detection
        age_patterns = ["i am ", "i'm ", "age is ", "years old"]
        for pattern in age_patterns:
            if pattern in content_lower and any(
                str(num) in content_lower for num in range(18, 100)
            ):
                # Try to extract the age
                for age in range(18, 100):
                    age_str = str(age)
                    if (
                        f" {age_str} " in content_lower
                        or f"{age_str} " in content_lower
                        or f" {age_str}" in content_lower
                    ):
                        extracted_info["background"].append(
                            f"The user is {age} years old"
                        )
                        break

        # Work/job patterns
        work_patterns = [
            ("i work as ", "works as"),
            ("i am a ", "is a"),
            ("i'm a ", "is a"),
            ("my job is ", "job is"),
            ("my profession is ", "profession is"),
            ("i work in ", "works in"),
            ("my role is ", "role is"),
            ("my career is ", "career is"),
            ("i work for ", "works for"),
        ]

        for pattern, result_format in work_patterns:
            if pattern in content_lower:
                idx = content_lower.find(pattern) + len(pattern)
                end_idx = len(content)
                for marker in [".", ",", ";", "!", "?", " but ", " and "]:
                    marker_idx = content_lower[idx:].find(marker)
                    if marker_idx != -1 and marker_idx < end_idx - idx:
                        end_idx = idx + marker_idx

                job = content[idx:end_idx].strip()
                if job and len(job) > 1:
                    extracted_info["work"].append(f"The user {result_format} {job}")

        # Projects/goals patterns
        project_patterns = [
            ("i'm working on ", "is working on"),
            ("i am working on ", "is working on"),
            ("my project is ", "project is"),
            ("my goal is to ", "goal is to"),
            ("i plan to ", "plans to"),
            ("i'm trying to ", "is trying to"),
            ("i am trying to ", "is trying to"),
        ]

        for pattern, result_format in project_patterns:
            if pattern in content_lower:
                idx = content_lower.find(pattern) + len(pattern)
                end_idx = len(content)
                for marker in [".", ",", ";", "!", "?", " but ", " and "]:
                    marker_idx = content_lower[idx:].find(marker)
                    if marker_idx != -1 and marker_idx < end_idx - idx:
                        end_idx = idx + marker_idx

                project = content[idx:end_idx].strip()
                if project and len(project) > 1:
                    extracted_info["projects"].append(
                        f"The user {result_format} {project}"
                    )

        # Technical knowledge patterns
        tech_patterns = [
            ("i know ", "knows"),
            ("i'm good at ", "is good at"),
            ("i am good at ", "is good at"),
            ("i'm skilled in ", "is skilled in"),
            ("i am skilled in ", "is skilled in"),
            ("i can ", "can"),
            ("i understand ", "understands"),
            ("i've learned ", "has learned"),
            ("i have learned ", "has learned"),
            ("i use ", "uses"),
        ]

        tech_keywords = [
            "programming",
            "coding",
            "python",
            "javascript",
            "java",
            "c++",
            "sql",
            "database",
            "software",
            "development",
            "machine learning",
            "ai",
            "frontend",
            "backend",
            "web",
            "app",
            "mobile",
            "cloud",
            "devops",
            "linux",
            "windows",
            "macos",
            "computer",
            "technology",
            "engineering",
            "design",
            "algorithm",
        ]

        for pattern, result_format in tech_patterns:
            if pattern in content_lower and any(
                keyword in content_lower for keyword in tech_keywords
            ):
                idx = content_lower.find(pattern) + len(pattern)
                end_idx = len(content)
                for marker in [".", ",", ";", "!", "?", " but ", " and "]:
                    marker_idx = content_lower[idx:].find(marker)
                    if marker_idx != -1 and marker_idx < end_idx - idx:
                        end_idx = idx + marker_idx

                skill = content[idx:end_idx].strip()
                if (
                    skill
                    and len(skill) > 1
                    and any(keyword in skill.lower() for keyword in tech_keywords)
                ):
                    extracted_info["technical_knowledge"].append(
                        f"The user {result_format} {skill}"
                    )

        # Only return categories with actual content
        return {k: v for k, v in extracted_info.items() if v}

    def run(self):
        """Main process loop."""
        # Initialize memories in the run method to avoid pickling issues
        self.memories = self._load_memories_from_file()
        OutputManager.print_info("Memory process started")

        while True:
            try:
                # Get next item from queue (non-blocking)
                try:
                    message = self.input_queue.get(block=False)
                except:
                    # No messages in queue, sleep briefly
                    time.sleep(0.1)
                    continue

                # Process the message based on its type
                if message["type"] == "shutdown":
                    # Save memories and exit
                    self._save_memories_to_file()
                    OutputManager.print_info("Memory process shutting down")
                    break

                elif message["type"] == "observation":
                    # Extract information from observation
                    observation = message["data"]
                    extracted_info = self._extract_information(observation)

                    # Store extracted information
                    if extracted_info:
                        timestamp = observation.get(
                            "timestamp", datetime.datetime.now().isoformat()
                        )

                        for category, items in extracted_info.items():
                            for item in items:
                                if item and not any(
                                    m["content"] == item
                                    for m in self.memories.get(category, [])
                                ):
                                    self.memories.setdefault(category, []).append(
                                        {
                                            "content": item,
                                            "timestamp": timestamp,
                                            "source": "conversation",
                                        }
                                    )

                        # Save after significant updates
                        self._save_memories_to_file()

                elif message["type"] == "get_memories":
                    # Return all memories or specific category
                    response_data = {}
                    category = message.get("category")

                    if category in self.memories:
                        response_data = {category: self.memories[category]}
                    else:
                        response_data = self.memories

                    # Convert to JSON string to avoid pickling issues
                    message["response_pipe"].send(json.dumps(response_data))

                elif message["type"] == "query":
                    # Search for relevant memories based on query
                    query = message["query"]
                    results = self._find_relevant_memories(query)

                    # Convert to JSON string to avoid pickling issues
                    message["response_pipe"].send(json.dumps(results))

                elif message["type"] == "add_memory":
                    # Manually add a memory entry
                    category = message["category"]
                    content = message["content"]

                    if category in MEMORY_CATEGORIES and content:
                        timestamp = message.get(
                            "timestamp", datetime.datetime.now().isoformat()
                        )

                        # Add if not a duplicate
                        if not any(
                            m["content"] == content
                            for m in self.memories.get(category, [])
                        ):
                            self.memories.setdefault(category, []).append(
                                {
                                    "content": content,
                                    "timestamp": timestamp,
                                    "source": message.get("source", "manual"),
                                }
                            )

                            # Save after manual additions
                            self._save_memories_to_file()

            except Exception as e:
                OutputManager.print_error(f"Error in memory process: {e}")

    def _find_relevant_memories(self, query: str) -> List[Dict]:
        """Find memories relevant to a query using simple keyword matching."""
        # For now, simple keyword matching
        results = []

        try:
            # For each category of memories
            for category, memories in self.memories.items():
                for memory in memories:
                    # Simple keyword matching for now
                    if any(
                        word.lower() in memory["content"].lower()
                        for word in query.lower().split()
                    ):
                        result = memory.copy()
                        result["category"] = category
                        results.append(result)
        except Exception as e:
            OutputManager.print_error(f"Error finding relevant memories: {e}")

        return results


class MemoryManager:
    """Manager for long-term memory storage and retrieval."""

    def __init__(self):
        """Initialize the memory manager."""
        self.memory_process = None
        self.input_queue = Queue()
        self.memories = {category: [] for category in MEMORY_CATEGORIES}
        from multiprocessing import Pipe

        self.parent_pipe, self.child_pipe = Pipe()

    def initialize_process(self) -> int:
        """Start the memory process and return its PID."""
        if self.memory_process and self.memory_process.is_alive():
            return self.memory_process.pid

        # Create memory directory if it doesn't exist
        MEMORY_DIR.mkdir(exist_ok=True)

        # Start the memory process
        self.memory_process = MemoryProcess(MEMORY_FILE, self.input_queue)
        self.memory_process.start()

        # Load initial memories
        self.load_memories()

        return self.memory_process.pid

    def load_memories(self):
        """Load memories from the memory process."""
        if not self.memory_process or not self.memory_process.is_alive():
            OutputManager.print_warning(
                "Memory process not running, cannot load memories"
            )
            return

        try:
            # Request all memories using pipe instead of queue to avoid pickling issues
            self.input_queue.put(
                {"type": "get_memories", "response_pipe": self.child_pipe}
            )

            # Wait for response (with timeout)
            if self.parent_pipe.poll(5.0):
                memory_json = self.parent_pipe.recv()
                self.memories = json.loads(memory_json)
            else:
                OutputManager.print_error(
                    "Timeout loading memories from memory process"
                )
        except Exception as e:
            OutputManager.print_error(f"Error loading memories: {e}")

    def add_observation(self, observation: Dict):
        """Add an observation to the memory process for information extraction."""
        if not self.memory_process or not self.memory_process.is_alive():
            return

        try:
            self.input_queue.put({"type": "observation", "data": observation})
        except Exception as e:
            OutputManager.print_error(f"Error adding observation: {e}")

    def add_memory(self, category: str, content: str, source: str = "manual"):
        """Manually add a memory entry."""
        if not self.memory_process or not self.memory_process.is_alive():
            return

        if category not in MEMORY_CATEGORIES:
            OutputManager.print_error(f"Invalid memory category: {category}")
            return

        try:
            self.input_queue.put(
                {
                    "type": "add_memory",
                    "category": category,
                    "content": content,
                    "source": source,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )
        except Exception as e:
            OutputManager.print_error(f"Error adding memory: {e}")

    def get_relevant_memories(self, query: str) -> List[Dict]:
        """Get memories relevant to a query."""
        if not self.memory_process or not self.memory_process.is_alive():
            return []

        try:
            # Query for relevant memories using pipe instead of queue
            self.input_queue.put(
                {"type": "query", "query": query, "response_pipe": self.child_pipe}
            )

            # Wait for response (with timeout)
            if self.parent_pipe.poll(3.0):
                results_json = self.parent_pipe.recv()
                return json.loads(results_json)
            else:
                OutputManager.print_error("Timeout getting relevant memories")
                return []
        except Exception as e:
            OutputManager.print_error(f"Error getting relevant memories: {e}")
            return []

    def get_memory_summary(self, category: Optional[str] = None) -> str:
        """Get a summary of stored memories."""
        # Refresh memories from the process
        self.load_memories()

        if category:
            if category not in self.memories:
                return f"No memories found for category: {category}"

            category_data = self.memories[category]
            if not category_data:
                return f"No memories stored in category: {category}"

            # Format memories for the specific category
            memory_items = [f"- {item['content']}" for item in category_data]
            return f"Memory for {category} ({len(memory_items)} items):\n" + "\n".join(
                memory_items
            )
        else:
            # Generate overall summary
            summary = []
            total_memories = 0

            for category, items in self.memories.items():
                count = len(items)
                total_memories += count
                if count > 0:
                    summary.append(f"{category}: {count} items")

            if total_memories == 0:
                return "No memories stored yet."

            return f"Memory Summary ({total_memories} total items):\n" + "\n".join(
                summary
            )

    def get_memory_for_prompt(self, query: str) -> str:
        """Generate memory context for a prompt based on query."""
        relevant_memories = self.get_relevant_memories(query)

        if not relevant_memories:
            return ""

        # Format relevant memories
        memory_sections = {}

        for memory in relevant_memories:
            category = memory.get("category", "unknown")
            if category not in memory_sections:
                memory_sections[category] = []
            memory_sections[category].append(memory["content"])

        # Construct prompt context
        prompt_parts = ["## User Memory Context"]

        for category, items in memory_sections.items():
            prompt_parts.append(f"\n### {category.title()}")
            for item in items:
                prompt_parts.append(f"- {item}")

        return "\n".join(prompt_parts)

    def shutdown(self):
        """Shut down the memory process."""
        if self.memory_process and self.memory_process.is_alive():
            try:
                self.input_queue.put({"type": "shutdown"})
                # Give it a moment to save
                self.memory_process.join(timeout=3)

                if self.memory_process.is_alive():
                    self.memory_process.terminate()
                    OutputManager.print_warning("Memory process terminated forcefully")
            except Exception as e:
                OutputManager.print_error(f"Error shutting down memory process: {e}")
