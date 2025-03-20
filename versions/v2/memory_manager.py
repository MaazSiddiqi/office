#!/usr/bin/env python3

"""
AI Office v2 - Memory Manager
============================

This module manages long-term memory for the AI Office system.
"""

import os
import json
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from logger import Logger

# Configuration
MEMORY_DIR = "memory"
BACKUP_DIR = "memory/backups"
MAX_MEMORIES = 1000
MEMORY_EXPIRY_DAYS = 30


@dataclass
class Memory:
    """Represents a single memory entry."""

    id: str
    content: str
    timestamp: float
    tags: List[str]
    importance: float
    context: Dict[str, Any]


class MemoryManager:
    """Manages the system's long-term memory."""

    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self._setup_directories()
        self._load_memories()

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        Path(MEMORY_DIR).mkdir(exist_ok=True)
        Path(BACKUP_DIR).mkdir(exist_ok=True)

    def _load_memories(self):
        """Load memories from disk."""
        try:
            memory_file = Path(MEMORY_DIR) / "memories.json"
            if memory_file.exists():
                with open(memory_file, "r") as f:
                    data = json.load(f)
                    for mem_data in data:
                        memory = Memory(**mem_data)
                        self.memories[memory.id] = memory
                Logger.print_info(f"Loaded {len(self.memories)} memories")
        except Exception as e:
            Logger.print_error(f"Error loading memories: {e}")

    def _save_memories(self):
        """Save memories to disk."""
        try:
            memory_file = Path(MEMORY_DIR) / "memories.json"
            with open(memory_file, "w") as f:
                json.dump([vars(mem) for mem in self.memories.values()], f)
        except Exception as e:
            Logger.print_error(f"Error saving memories: {e}")

    def add_memory(
        self,
        content: str,
        tags: List[str] = None,
        importance: float = 1.0,
        context: Dict[str, Any] = None,
    ) -> str:
        """Add a new memory."""
        memory_id = f"mem_{int(time.time())}_{len(self.memories)}"
        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=time.time(),
            tags=tags or [],
            importance=importance,
            context=context or {},
        )
        self.memories[memory_id] = memory
        self._save_memories()
        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        return self.memories.get(memory_id)

    def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content and tags."""
        results = []
        for memory in self.memories.values():
            if query.lower() in memory.content.lower():
                results.append(memory)
            elif any(query.lower() in tag.lower() for tag in memory.tags):
                results.append(memory)

        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:limit]

    def update_memory(self, memory_id: str, **kwargs) -> bool:
        """Update an existing memory."""
        if memory_id not in self.memories:
            return False

        memory = self.memories[memory_id]
        for key, value in kwargs.items():
            if hasattr(memory, key):
                setattr(memory, key, value)

        self._save_memories()
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            self._save_memories()
            return True
        return False

    def cleanup_old_memories(self):
        """Remove memories older than MEMORY_EXPIRY_DAYS."""
        current_time = time.time()
        old_memories = [
            mem_id
            for mem_id, mem in self.memories.items()
            if (current_time - mem.timestamp) > (MEMORY_EXPIRY_DAYS * 24 * 60 * 60)
        ]

        for mem_id in old_memories:
            del self.memories[mem_id]

        if old_memories:
            self._save_memories()
            Logger.print_info(f"Cleaned up {len(old_memories)} old memories")

    def backup_memories(self):
        """Create a backup of all memories."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_file = Path(BACKUP_DIR) / f"memories_{timestamp}.json"

            with open(backup_file, "w") as f:
                json.dump([vars(mem) for mem in self.memories.values()], f)

            Logger.print_success(f"Created memory backup: {backup_file}")
        except Exception as e:
            Logger.print_error(f"Error creating memory backup: {e}")


# Singleton instance
_memory_manager_instance = None


def get_memory_manager() -> MemoryManager:
    """Get the singleton memory manager instance."""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance
