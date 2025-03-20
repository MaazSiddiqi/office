#!/usr/bin/env python3

"""
AI Office v2 - Agent Configuration
===============================

This module defines the shared configuration structure for AI agents.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    display_name: str
    description: str
    system_prompt: str
    model: str
    temperature: float
    status: str
    keywords: List[str] = None

    def __post_init__(self):
        """Initialize default values and generate keywords if needed."""
        if not hasattr(self, "status"):
            self.status = "idle"
        if self.keywords is None:
            # Generate keywords based on description and name
            self.keywords = []
            # Add name-based keywords
            self.keywords.extend(self.name.lower().split("_"))
            # Add description-based keywords
            self.keywords.extend(self.description.lower().split())
