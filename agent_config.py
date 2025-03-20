#!/usr/bin/env python3

from dataclasses import dataclass


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
