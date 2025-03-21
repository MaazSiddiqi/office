#!/usr/bin/env python3

from dataclasses import dataclass


class AgentStatus:
    """Status of an agent."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    display_name: str
    description: str
    system_prompt: str
    model: str
    temperature: float
    status: AgentStatus
