#!/usr/bin/env python3

"""
AI Office v2 - Query Router
=========================

This module routes user queries to the most appropriate agent.
"""

from typing import Dict, List, Optional, Any
from logger import Logger
from agent_registry import AgentRegistry
from agent_config import AgentConfig


class QueryRouter:
    """Routes queries to the most appropriate agent."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.agents: Dict[str, AgentConfig] = {}
        self._load_agents()

    def _load_agents(self):
        """Load agent configurations from the registry."""
        if not self.registry:
            Logger.print_warning("Agent registry not available")
            return

        # Get agent configurations from registry
        for agent_name, agent_data in self.registry.agents.items():
            try:
                # If agent_data is already an AgentConfig, use it directly
                if isinstance(agent_data, AgentConfig):
                    self.agents[agent_name] = agent_data
                else:
                    # Otherwise create a new AgentConfig from the data
                    agent = AgentConfig(**agent_data)
                    self.agents[agent.name] = agent
                Logger.print_info(
                    f"Loaded agent: {self.agents[agent_name].display_name}"
                )
            except Exception as e:
                Logger.print_error(f"Error loading agent '{agent_name}': {e}")

    def route_query(self, query: str) -> Optional[str]:
        """Route a query to the most appropriate agent."""
        if not self.agents:
            Logger.print_warning("No agents available for routing")
            return None

        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()

        # Find matching agents based on keywords
        matching_agents = []
        for agent_name, agent in self.agents.items():
            if agent.status != "idle":
                continue

            # Check if any keywords match
            if any(keyword in query_lower for keyword in agent.keywords):
                # Calculate match score based on number of matching keywords
                match_score = sum(
                    1 for keyword in agent.keywords if keyword in query_lower
                )
                matching_agents.append((agent_name, agent, match_score))

        if not matching_agents:
            Logger.print_warning(f"No matching agent found for query: {query}")
            return None

        # Sort by match score (higher score first)
        matching_agents.sort(key=lambda x: x[2], reverse=True)

        # Return the agent with the highest match score
        selected_agent = matching_agents[0][0]
        Logger.print_info(
            f"Routed query to agent: {self.agents[selected_agent].display_name}"
        )
        return selected_agent

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all available agent names."""
        return list(self.agents.keys())
