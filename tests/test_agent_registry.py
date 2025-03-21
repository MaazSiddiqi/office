#!/usr/bin/env python3

import os
import json
import pytest
from agent_registry import AgentRegistry
from agent_config import AgentConfig


@pytest.fixture
def test_registry_dir(tmp_path):
    """Create a temporary registry directory with test agent configs."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()

    # Create test agent configs
    test_agents = [
        {
            "name": "test_agent1",
            "display_name": "Test Agent 1",
            "description": "A test agent",
            "system_prompt": "You are a test agent",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "status": "active",
        },
        {
            "name": "test_agent2",
            "display_name": "Test Agent 2",
            "description": "Another test agent",
            "system_prompt": "You are another test agent",
            "model": "gpt-4",
            "temperature": 0.8,
            "status": "inactive",
        },
    ]

    for agent in test_agents:
        with open(registry_dir / f"{agent['name']}.json", "w") as f:
            json.dump(agent, f)

    return registry_dir


def test_agent_registry_initialization(test_registry_dir):
    """Test that AgentRegistry initializes correctly with a registry directory."""
    registry = AgentRegistry(str(test_registry_dir))
    assert len(registry.agents) == 2
    assert "test_agent1" in registry.agents
    assert "test_agent2" in registry.agents


def test_get_agent(test_registry_dir):
    """Test retrieving an agent by name."""
    registry = AgentRegistry(str(test_registry_dir))
    agent = registry.get_agent("test_agent1")
    assert agent is not None
    assert agent.name == "test_agent1"
    assert agent.display_name == "Test Agent 1"
    assert agent.model == "gpt-3.5-turbo"


def test_get_nonexistent_agent(test_registry_dir):
    """Test retrieving a nonexistent agent returns None."""
    registry = AgentRegistry(str(test_registry_dir))
    agent = registry.get_agent("nonexistent_agent")
    assert agent is None


def test_list_agents(test_registry_dir):
    """Test listing all agents."""
    registry = AgentRegistry(str(test_registry_dir))
    agents = registry.list_agents()
    assert len(agents) == 2
    assert all(isinstance(agent, AgentConfig) for agent in agents.values())
    assert all(
        agent.name in ["test_agent1", "test_agent2"] for agent in agents.values()
    )


def test_empty_registry(tmp_path):
    """Test AgentRegistry with an empty registry directory."""
    registry = AgentRegistry(str(tmp_path))
    assert len(registry.agents) == 0
    assert registry.get_agent("any_agent") is None
    assert len(registry.list_agents()) == 0
