#!/usr/bin/env python3

"""
AI Office v2 - Query Router
==========================

This module defines the QueryRouter class, which analyzes user queries and determines
which agent would be best suited to handle them.
"""

import json
import requests
import time
import re
from typing import Dict, Any, List, Optional, Union
from output_manager import OutputManager
from agent_registry import get_registry

# Router configuration
ROUTER_MODEL = "llama3.2:latest"  # Using a capable model for accurate routing
API_URL = "http://localhost:11434/api/generate"

# Map internal agent names to actual agent names
AGENT_NAME_MAP = {
    "coder": "technical_specialist",
    "writer": "creative_assistant",  # This may not exist yet
    "researcher": "research_assistant",
    "planner": "calendar_manager",
}


class QueryRouter:
    """Routes user queries to appropriate specialist agents based on query content."""

    def __init__(self, speed_mode="fast"):
        """Initialize the query router.

        Args:
            speed_mode (str): The routing speed mode: "fastest", "fast", or "accurate"
        """
        self.speed_mode = speed_mode
        self.registry = get_registry()

        # Get actual available agents
        self.available_agents = (
            list(self.registry.agent_processes.keys()) if self.registry else []
        )

        # System prompt for the router
        self._create_system_prompt()

        # Keywords for fastest mode
        self._init_keywords()

    def _create_system_prompt(self):
        """Create the appropriate system prompt based on the current configuration."""
        # Build list of available agents with descriptions
        agent_descriptions = []

        # Check if we have actual agents running
        running_agents = self.available_agents

        if running_agents:
            for agent_name in running_agents:
                # Get a description for this agent
                description = "Unknown"
                if agent_name == "technical_specialist":
                    description = "Programming, coding, debugging, software development, technical questions"
                elif agent_name == "research_assistant":
                    description = "Research, information retrieval, current events, weather, news, facts"
                elif agent_name == "calendar_manager":
                    description = (
                        "Scheduling, appointments, time management, dates, reminders"
                    )
                elif agent_name == "creative_assistant":
                    description = "Writing, editing, summarizing text, creative writing"

                agent_descriptions.append(f'- "{agent_name}": {description}')
        else:
            # Use generic descriptions if no agents are running
            agent_descriptions = [
                f'- "coder": Programming, software development, code review, debugging',
                f'- "writer": Writing, editing, summarizing text, creative writing',
                f'- "researcher": Finding information, data analysis, summarizing research',
                f'- "planner": Task planning, project management, scheduling, organization',
            ]

        # Join agent descriptions with newlines
        agent_desc_text = "\n".join(agent_descriptions)

        # Base system prompt with agent descriptions
        self.system_prompt = f"""You are a query routing system for an AI Office with specialist agents.
Your job is to analyze user queries and determine which agent would be best suited to respond.

Available specialist agents:
{agent_desc_text}

Executive Assistant (EA):
The EA is the default handler for most user queries, particularly those that are general,
conversational, or do not clearly require specialized expertise. The EA has personal memory
and knowledge about the user, and should handle any query that doesn't STRONGLY require
specialized knowledge.

For each query, determine:
1. Which single agent is best suited to handle this query, or if the Executive Assistant should handle it
2. A confidence score (0.0-1.0) in your routing decision
3. Brief reasoning for your choice

Respond with a JSON object containing:
{{"agent": "agent_name", "confidence": 0.8, "reasoning": "Brief explanation"}}

Use ONLY the agent names listed above or "none" to indicate the Executive Assistant should handle it.

IMPORTANT GUIDELINES:
- The EA (agent: "none") should be the DEFAULT choice for most queries
- Only route to a specialist agent if the query ABSOLUTELY REQUIRES their specific expertise
- Specialist agents should ONLY handle queries that CANNOT be reasonably handled by the EA
- General questions, personal topics, small talk, and anything not clearly requiring specialized expertise should ALWAYS be handled by the EA (agent: "none")
- Require at least 0.85 confidence to route to a specialist agent, otherwise default to the EA
- "What can you do" style questions should ALWAYS be handled by the EA (agent: "none")
- Questions about capabilities, features, or how things work should ALWAYS be handled by the EA
- Greetings, farewells, thank you messages, and short responses should ALWAYS be handled by the EA
- When in doubt, ALWAYS choose the EA (agent: "none")"""

    def _init_keywords(self):
        """Initialize keywords for fastest routing mode."""
        self.keywords = {
            "technical_specialist": [
                "code",
                "program",
                "programming",
                "develop",
                "function",
                "bug",
                "debugging",
                "software",
                "algorithm",
                "github",
                "repository",
                "git",
                "python",
                "javascript",
                "typescript",
                "java",
                "c++",
                "rust",
                "html",
                "css",
                "api",
                "backend",
                "frontend",
                "database",
                "sql",
                "error",
                "framework",
                "library",
                "class",
                "object",
            ],
            "research_assistant": [
                "research",
                "information",
                "data",
                "analysis",
                "find",
                "search",
                "summary",
                "statistics",
                "facts",
                "evidence",
                "study",
                "report",
                "topic",
                "question",
                "investigate",
                "explore",
                "examine",
                "understand",
                "learn",
                "knowledge",
                "weather",
                "news",
                "current events",
                "history",
                "science",
                "technology",
            ],
            "calendar_manager": [
                "plan",
                "schedule",
                "organize",
                "task",
                "project",
                "goal",
                "productivity",
                "time",
                "management",
                "calendar",
                "deadline",
                "reminder",
                "priority",
                "workflow",
                "efficiency",
                "habit",
                "strategy",
                "objective",
                "track",
                "meeting",
                "appointment",
                "event",
                "date",
            ],
        }

        # Add backward compatibility for generic names
        self.keywords["coder"] = self.keywords["technical_specialist"]
        self.keywords["researcher"] = self.keywords["research_assistant"]
        self.keywords["planner"] = self.keywords["calendar_manager"]
        self.keywords["writer"] = [
            "write",
            "writing",
            "essay",
            "article",
            "blog",
            "content",
            "story",
            "edit",
            "grammar",
            "text",
            "paragraph",
            "summarize",
            "creative",
            "proofread",
            "poem",
            "novel",
            "screenplay",
            "email",
            "letter",
            "document",
            "copywriting",
            "review",
        ]

    def keyword_routing(self, query: str) -> Optional[Dict[str, Any]]:
        """Perform fast keyword-based routing without using an LLM.

        Args:
            query (str): The user query

        Returns:
            Optional[Dict]: Routing result or None if no clear match
        """
        query = query.lower()

        # Count keyword matches for each agent
        matches = {agent: 0 for agent in self.keywords}
        for agent, agent_keywords in self.keywords.items():
            for keyword in agent_keywords:
                if keyword in query:
                    matches[agent] += 1

        # Find the agent with the most matches
        max_matches = max(matches.values())

        # Require more keyword matches for delegation (increased threshold)
        if max_matches > 2:  # Requiring at least 3 matches now (was effectively 1)
            # Get all agents with the maximum number of matches
            top_agents = [
                agent for agent, count in matches.items() if count == max_matches
            ]

            # Filter to only include available agents if possible
            available_top_agents = [
                agent for agent in top_agents if agent in self.available_agents
            ]

            if available_top_agents:
                top_agents = available_top_agents

            # If there's a clear winner
            if len(top_agents) == 1:
                chosen_agent = top_agents[0]
                # Calculate confidence based on match count (higher matches = higher confidence)
                # Start with a higher base confidence since we already required more matches
                confidence = min(0.75 + (0.05 * max_matches), 0.95)

                # Map generic agent name to actual agent name if needed
                if (
                    chosen_agent in AGENT_NAME_MAP
                    and AGENT_NAME_MAP[chosen_agent] in self.available_agents
                ):
                    chosen_agent = AGENT_NAME_MAP[chosen_agent]

                # Only return if the agent is actually available
                if chosen_agent in self.available_agents:
                    return {
                        "agent": chosen_agent,
                        "confidence": round(confidence, 2),
                        "reasoning": f"Matched {max_matches} keywords related to {chosen_agent}",
                    }

        # No clear keyword match or no available agents, or not enough matches
        return None

    def map_agent_name(self, agent_name: str) -> str:
        """Map generic agent name to actual agent name if needed."""
        # If the agent name is already available, use it
        if agent_name in self.available_agents:
            return agent_name

        # Try to map the name
        if agent_name in AGENT_NAME_MAP:
            mapped_name = AGENT_NAME_MAP[agent_name]
            if mapped_name in self.available_agents:
                return mapped_name

        # If we can't map it or the mapped name isn't available,
        # return the first available agent with lower confidence
        if self.available_agents:
            return self.available_agents[0]

        # If no agents are available, return the original name
        return agent_name

    def route_query(
        self,
        user_prompt: str,
        memory_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_preference: Optional[str] = None,
        force_json: bool = False,
    ) -> Dict[str, Any]:
        """Route a user query to the appropriate agent.

        Args:
            user_prompt (str): The user's query
            memory_context (str, optional): Memory context to include in the prompt
            system_prompt (str, optional): Override the default system prompt
            model_preference (str, optional): Override the speed mode for this query
            force_json (bool): Force the response to be valid JSON

        Returns:
            Dict: The routing decision with agent name and confidence
        """
        # Update available agents before routing
        self.available_agents = (
            list(self.registry.agent_processes.keys()) if self.registry else []
        )

        # If no agents are available, return a default response
        if not self.available_agents:
            return {
                "agent": "none",
                "confidence": 0.0,
                "reasoning": "No agents are currently available",
            }

        # First, check if we should use fastest mode (keyword-based routing)
        if self.speed_mode == "fastest" and not system_prompt:
            keyword_result = self.keyword_routing(user_prompt)
            if keyword_result:
                return keyword_result

        # If keyword routing didn't produce a result or wasn't used, fall back to LLM
        # Determine which model to use based on mode or override
        mode = model_preference or self.speed_mode

        # Select the appropriate model based on the speed/accuracy mode
        model = ROUTER_MODEL
        if mode == "fastest":
            model = "tinyllama"  # Smallest, fastest model
        elif mode == "fast":
            model = "llama3.1:latest"  # Good balance of speed and accuracy

        # Use system prompt override if provided, otherwise use default
        prompt_to_use = system_prompt or self.system_prompt

        # Prepare the user prompt with memory context if available
        enhanced_prompt = user_prompt
        if memory_context:
            enhanced_prompt = f"{user_prompt}\n\n{memory_context}"

        # Set up retry parameters
        max_retries = 2 if force_json else 1
        retry_count = 0
        timeout = 10.0  # seconds

        while retry_count <= max_retries:
            try:
                # Log what we're doing
                (
                    OutputManager.print_info(
                        f"Routing query using {model} in {mode} mode"
                    )
                    if mode == "accurate"
                    else None
                )

                # Create payload for the LLM API
                payload = {
                    "model": model,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for more deterministic routing
                        "max_tokens": 300,
                    },
                    "system": prompt_to_use,
                }

                # Call the LLM API
                response = requests.post(API_URL, json=payload, timeout=timeout)
                response.raise_for_status()
                response_data = response.json()

                if "response" in response_data:
                    response_text = response_data["response"].strip()

                    # Extract the JSON part from the response
                    try:
                        # First try to parse the entire response as JSON
                        result = json.loads(response_text)

                        # If we have an agent field, check and map the agent name
                        if "agent" in result:
                            result["agent"] = self.map_agent_name(result["agent"])

                        return result
                    except json.JSONDecodeError:
                        # If that fails, try to extract JSON from the text
                        json_match = re.search(r"({[\s\S]*})", response_text)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(1))

                                # If we have an agent field, check and map the agent name
                                if "agent" in result:
                                    result["agent"] = self.map_agent_name(
                                        result["agent"]
                                    )

                                return result
                            except json.JSONDecodeError:
                                if retry_count < max_retries:
                                    retry_count += 1
                                    continue

                                # Return a fallback result on final failure
                                if force_json:
                                    return {"error": "Failed to parse JSON response"}
                                else:
                                    # Pick the first available agent with low confidence
                                    return {
                                        "agent": self.available_agents[0],
                                        "confidence": 0.3,
                                        "reasoning": "Fallback due to parsing error",
                                    }

                # If we can't parse JSON, return the raw response
                return {"content": response_text}

            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    # Return a fallback result on failure
                    if force_json:
                        return {"error": f"Error calling LLM API: {str(e)}"}
                    else:
                        # Pick the first available agent with low confidence
                        return {
                            "agent": (
                                self.available_agents[0]
                                if self.available_agents
                                else "none"
                            ),
                            "confidence": 0.3,
                            "reasoning": f"Error: {str(e)}",
                        }

        # Default fallback (should rarely get here)
        return {
            "agent": self.available_agents[0] if self.available_agents else "none",
            "confidence": 0.3,
            "reasoning": "Default fallback when all else fails",
        }

    # Create a special evaluation prompt for checking if a query is still relevant for current agent
    def create_agent_relevance_prompt(self, agent_name, query):
        """Create a specialized prompt for evaluating if a query is still relevant for an agent.

        Args:
            agent_name (str): The current agent name
            query (str): The user query

        Returns:
            str: The specialized system prompt
        """
        return f"""You are evaluating if a user query is still relevant for the currently active agent.

Current agent: {agent_name}

For this evaluation, you should be VERY strict. Only determine that the query is still relevant
for the current agent if the query CLEARLY requires this agent's specific expertise.

If the query is general, about capabilities, about how things work, or can be handled by a general assistant,
it should be handled by the Executive Assistant instead.

Evaluation guidelines:
1. Is this query CLEARLY within the {agent_name}'s specialized expertise domain?
2. Does this query REQUIRE specialized knowledge that only the {agent_name} would have?
3. Would routing this to the Executive Assistant (EA) result in a worse user experience?

Unless ALL THREE conditions are met with high confidence, the Executive Assistant should handle it.

"What can you do" style questions should ALWAYS be handled by the EA.
Questions about capabilities, features, or how things work should be handled by the EA.

Respond with a JSON object:
{{"agent": "{agent_name}", "confidence": 0.9, "reasoning": "Explanation"}}

OR

{{"agent": "none", "confidence": 0.8, "reasoning": "Explanation"}}

Only use one of these two options."""
