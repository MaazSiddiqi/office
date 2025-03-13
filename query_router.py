#!/usr/bin/env python3

"""
AI Office v2 - Query Router
==========================

This module defines the QueryRouter class, which analyzes user queries and determines
if they should be delegated to a specialized agent and which agent to use.
"""

import json
import requests
import time
from output_manager import SUBTLE_COLOR, RESET

# Configuration
ROUTER_MODEL = "llama3.2:latest"  # Using more capable but still fast model for routing
API_URL = "http://localhost:11434/api/generate"


class QueryRouter:
    """
    Analyzes user queries and determines if they should be delegated to a specialized agent.

    This class uses a small, fast LLM to:
    1. Analyze user input
    2. Determine if the query should be handled by the EA or delegated
    3. Select the appropriate agent for delegation if needed
    """

    def __init__(self, registry, enabled=True, verbose=False, speed_mode="fast"):
        """
        Initialize the Query Router.

        Args:
            registry: The agent registry containing available agents
            enabled (bool): Whether the router is enabled (default: True)
            verbose (bool): Whether to include verbose debugging in prompts
            speed_mode (str): Routing speed/accuracy tradeoff: "fastest", "fast", or "accurate"
        """
        self.registry = registry
        self.verbose = verbose
        self.speed_mode = (
            speed_mode  # New parameter for routing speed/accuracy tradeoff
        )
        self.system_prompt = self._build_system_prompt()
        self.debug_log = []
        self.enabled = enabled

        # Keywords for fastest mode
        self.technical_keywords = [
            "code",
            "program",
            "debug",
            "python",
            "javascript",
            "coding",
            "java",
            "c++",
            "ruby",
            "algorithm",
            "function",
            "compile",
        ]
        self.research_keywords = [
            "information",
            "weather",
            "news",
            "research",
            "find out",
            "what is",
            "when did",
            "who is",
            "where is",
            "history",
            "data",
            "statistics",
        ]
        self.calendar_keywords = [
            "schedule",
            "appointment",
            "calendar",
            "meeting",
            "remind",
            "event",
            "date",
            "time",
            "planning",
            "reservation",
        ]

    def _build_system_prompt(self):
        """
        Build the system prompt for the routing LLM based on available agents.

        Returns:
            str: The system prompt for the routing LLM
        """
        # Get available agents
        available_agents = self.registry.list_available_agents()
        running_agents = list(self.registry.agent_processes.keys())

        # Build agent descriptions - shorter and more focused
        agent_descriptions = []
        for agent_name in running_agents:
            agent_config = self.registry.get_agent_info(agent_name)
            if agent_config and "description" in agent_config:
                description = agent_config["description"]
            else:
                # Use default descriptions if not specified in config
                if agent_name == "research_assistant":
                    description = "Research, information retrieval, current events, weather, news, facts"
                elif agent_name == "calendar_manager":
                    description = (
                        "Scheduling, appointments, time management, dates, reminders"
                    )
                elif agent_name == "technical_specialist":
                    description = "Programming, coding, debugging, software development, technical questions"
                elif agent_name == "example_agent":
                    description = "Demo agent"
                else:
                    description = f"{agent_name.replace('_', ' ')} tasks"

            agent_descriptions.append(f"- {agent_name}: {description}")

        # Create a more direct prompt focused on correct classification
        system_prompt = f"""You are a query classifier that determines if a user query should be handled by a specialized agent.

AVAILABLE AGENTS:
{chr(10).join(agent_descriptions)}

YOUR TASK:
Analyze the query and decide whether the EA should handle it directly or delegate it to a specialized agent.

MUST RESPOND AS VALID JSON with this structure:
{{
    "delegate": true/false,
    "agent": "agent_name_here",  // Include ONLY if delegate is true
    "confidence": 0.0-1.0,  // How confident you are
    "explanation": "Brief explanation"
}}

DELEGATION RULES:
- Delegate if the query clearly matches an agent's expertise
- Research questions go to research_assistant
- Technical and programming questions go to technical_specialist
- Scheduling and calendar questions go to calendar_manager
- DON'T delegate general conversation or small talk
- DON'T delegate system questions or questions about the AI office itself
- ONLY delegate to the available agents listed above

EXAMPLES:
User: "What's the weather in New York today?"
{{
    "delegate": true,
    "agent": "research_assistant",
    "confidence": 0.9,
    "explanation": "Weather information requires external research"
}}

User: "How do I debug Python code?"
{{
    "delegate": true,
    "agent": "technical_specialist",
    "confidence": 0.95,
    "explanation": "Programming and debugging questions should go to technical specialist"
}}

User: "Tell me about yourself"
{{
    "delegate": false,
    "confidence": 0.9,
    "explanation": "General question about the system should be handled by EA"
}}

IMPORTANT: Output ONLY valid JSON. No markdown, no explanation, just the JSON object."""

        # Add more verbose instructions if needed
        if self.verbose:
            system_prompt += """

ADDITIONAL GUIDANCE:
- Be precise in your analysis of the query
- Consider both explicit and implicit requirements in the query
- Technical questions include programming languages, debugging, coding, etc.
- Research questions include facts, news, events, data, etc.
- Calendar queries include scheduling, time management, dates, etc.
- Higher confidence (>0.8) should only be given when the match is very clear
- Always return properly formatted JSON with all required fields"""

        return system_prompt

    def keyword_routing(self, user_query):
        """
        Perform simple keyword-based routing for extremely fast decision making.

        Args:
            user_query (str): The user's query

        Returns:
            dict or None: Routing decision if a clear match is found, None otherwise
        """
        query_lower = user_query.lower()

        # Check for technical keywords
        for keyword in self.technical_keywords:
            if keyword in query_lower:
                if "technical_specialist" in self.registry.agent_processes:
                    self.debug_log.append(
                        "Fast-tracked to technical_specialist based on keywords"
                    )
                    return {
                        "delegate": True,
                        "agent": "technical_specialist",
                        "confidence": 0.85,
                        "explanation": f"Query contains technical keyword: '{keyword}'",
                    }

        # Check for research keywords
        for keyword in self.research_keywords:
            if keyword in query_lower:
                if "research_assistant" in self.registry.agent_processes:
                    self.debug_log.append(
                        "Fast-tracked to research_assistant based on keywords"
                    )
                    return {
                        "delegate": True,
                        "agent": "research_assistant",
                        "confidence": 0.85,
                        "explanation": f"Query contains research keyword: '{keyword}'",
                    }

        # Check for calendar keywords
        for keyword in self.calendar_keywords:
            if keyword in query_lower:
                if "calendar_manager" in self.registry.agent_processes:
                    self.debug_log.append(
                        "Fast-tracked to calendar_manager based on keywords"
                    )
                    return {
                        "delegate": True,
                        "agent": "calendar_manager",
                        "confidence": 0.85,
                        "explanation": f"Query contains calendar keyword: '{keyword}'",
                    }

        # No clear keyword match found
        return None

    def route_query(
        self, user_query, conversation_history=None, cancel_check_callback=None
    ):
        """
        Analyze a user query and determine if it should be delegated to a specialized agent.

        Args:
            user_query (str): The user's query
            conversation_history (list, optional): Recent conversation history for context
            cancel_check_callback (callable, optional): Function to check if request should be cancelled

        Returns:
            dict: A dictionary with routing information:
                - delegate (bool): Whether to delegate this query
                - agent (str): Which agent to delegate to (if delegate=True)
                - confidence (float): Confidence score (0.0-1.0)
                - explanation (str): Explanation of the routing decision
        """
        # Clear previous debug log
        self.debug_log = []

        # Skip analysis if router is disabled
        if not self.enabled:
            return {
                "delegate": False,
                "confidence": 0.0,
                "explanation": "Router is disabled",
            }

        # Default result if API fails
        default_result = {
            "delegate": False,
            "confidence": 0.0,
            "explanation": "Failed to analyze query",
        }

        # Check for cancellation before starting
        if cancel_check_callback and cancel_check_callback():
            return {
                "delegate": False,
                "confidence": 0.0,
                "explanation": "Query routing cancelled",
            }

        # If in fastest mode, try keyword routing first
        if self.speed_mode == "fastest":
            keyword_result = self.keyword_routing(user_query)
            if keyword_result:
                return keyword_result

        # Configure retry parameters - adjust based on speed mode
        max_retries = 1 if self.speed_mode == "fast" else 2
        timeout = 5 if self.speed_mode == "fast" else 8  # Shorter timeout for fast mode
        retry_delay = 0.3 if self.speed_mode == "fast" else 0.5

        for attempt in range(max_retries + 1):
            try:
                # Check for cancellation
                if cancel_check_callback and cancel_check_callback():
                    return {
                        "delegate": False,
                        "confidence": 0.0,
                        "explanation": "Query routing cancelled",
                    }

                # Prepare minimal context from conversation history if available
                context = ""
                if conversation_history and len(conversation_history) > 0:
                    # Include only the last exchange for minimal context
                    last_entry = conversation_history[-1]
                    speaker = (
                        "User" if last_entry.get("role") == "user" else "Assistant"
                    )
                    context = f"Last message: {speaker}: {last_entry.get('content', '')[:100]}...\n\n"

                # Create the prompt for the LLM - keep it short
                prompt = f"{context}User: {user_query}\n\n"

                # Create payload for the LLM API - optimize for accuracy or speed based on mode
                payload = {
                    "model": ROUTER_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "max_tokens": 200,
                        "num_ctx": 512,  # Small context window
                        "num_thread": 4,  # More threads for faster inference
                    },
                    "system": self.system_prompt,
                }

                # Log the attempt
                self.debug_log.append(
                    f"Router API call attempt {attempt+1}/{max_retries+1}"
                )

                # Call the LLM API
                start_time = time.time()
                response = None
                with requests.Session() as session:
                    response = session.post(API_URL, json=payload, timeout=timeout)

                if response:
                    response.raise_for_status()
                    response_data = response.json()
                    elapsed_time = time.time() - start_time

                    # Log the routing time
                    self.debug_log.append(
                        f"Router LLM call completed in {elapsed_time:.2f}s"
                    )

                    # Extract and parse the JSON response
                    if "response" in response_data:
                        try:
                            # Clean the response to ensure it's valid JSON
                            json_str = response_data["response"].strip()

                            # Save the raw response for debugging
                            self.debug_log.append(f"Raw response: {json_str[:100]}...")

                            # Remove any markdown code block markers
                            if json_str.startswith("```json"):
                                json_str = json_str[7:]
                            if json_str.startswith("```"):
                                json_str = json_str[3:]
                            if json_str.endswith("```"):
                                json_str = json_str[:-3]

                            # Find the JSON object if there's extra text
                            start_idx = json_str.find("{")
                            end_idx = json_str.rfind("}")

                            if start_idx >= 0 and end_idx >= 0:
                                json_str = json_str[start_idx : end_idx + 1]

                            json_str = json_str.strip()

                            try:
                                result = json.loads(json_str)
                            except json.JSONDecodeError as e:
                                self.debug_log.append(f"JSON decode error: {e}")

                                # Try to fix common JSON format issues
                                json_str = json_str.replace(
                                    "'", '"'
                                )  # Replace single quotes with double quotes
                                json_str = json_str.replace(
                                    "True", "true"
                                )  # Fix Python-style booleans
                                json_str = json_str.replace("False", "false")

                                # Try again with the fixed JSON
                                result = json.loads(json_str)

                            # Validate the result structure
                            if "delegate" not in result:
                                raise ValueError("Missing 'delegate' field in response")

                            if result.get("delegate") and "agent" not in result:
                                raise ValueError("Missing 'agent' field for delegation")

                            # Check if the specified agent is available
                            if result.get("delegate"):
                                agent_name = result.get("agent")
                                if agent_name not in self.registry.agent_processes:
                                    # Modify the result if the agent isn't running
                                    result["delegate"] = False
                                    result["explanation"] = (
                                        f"Agent '{agent_name}' is not currently running"
                                    )

                            # Ensure confidence is a float between 0 and 1
                            if "confidence" in result:
                                try:
                                    conf = float(result["confidence"])
                                    result["confidence"] = max(0.0, min(1.0, conf))
                                except (ValueError, TypeError):
                                    result["confidence"] = 0.5

                            return result
                        except (json.JSONDecodeError, ValueError) as e:
                            self.debug_log.append(f"Error parsing router response: {e}")
                            self.debug_log.append(
                                f"Raw response: {response_data['response']}"
                            )
                    else:
                        self.debug_log.append(
                            "No response field in router API response"
                        )

                # If we got here with a response but couldn't parse it, break the retry loop
                break

            except requests.Timeout:
                self.debug_log.append(
                    f"Router API call timed out (attempt {attempt+1}/{max_retries+1})"
                )

                if attempt < max_retries:
                    self.debug_log.append(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

            except Exception as e:
                self.debug_log.append(f"Error in router API call: {e}")

                if attempt < max_retries:
                    self.debug_log.append(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        # If we got here, all retries failed
        self.debug_log.append("All router API attempts failed")
        return default_result

    def update_system_prompt(self):
        """Update the system prompt based on current available agents."""
        self.system_prompt = self._build_system_prompt()
