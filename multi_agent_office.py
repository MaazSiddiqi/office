#!/usr/bin/env python3

"""
AI Office - Multi-Agent Office
=============================

This module defines the MultiAgentOffice class, which is the central orchestration
component of the AI Office system. It manages multiple agent processes, routes queries
to the appropriate agents, and maintains the shared memory system.

The multi-agent architecture allows different specialized agents to work in parallel,
each with their own expertise, while sharing information through a common memory system.
"""

import os
import json
import requests
import time
import datetime
import re
from multiprocessing import Pipe
from agent_process import AgentProcess
from memory_manager import MemoryManagerProcess
from agent_trainer import AgentTrainerProcess
import uuid
from collections import defaultdict
import multiprocessing

# Configuration constants
DEFAULT_MODEL = "llama3.1:latest"
DEFAULT_API_URL = "http://localhost:11434/api/generate"
REGISTRY_FILE = "agent_registry.json"
MEMORY_FILE = "office_memory.json"

# Message types
MSG_TYPE_REQUEST = "request"
MSG_TYPE_SHUTDOWN = "shutdown"
MSG_TYPE_RESPONSE = "response"

# Memory operation types
MEMORY_OP_EXTRACT = "extract"
MEMORY_OP_READ = "read"
MEMORY_OP_QUERY = "query"

# Command constants
CMD_WHO = "/who"
CMD_SWITCH = "/switch_to"
CMD_EA = "/ea"
CMD_HELP = "/help"
CMD_CLEAR = "/clear"

# Thresholds and limits
MAX_FACTS_PER_QUERY = 5
TRAINER_CHECK_INTERVAL = 5.0
MEMORY_WAIT_TIMEOUT = 5.0
MIN_DELEGATION_CONFIDENCE = 0.6
MAX_CONTEXT_LENGTH = 5
EA_AGENT_NAME = "executive_assistant"


class MultiAgentOffice:
    """
    The main orchestration class for the multi-agent office system.

    This class manages all agent processes, routes queries to the appropriate agents,
    and maintains the shared memory system. It acts as the central hub for all
    communication between agents and with the user.

    Attributes:
        registry (dict): Dictionary of registered agents and their details
        agent_processes (dict): Dictionary of running agent processes
        agent_pipes (dict): Dictionary of communication pipes to agent processes
        memory_manager (MemoryManagerProcess): The memory manager process
        agent_trainer (AgentTrainerProcess): The agent training/improvement process
        current_agent (str): The agent currently in conversation with the user
        conversation_history (list): History of recent conversations
    """

    def __init__(self, registry_file=REGISTRY_FILE, memory_file=MEMORY_FILE):
        """
        Initialize the multi-agent office system.

        Args:
            registry_file (str): Path to the agent registry JSON file
            memory_file (str): Path to the memory storage JSON file
        """
        self.registry = {}  # Store agent details
        self.agent_processes = {}  # Store agent processes
        self.agent_pipes = {}  # Store communication pipes
        self.registry_file = registry_file
        self.memory_file = memory_file

        # Define the EA system prompt
        self.ea_system_prompt = """You are the Executive Assistant AI in our AI Office system. Your role is to:
1. Answer user questions directly when you can, using ONLY information from our memory system or provided in the current conversation
2. Delegate specific tasks to specialized agents when necessary
3. Maintain context and continuity throughout the conversation
4. Remember important information through our memory system

IMPORTANT - CONVERSATION FLOW:
- When a user's message follows directly after a specialized agent's response, assume it's likely a reply to that agent
- If a user's message seems to be responding to questions or information from a specialized agent, route it back to that agent
- Don't interrupt ongoing conversations between a user and a specialized agent unless:
  a) The user explicitly addresses you
  b) The user introduces a completely new topic
  c) The agent has completed its task (indicated by phrases like "Is there anything else I can help with?")

IMPORTANT - KNOWLEDGE LIMITATIONS:
- You have NO general knowledge about the world. You MUST ONLY reference:
- Facts stored in our memory system from past conversations
- Information explicitly stated by the user in the current conversation
- Your core functions and abilities

Never hallucinate information. If you don't have relevant information in memory or from the user, clearly state this and either:
1. Ask the user for the information you need
2. Delegate to a specialized agent with appropriate skills
3. Explain that you don't have this information in memory

Be helpful, professional, and efficient. When you don't know something, be honest and consider if another agent might be able to help."""

        # Initialize training status
        self.training_status = defaultdict(lambda: {
            "status": "idle",
            "last_updated": time.time(),
            "dissatisfaction_count": 0,
            "total_interactions": 0
        })

        # Flag to track user dissatisfaction
        self.previous_query_showed_dissatisfaction = False

        # Current agent tracking
        self.current_agent = None
        self.conversation_history = []

        # Initialize memory cache
        self.memory = {"facts": [], "conversation_history": []}

        # Start memory manager process
        self.memory_manager = MemoryManagerProcess(memory_file=memory_file)
        self.memory_manager.start()
        self.pending_memory_tasks = {}  # Track in-progress memory tasks

        # Add session tracking parameters
        self.session_timeout = 120  # Session timeout in seconds
        self.last_activity = time.time()  # Track last activity time
        self.last_memory_check = (
            time.time()
        )  # Track when we last checked for memory updates
        self.memory_check_interval = 1.0  # Check for memory updates every second

        # Start agent trainer process
        self.agent_trainer = AgentTrainerProcess(registry_file=registry_file)
        self.agent_trainer.start()
        self.pending_trainer_tasks = {}  # Track in-progress trainer tasks
        self.last_trainer_check = (
            time.time()
        )  # Track when we last checked for trainer updates
        self.trainer_check_interval = (
            TRAINER_CHECK_INTERVAL  # Check for trainer updates interval
        )
        self.auto_improve_agents = (
            True  # Flag to enable/disable automatic agent improvement
        )

        # Track interaction counts for agents
        self.agent_interaction_counts = defaultdict(int)

        # Load existing agents from registry
        self.load_registry()

        # Start agent processes
        self.start_agent_processes()

        # Set up tracking for unsatisfied queries
        self.recent_queries = []
        self.dissatisfaction_threshold = (
            3  # Number of dissatisfied queries before intervention
        )

    def load_registry(self):
        """Load agent registry from file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    self.registry = json.load(f)
                print(f"Loaded {len(self.registry)} agents from registry")
            except Exception as e:
                print(f"Error loading registry: {e}")
                self.registry = {}
        else:
            print("No registry file found. Creating default registry.")
            self.registry = {}

    def save_registry(self):
        """Save agent registry to file."""
        try:
            print(f"\n>> Saving registry to {self.registry_file}...")

            # Make a backup of the current registry file if it exists
            if os.path.exists(self.registry_file):
                backup_name = f"{self.registry_file}.bak"
                try:
                    with open(self.registry_file, "r") as old_f:
                        old_content = old_f.read()
                    with open(backup_name, "w") as backup_f:
                        backup_f.write(old_content)
                    print(f">> Created backup of previous registry at {backup_name}")
                except Exception as e:
                    print(f">> Warning: Failed to create backup: {e}")

            # Write the updated registry to file
            with open(self.registry_file, "w") as f:
                json.dump(self.registry, f, indent=2)

            # Verify the file was written correctly
            if os.path.exists(self.registry_file):
                with open(self.registry_file, "r") as verify_f:
                    content = verify_f.read()
                    if content and len(content) > 10:  # Simple sanity check
                        print(
                            f">> Registry saved successfully to {self.registry_file} ({len(content)} bytes)"
                        )
                        for agent_name in self.registry:
                            print(f">>   - Agent '{agent_name}' saved")
                        return True
                    else:
                        print(f">> Warning: Registry file seems empty or corrupted")
            else:
                print(f">> Error: Registry file not found after save operation")

        except Exception as e:
            print(f">> Error saving registry: {e}")
            import traceback

            print(traceback.format_exc())

        return False

    def _load_memory_blocking(self):
        """Load memory synchronously (used only at startup)"""
        task_id = str(uuid.uuid4())
        self.memory_manager.task_queue.put({"type": "read", "task_id": task_id})

        # Wait for result (blocking)
        result = None
        timeout = 5  # seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.memory_manager.result_queue.empty():
                result = self.memory_manager.result_queue.get()
                if result.get("task_id") == task_id:
                    return result.get(
                        "memory", {"facts": [], "conversation_history": []}
                    )
            time.sleep(0.1)

        # Fallback if timeout
        print("Warning: Memory load timeout, using empty memory")
        return {"facts": [], "conversation_history": []}

    def get_relevant_context(self, query, max_facts=5):
        """Get relevant facts from memory for a query."""
        if not self.memory["facts"]:
            return []

        # Extract key terms from the query for better matching
        query_terms = self._extract_key_terms(query)

        # Send asynchronous query request to memory manager
        task_id = str(uuid.uuid4())
        self.memory_manager.task_queue.put(
            {
                "type": "query",
                "query_terms": list(query_terms),
                "max_facts": max_facts,
                "task_id": task_id,
            }
        )

        # Wait for result (blocking, but should be fast)
        result = None
        timeout = 2  # seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.memory_manager.result_queue.empty():
                result = self.memory_manager.result_queue.get()
                if result.get("task_id") == task_id:
                    facts = result.get("facts", [])
                    if facts:
                        print(f"[Context] Found {len(facts)} relevant facts for query")
                    return facts
            time.sleep(0.1)

        # If we didn't get a quick response, try direct matching with our in-memory facts
        if not result:
            return self._direct_fact_matching(query_terms, max_facts)

        # Fallback to empty list
        return []

    def _extract_key_terms(self, query):
        """Extract key terms from the query, removing common stopwords."""
        # List of common stopwords to ignore
        stopwords = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "will",
            "would",
            "should",
            "may",
            "might",
            "must",
            "to",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "of",
        }

        # Normalize and split the query
        query_lower = query.lower()

        # Remove common punctuation
        for char in ".,;:!?()[]{}\"'":
            query_lower = query_lower.replace(char, " ")

        # Split and filter words
        words = query_lower.split()
        key_terms = set()

        for word in words:
            if (
                word not in stopwords and len(word) > 2
            ):  # Skip very short words and stopwords
                key_terms.add(word)

        # Always include certain important keywords regardless of stopword status
        important_keywords = ["meeting", "schedule", "appointment", "tomorrow", "today"]
        for keyword in important_keywords:
            if keyword in query_lower:
                key_terms.add(keyword)

        return key_terms

    def _direct_fact_matching(self, query_terms, max_facts=5):
        """Directly match query terms against in-memory facts."""
        if not query_terms or not self.memory["facts"]:
            return []

        # Score each fact by relevance
        scored_facts = []
        for fact in self.memory["facts"]:
            fact_lower = fact.lower()
            # Calculate a simple score based on presence of query terms
            score = sum(1 for term in query_terms if term in fact_lower)
            if score > 0:
                scored_facts.append((fact, score))

        # Sort by score (highest first) and return the top ones
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, score in scored_facts[:max_facts]]

    def setup_default_agents(self):
        """Set up default agents."""
        default_agents = {
            "calendar_manager": {
                "system_prompt": "You are a calendar management specialist. Your job is to organize schedules, create agendas, and manage time efficiently. You help with scheduling meetings, tracking appointments, and planning events. When asked about calendar-related tasks, provide detailed and organized responses.",
                "model": DEFAULT_MODEL,
                "temperature": 0.7,
            },
            "research_assistant": {
                "system_prompt": "You are a research specialist. Your job is to find information, summarize content, and provide well-researched answers. You excel at answering factual questions, synthesizing complex topics, and providing balanced perspectives on various subjects.",
                "model": DEFAULT_MODEL,
                "temperature": 0.7,
            },
            "technical_expert": {
                "system_prompt": "You are a technical expert specialized in software, programming, and technology. You provide detailed explanations of technical concepts, help debug code issues, and offer guidance on best practices in software development and IT.",
                "model": DEFAULT_MODEL,
                "temperature": 0.7,
            },
            "creative_writer": {
                "system_prompt": "You are a creative writing specialist. Your job is to generate engaging, original content including stories, poems, scripts, and marketing copy. You can adapt your writing style to different tones and formats as requested.",
                "model": DEFAULT_MODEL,
                "temperature": 0.7,
            },
            "personal_assistant": {
                "system_prompt": "You are a personal life assistant. You help with personal tasks, offer advice on daily challenges, suggest self-improvement strategies, and provide thoughtful recommendations for health, wellness, and lifestyle choices.",
                "model": DEFAULT_MODEL,
                "temperature": 0.7,
            },
        }

        self.registry = default_agents
        self.save_registry()

    def _generate_agent_system_prompt(self, agent_name, base_prompt):
        """Generate an enhanced system prompt in Alpaca format for an agent."""
        # Start with the base prompt
        if not base_prompt:
            return base_prompt

        # Don't modify already formatted prompts
        if "### Instruction:" in base_prompt:
            return base_prompt

        # Anti-hallucination and memory grounding instructions to add to all prompts
        memory_grounding = (
            "### MEMORY GROUNDING INSTRUCTIONS:\n"
            "- You have NO general knowledge about the world.\n"
            "- ONLY use information from:\n"
            "  1. Facts provided in the context from previous conversations\n"
            "  2. Information explicitly stated by the user in the current conversation\n"
            "  3. Your core functions and specialized skills\n"
            "- Never hallucinate or make up information not found in memory or the conversation.\n"
            "- If you don't have relevant information available, explicitly say so.\n"
            "- Always indicate the source of your information (memory, current conversation).\n"
            "- Ask clarifying questions when needed rather than making assumptions.\n\n"
        )

        # Format templates by agent type
        if agent_name == "calendar_manager":
            return (
                "### Instruction:\n"
                "You are a specialized Calendar Management Agent in an AI Office system. Your expertise is in organizing schedules, tracking appointments, and managing time efficiently.\n\n"
                "### Context:\n"
                "- You have access to facts about the user's schedule from previous conversations.\n"
                "- The user will ask you to organize meetings, track appointments, plan events, and help manage their time.\n"
                "- You must provide specific, actionable responses related to calendar and time management.\n\n"
                + memory_grounding +
                "### Guidelines:\n"
                f"{base_prompt}\n\n"
                "- Always clarify time zones when discussing meeting times.\n"
                "- When scheduling new events, check for conflicts with existing commitments.\n"
                "- Provide specific times and dates in your suggestions and confirmations.\n"
                "- Ask for clarification if the user's request lacks important details.\n"
                "- Organize information chronologically when presenting multiple events.\n"
                "- Suggest buffer time between meetings when appropriate.\n\n"
                "### Response Format:\n"
                "Provide well-structured responses with clear meeting details. Use bullet points for multiple items. When presenting schedules, use consistent formatting for times and dates."
            )

        elif agent_name == "research_assistant":
            return (
                "### Instruction:\n"
                "You are a specialized Research Assistant Agent in an AI Office system. Your expertise is in finding information, summarizing content, and providing well-researched answers.\n\n"
                "### Context:\n"
                "- The user will ask you to research topics, find information, summarize articles, and answer factual questions.\n"
                "- You should provide comprehensive, accurate, and balanced information on requested topics.\n"
                "- You must ONLY use information from the memory system or current conversation.\n\n"
                + memory_grounding +
                "### Guidelines:\n"
                f"{base_prompt}\n\n"
                "- Only present information that you can source from memory or the current conversation.\n"
                "- Clearly distinguish between facts, theories, and opinions in your responses.\n"
                "- Organize information logically with appropriate headings and sections.\n"
                "- When citing sources, provide specific attribution.\n"
                "- If you're uncertain about information, clearly indicate this rather than making guesses.\n\n"
                "### Response Format:\n"
                "Present information in a well-organized structure with headings when appropriate. Use bullet points for lists of facts or key points. For complex topics, provide a brief summary followed by detailed information."
            )

        elif agent_name == "technical_expert":
            return (
                "### Instruction:\n"
                "You are a specialized Technical Expert Agent in an AI Office system. Your expertise is in software, programming, and technology concepts.\n\n"
                "### Context:\n"
                "- The user will ask you technical questions, request code solutions, and seek debugging help.\n"
                "- You should provide clear technical explanations and practical coding solutions.\n"
                "- You must ONLY use information from the memory system, current conversation, or your core technical skills.\n\n"
                + memory_grounding +
                "### Guidelines:\n"
                f"{base_prompt}\n\n"
                "- Explain technical concepts in clear, accessible language.\n"
                "- When providing code, include comments to explain key sections.\n"
                "- Consider performance, security, and best practices in your solutions.\n"
                "- Break down complex problems into manageable steps.\n"
                "- Suggest debugging approaches for issues rather than just solutions.\n\n"
                "### Response Format:\n"
                "For explanations, use clear paragraphs with examples. For code solutions, provide properly formatted code blocks with syntax highlighting when possible. Include explanations before and after code examples."
            )

        elif agent_name == "creative_writer":
            return (
                "### Instruction:\n"
                "You are a specialized Creative Writing Agent in an AI Office system. Your expertise is in generating engaging, original content in various formats and styles.\n\n"
                "### Context:\n"
                "- The user will ask you to create stories, poems, scripts, marketing copy, and other creative content.\n"
                "- You should adapt your writing style to match the requested tone and format.\n"
                "- While you can be creative, any facts you reference must come from memory or the conversation.\n\n"
                + memory_grounding +
                "### Guidelines:\n"
                f"{base_prompt}\n\n"
                "- Craft engaging content with vibrant descriptions and compelling narratives.\n"
                "- Adapt your writing style to match the requested genre, tone, and audience.\n"
                "- Create original characters, settings, and plots when writing fiction.\n"
                "- Use appropriate literary techniques (metaphor, imagery, etc.) to enhance your writing.\n"
                "- When writing marketing copy, focus on benefits and persuasive language.\n\n"
                "### Response Format:\n"
                "Present creative content in a clean, well-formatted structure appropriate to the genre. For stories, use proper paragraphing. For poetry, use appropriate line breaks and stanzas. For scripts, follow standard script formatting."
            )

        elif agent_name == "personal_assistant":
            return (
                "### Instruction:\n"
                "You are a specialized Personal Life Assistant Agent in an AI Office system. Your expertise is in providing advice on daily challenges and personal development.\n\n"
                "### Context:\n"
                "- The user will ask you for advice on personal matters, self-improvement strategies, and lifestyle choices.\n"
                "- You should provide thoughtful, balanced guidance tailored to their needs.\n"
                "- Your advice must be based on information from memory or the current conversation, not general knowledge.\n\n"
                + memory_grounding +
                "### Guidelines:\n"
                f"{base_prompt}\n\n"
                "- Offer empathetic, practical advice for personal challenges.\n"
                "- Suggest realistic strategies for self-improvement and habit formation.\n"
                "- Provide balanced perspectives that consider the user's wellbeing.\n"
                "- Respect privacy and maintain appropriate boundaries in sensitive discussions.\n"
                "- When appropriate, suggest resources or tools that might be helpful.\n\n"
                "### Response Format:\n"
                "Provide advice in a supportive, conversational tone. Structure responses with clear recommendations and actionable steps when applicable. Use empathetic language that acknowledges the user's situation."
            )

        # Default format for other agents
        return (
            "### Instruction:\n"
            f"You are a specialized {agent_name} Agent in an AI Office system.\n\n"
            "### Context:\n"
            "- You are part of a team of AI agents with different specializations.\n"
            "- You have been selected to handle this task based on your specific expertise.\n"
            "- You must ONLY use information from the memory system or current conversation.\n\n"
            + memory_grounding +
            "### Guidelines:\n"
            f"{base_prompt}\n\n"
            "- Provide clear, specific, and actionable responses in your area of expertise.\n"
            "- Ask clarifying questions when necessary to better understand the user's needs.\n"
            "- Be concise but thorough in your explanations.\n\n"
            "### Response Format:\n"
            "Structure your responses logically with clear explanations. Use appropriate formatting to enhance readability."
        )

    def start_agent_processes(self):
        """Start processes for all agents in the registry."""
        for agent_name, details in self.registry.items():
            # Create a pipe for communication
            parent_conn, child_conn = Pipe()

            # Get the enhanced system prompt in Alpaca format
            system_prompt = self._generate_agent_system_prompt(
                agent_name, details["system_prompt"]
            )

            # Create and start the agent process
            agent_process = AgentProcess(
                agent_name=agent_name,
                system_prompt=system_prompt,
                model_name=details.get("model", DEFAULT_MODEL),
                pipe=child_conn,
            )

            # Store the process and pipe
            self.agent_processes[agent_name] = agent_process
            self.agent_pipes[agent_name] = parent_conn

            # Start the process
            agent_process.start()
            print(f"Started agent '{agent_name}' in process {agent_process.pid}")

    def shutdown_agent_processes(self):
        """Shutdown all agent processes and memory manager."""
        for agent_name, pipe in self.agent_pipes.items():
            try:
                # Send shutdown message
                pipe.send({"type": "shutdown"})
                print(f"Sent shutdown signal to agent '{agent_name}'")
            except Exception as e:
                print(f"Error shutting down agent '{agent_name}': {e}")

        # Wait for all processes to terminate
        for agent_name, process in self.agent_processes.items():
            process.join(timeout=2)  # Wait up to 2 seconds
            if process.is_alive():
                print(f"Force terminating agent '{agent_name}'")
                process.terminate()

        # Shutdown memory manager
        try:
            self.memory_manager.task_queue.put({"type": "shutdown"})
            self.memory_manager.join(timeout=2)
            if self.memory_manager.is_alive():
                print("Force terminating memory manager")
                self.memory_manager.terminate()
        except Exception as e:
            print(f"Error shutting down memory manager: {e}")

        # Shutdown agent trainer
        try:
            self.agent_trainer.task_queue.put({"type": "shutdown"})
            self.agent_trainer.join(timeout=2)
            if self.agent_trainer.is_alive():
                print("Force terminating agent trainer")
                self.agent_trainer.terminate()
        except Exception as e:
            print(f"Error shutting down agent trainer: {e}")

    def should_delegate_to_agent(self, query):
        """Determine if the EA should delegate this query to a specialized agent.

        Returns:
            tuple: (should_delegate, best_agent_name)
        """
        # First check if we can find a highly relevant agent using keyword matching
        best_agent = self.determine_best_agent(query)
        if not best_agent:
            return False, None

        # For basic or generic queries, EA can handle them
        query_lower = query.lower()

        # EA handles these types of queries directly
        ea_keywords = [
            "hello",
            "hi",
            "hey",
            "thank",
            "thanks",
            "how are you",
            "good morning",
            "good afternoon",
            "good evening",
            "what can you do",
            "who are you",
            "help me",
            "what is your name",
            "what time",
            "memory",
        ]

        if any(keyword in query_lower for keyword in ea_keywords):
            return False, None

        # EA handles short general queries
        if len(query.split()) < 4 and not any(char.isdigit() for char in query):
            return False, None

        # For specific domain keywords, delegate to specialists
        # Define tasks that DEFINITELY should be delegated based on specific keywords
        delegation_keywords = {
            "calendar_manager": [
                "schedule",
                "meeting",
                "appointment",
                "event",
                "calendar",
                "remind me",
                "agenda",
                "booking",
            ],
            "research_assistant": [
                "research",
                "information about",
                "find out",
                "look up",
                "search for",
                "data on",
                "statistics",
                "analyze",
            ],
            "technical_expert": [
                "code",
                "programming",
                "debug",
                "technical",
                "software",
                "algorithm",
                "database",
                "function",
                "api",
            ],
            "creative_writer": [
                "write a story",
                "poem",
                "creative",
                "fiction",
                "narrative",
                "compose",
                "draft",
                "screenplay",
                "script",
            ],
            "personal_assistant": [
                "life goals",
                "self-improvement",
                "routine",
                "habits",
                "lifestyle",
                "personal growth",
                "motivation",
                "health plan",
            ],
        }

        # Check if any delegation keywords are present
        for agent, keywords in delegation_keywords.items():
            if agent in self.registry and any(
                keyword in query_lower for keyword in keywords
            ):
                return True, agent

        # By default, have the EA handle it
        return False, None

    def determine_best_agent(self, query):
        """Determine which agent is best suited for a query."""
        # Simple keyword matching for fast agent selection
        query_lower = query.lower()

        # Define keyword patterns for common agent tasks
        agent_keywords = {
            "calendar_manager": [
                "calendar",
                "schedule",
                "meeting",
                "appointment",
                "agenda",
                "event",
                "time",
                "remind",
            ],
            "research_assistant": [
                "research",
                "find",
                "information",
                "article",
                "search",
                "learn about",
                "tell me about",
                "what is",
                "how does",
            ],
            "technical_expert": [
                "code",
                "program",
                "technical",
                "software",
                "computer",
                "programming",
                "debug",
                "fix",
                "build",
                "develop",
            ],
            "creative_writer": [
                "write",
                "story",
                "poem",
                "creative",
                "fiction",
                "script",
                "content",
                "create",
                "compose",
            ],
            "personal_assistant": [
                "advice",
                "help me with",
                "personal",
                "life",
                "recommend",
                "suggestion",
                "plan",
                "improve",
                "goal",
            ],
        }

        # Check for keyword matches
        for agent_name, keywords in agent_keywords.items():
            if agent_name in self.registry and any(
                keyword in query_lower for keyword in keywords
            ):
                return agent_name

        # If no match found, return None - EA will handle it
        return None

    def process_query(self, query):
        """Process a user query by routing it to the appropriate agent based on session state."""
        # Check for memory manager updates
        self.check_memory_updates()

        # Check for trainer updates
        self.check_trainer_updates()

        # Special commands always go to EA
        if query.startswith("/"):
            # Reset active agent session when a command is used
            self.current_agent = None
            return self.handle_command(query)

        # Check for explicit handoff command (non-slash version)
        if query.lower() == "talk to ea":
            self.current_agent = None
            return "I'm now talking to the Executive Assistant. How can I help you?"

        # Get the latest agent interaction from history (who responded last)
        last_agent = self._get_last_agent_from_history()

        # Get relevant context from memory
        context = self.get_relevant_context(query)

        # Update last activity time
        self.last_activity = time.time()

        # Check for session timeout
        if self.current_agent and (
            time.time() - self.last_activity > self.session_timeout
        ):
            # Session expired, reset to EA
            prev_agent = self.current_agent
            self.current_agent = None
            return f"Your session with {prev_agent} has timed out. I'm now handling your requests. How can I help you?"

        # If there is no current agent but there was a last agent that isn't the EA,
        # and the user's message appears to be responding to that agent
        if (not self.current_agent and
            last_agent and
            last_agent != "executive_assistant" and
            last_agent in self.registry and
            self._appears_to_be_response_to_agent(query)):

            print(f"User appears to be responding to {last_agent}, routing message there...")
            self.current_agent = last_agent
            return self.delegate_to_agent(last_agent, query, context)

        # Route based on active agent
        if self.current_agent:
            # Implicit topic change detection - check if query seems unrelated to active agent
            if self.is_topic_change(query, self.current_agent):
                # Determine if another agent is better for this query
                should_delegate, new_agent = self.should_delegate_to_agent(query)

                if should_delegate:
                    # Topic has changed to another agent's expertise
                    prev_agent = self.current_agent
                    self.current_agent = new_agent
                    print(f"Conversation shifted from {prev_agent} to {new_agent}...")
                    return self.delegate_to_agent(new_agent, query, context)
                else:
                    # Topic has changed to something the EA can handle
                    prev_agent = self.current_agent
                    self.current_agent = None
                    print(f"Conversation shifted from {prev_agent} to EA...")
                    return self.handle_with_ea(query, context)

            # Continue with the active agent
            print(f"Continuing conversation with agent '{self.current_agent}'...")
            return self.delegate_to_agent(self.current_agent, query, context)
        else:
            # No active agent (EA is handling), determine if we should delegate
            should_delegate, agent_name = self.should_delegate_to_agent(query)

            if should_delegate and agent_name:
                # Begin a new session with this agent
                self.current_agent = agent_name
                print(f"EA is delegating to agent '{agent_name}'...")
                return self.delegate_to_agent(agent_name, query, context)
            else:
                print("EA is handling this directly...")
                return self.handle_with_ea(query, context)

    def _get_last_agent_from_history(self):
        """Get the most recent agent that responded in the conversation."""
        if not self.memory["conversation_history"]:
            return None

        # Get the most recent interaction
        last_interaction = self.memory["conversation_history"][-1]
        return last_interaction.get("agent")

    def _appears_to_be_response_to_agent(self, query):
        """Check if a query appears to be responding to a previous agent message."""
        # Check for short responses that are likely replies
        if len(query.split()) <= 5:
            return True

        # Check for response patterns
        response_indicators = [
            "yes", "no", "ok", "sure", "thanks", "thank", "good", "great",
            "here", "that", "those", "they", "it", "this"
        ]
        query_lower = query.lower().strip()

        # Check for answers to questions
        if query_lower.startswith(tuple(response_indicators)):
            return True

        # Check for response to a request for information
        if (":" in query or "," in query_lower.split()[0]) and len(query.split()) <= 10:
            return True

        # Check for direct answers to questions about times, dates, etc.
        time_patterns = re.compile(r'\b(\d{1,2}[:\.]\d{2}|\d{1,2}\s*(am|pm|AM|PM))\b')
        if time_patterns.search(query):
            return True

        return False

    def is_topic_change(self, query, current_agent):
        """Determine if the query represents a topic change from the current agent's domain."""
        query_lower = query.lower()

        # Define patterns that indicate continuing the conversation
        continuation_patterns = [
            "yes",
            "no",
            "maybe",
            "sure",
            "okay",
            "ok",
            "thanks",
            "thank you",
            "please",
            "go ahead",
            "continue",
            "next",
            "more",
            "tell me more",
            "explain",
            "elaborate",
        ]

        # If the query is a short continuation, don't consider it a topic change
        if (
            any(
                pattern == query_lower or pattern in query_lower.split()
                for pattern in continuation_patterns
            )
            and len(query.split()) < 4
        ):
            return False

        # Check if query matches other agents' keywords better
        best_agent = self.determine_best_agent(query)

        # If we find a better agent match and it's different from current agent
        if best_agent and best_agent != current_agent:
            # Only consider it a topic change if the match is strong
            agent_keywords = {
                "calendar_manager": ["calendar", "schedule", "meeting", "appointment"],
                "research_assistant": ["research", "information", "find", "search"],
                "technical_expert": ["code", "program", "technical", "software"],
                "creative_writer": ["write", "story", "creative", "content"],
                "personal_assistant": ["advice", "personal", "life", "recommend"],
            }

            # Check if any strong keywords for the new agent are in the query
            if best_agent in agent_keywords:
                for keyword in agent_keywords[best_agent]:
                    if keyword in query_lower:
                        return True

        # By default, assume it's not a topic change
        return False

    def _get_recent_conversation_context(self, agent_name=None, max_entries=3):
        """Get recent conversation context relevant to the current interaction.

        Args:
            agent_name: If provided, only get conversations with this agent
            max_entries: Maximum number of conversation entries to include

        Returns:
            str: Formatted conversation context string
        """
        if not self.memory["conversation_history"]:
            return ""

        # Get relevant history entries
        history = self.memory["conversation_history"]
        relevant_history = []

        if agent_name:
            # Get conversations with specified agent
            agent_history = [entry for entry in history if entry.get("agent") == agent_name]
            relevant_history = agent_history[-max_entries:] if agent_history else []
        else:
            # Get most recent conversations
            relevant_history = history[-max_entries:]

        if not relevant_history:
            return ""

        # Format the conversation context
        context_str = "\nRECENT CONVERSATION HISTORY:\n"
        for entry in relevant_history:
            user_input = entry.get("user_input", "")
            agent = entry.get("agent", "unknown")
            response = entry.get("response", "")

            agent_display_name = "Assistant" if agent == "executive_assistant" else agent.replace("_", " ").title()
            context_str += f"User: {user_input}\n{agent_display_name}: {response}\n\n"

        return context_str

    def handle_with_ea(self, query, context=None):
        """Have the Executive Assistant handle a query directly."""
        # Create the prompt with context if available
        memory_context = ""
        if context:
            print(f"Adding {len(context)} relevant facts from memory")
            memory_context = "IMPORTANT MEMORY CONTEXT:\n"
            memory_context += "The following facts are from our memory system and represent information from previous conversations. These are the ONLY facts you should reference outside of what's in the current conversation:\n\n"

            # Sort context by likely relevance
            sorted_context = self._order_context_by_relevance(context, query)

            for i, fact in enumerate(sorted_context, 1):
                memory_context += f"FACT {i}: {fact}\n"

            # Add a reminder to use this context
            memory_context += "\nYou MUST ONLY reference these facts and information provided in the current conversation. Do not hallucinate or make up additional information."
        else:
            memory_context = "MEMORY CONTEXT: No relevant facts found in memory. You MUST ONLY reference information provided in the current conversation. Do not hallucinate or make up additional information."

        # Add recent conversation context
        conversation_context = self._get_recent_conversation_context()

        # Combine all context elements
        full_prompt = f"{memory_context}\n\n{conversation_context}\n===\n\nCURRENT USER REQUEST: {query}"

        # Create the LLM payload
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": full_prompt,
            "stream": True,
            "options": {"temperature": 0.7, "max_tokens": 1000},
            "system": self.ea_system_prompt,
        }

        try:
            # Call the LLM API
            response = requests.post(DEFAULT_API_URL, json=payload, stream=True)
            response.raise_for_status()

            result = ""
            # Always stream the output - we're always running in the main context now
            should_print_directly = True

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line)
                if "response" in data:
                    chunk = data["response"]
                    result += chunk
                    # Always print chunks directly
                    if should_print_directly:
                        print(chunk, end="", flush=True)

                if data.get("done", False):
                    if should_print_directly:
                        print()
                        print("-" * 40)
                    break

            # Extract facts and add to memory asynchronously
            task_id = str(uuid.uuid4())
            self.memory_manager.task_queue.put(
                {
                    "type": "extract",
                    "query": query,
                    "response": result,
                    "agent": "executive_assistant",
                    "task_id": task_id,
                }
            )

            print("Memory extraction in progress...")

            # Store task ID for potential later checking
            self.pending_memory_tasks[task_id] = time.time()

            # Clean up old pending tasks
            self._cleanup_pending_tasks()

            # Record interaction with EA for training
            self._record_interaction("executive_assistant", query, result)

            # Check if previous query showed dissatisfaction, and this is a follow-up
            if self.previous_query_showed_dissatisfaction:
                # Reset the flag
                self.previous_query_showed_dissatisfaction = False

                # If EA handled it directly after dissatisfaction, analyze active agent
                if self.current_agent:
                    self._analyze_agent_performance(self.current_agent)

            # Add to conversation history
            self.add_to_history(query, "executive_assistant", result)

            # Check for memory updates after processing
            self.check_memory_updates()

            # Check for trainer updates after processing
            self.check_trainer_updates()

            return result
        except Exception as e:
            print(f"Error processing with EA: {e}")
            return (
                f"I apologize, but I encountered an error processing your request: {e}"
            )

    def _order_context_by_relevance(self, context, query):
        """Sort context facts by relevance to the current query."""
        if not context or len(context) <= 1:
            return context

        # Extract key terms from query
        query_terms = self._extract_key_terms(query)

        # Score each fact by relevance to the query
        scored_facts = []
        for fact in context:
            fact_lower = fact.lower()

            # Calculate how many query terms appear in this fact
            term_matches = sum(1 for term in query_terms if term in fact_lower)

            # Also check recency - assume more recent facts are at the end of the list
            # This is a simplified assumption, but works as a heuristic
            recency_score = context.index(fact) / len(context)

            # Combined score weights both relevance and recency
            total_score = term_matches + (recency_score * 0.5)

            scored_facts.append((fact, total_score))

        # Sort by relevance score, highest first
        scored_facts.sort(key=lambda x: x[1], reverse=True)

        # Return just the sorted facts
        return [fact for fact, _ in scored_facts]

    def delegate_to_agent(self, agent_name, query, context=None):
        """Delegate a query to a specialized agent."""
        # Send the query to the agent process
        try:
            memory_context = []
            if context:
                print(f"Adding {len(context)} relevant facts from memory")

                # Sort context for better relevance
                memory_context = self._order_context_by_relevance(context, query)

            # Add recent conversation context
            conversation_context = self._get_recent_conversation_context(agent_name=agent_name)

            self.agent_pipes[agent_name].send(
                {
                    "type": "request",
                    "query": query,
                    "context": memory_context if memory_context else [],
                    "conversation_history": conversation_context
                }
            )

            # Print an indicator that we're getting a response from this agent
            print(f"[Agent: {agent_name}] ", end="", flush=True)

            # Initialize response variables
            full_response = ""
            facts = []

            # Always stream the output - we're always running in the main context now
            should_print_directly = True

            while True:
                if self.agent_pipes[agent_name].poll(
                    timeout=60
                ):  # Wait up to 60 seconds
                    response_data = self.agent_pipes[agent_name].recv()

                    # Handle streaming chunks
                    if response_data["type"] == "chunk":
                        chunk = response_data["chunk"]
                        full_response += chunk
                        # Always print chunks directly
                        if should_print_directly:
                            print(chunk, end="", flush=True)  # Stream to console

                    # Handle completion message with extracted facts
                    elif response_data["type"] == "complete":
                        # Always print completion markers
                        if should_print_directly:
                            print()  # Add newline after streaming is complete
                            print("-" * 40)  # Add separator line similar to EA output

                        # Extract facts directly here if not provided by agent
                        extracted_facts = response_data.get("extracted_facts", [])

                        # If no facts were extracted by the agent, try to extract some ourselves
                        if not extracted_facts:
                            # Send the response to memory manager for fact extraction
                            task_id = str(uuid.uuid4())
                            self.memory_manager.task_queue.put(
                                {
                                    "type": "extract",
                                    "query": query,
                                    "response": full_response,
                                    "agent": agent_name,
                                    "task_id": task_id,
                                }
                            )
                            print("Memory extraction in progress...")
                            self.pending_memory_tasks[task_id] = time.time()
                        else:
                            # Use the pre-extracted facts
                            facts = extracted_facts
                            # Send facts to memory manager
                            task_id = str(uuid.uuid4())
                            self.memory_manager.task_queue.put(
                                {
                                    "type": "extract",
                                    "query": query,
                                    "response": full_response,
                                    "agent": agent_name,
                                    "task_id": task_id,
                                    "facts": facts,  # Pass pre-extracted facts
                                }
                            )
                            print(
                                f"Sending {len(facts)} pre-extracted facts to memory manager..."
                            )
                            self.pending_memory_tasks[task_id] = time.time()

                        # Record this interaction with the agent for training purposes
                        self._record_interaction(agent_name, query, full_response)

                        # Check if this query shows dissatisfaction
                        if self._check_query_for_dissatisfaction(query):
                            self.previous_query_showed_dissatisfaction = True
                            # Analyze agent performance
                            self._analyze_agent_performance(agent_name)
                        else:
                            self.previous_query_showed_dissatisfaction = False

                        # Add to conversation history
                        self.add_to_history(query, agent_name, full_response)

                        # Check if the agent wants to hand back control
                        if self.should_agent_handoff(query, full_response):
                            self.current_agent = None
                            print(
                                "Agent has completed its task and handed control back to EA"
                            )
                            # Return an empty string instead of the full response to prevent duplicate printing
                            # Since the response has already been streamed to the console
                            return ""

                        return full_response

                    # Handle error message
                    elif response_data["type"] == "error":
                        self.current_agent = None  # Reset to EA on error
                        return f"Error from {agent_name}: {response_data['error']}"
                else:
                    self.current_agent = None  # Reset to EA on timeout
                    return f"Timeout waiting for response from {agent_name}."
        except Exception as e:
            self.current_agent = None  # Reset to EA on exception
            return f"Error communicating with agent {agent_name}: {e}"

    def should_agent_handoff(self, query, response):
        """Determine if the agent should hand control back to the EA based on the response."""
        # Check if response explicitly indicates a completed task
        completion_indicators = [
            "anything else I can help you with",
            "anything else I can assist you with",
            "anything else you would like to know",
            "any other questions",
            "any other tasks",
            "is there anything else you need help with",
        ]

        response_lower = response.lower()

        # Only hand off if there's a very clear completion phrase
        if any(indicator in response_lower for indicator in completion_indicators):
            return True

        # Check if the response ends with a question that doesn't require continuation
        final_sentences = re.split(r'[.!?]\s+', response.strip())
        if final_sentences:
            last_sentence = final_sentences[-1].lower()
            # If the last sentence is not a question and contains a completion indicator
            if not last_sentence.endswith("?") and any(
                word in last_sentence for word in ["let me know", "feel free", "don't hesitate"]
            ):
                return True

        # More conservative about handing back to EA - in most cases, maintain the conversation
        return False

    def _cleanup_pending_tasks(self):
        """Clean up old pending memory tasks."""
        now = time.time()
        old_tasks = [
            task_id
            for task_id, timestamp in self.pending_memory_tasks.items()
            if now - timestamp > 60
        ]  # Older than 60 seconds

        for task_id in old_tasks:
            del self.pending_memory_tasks[task_id]

    def add_to_history(self, query, agent_name, response):
        """Add a conversation to the history."""
        # Update in-memory copy for immediate use
        timestamp = datetime.datetime.now().isoformat()
        history_entry = {
            "timestamp": timestamp,
            "user_input": query,
            "agent": agent_name,
            "response": response if len(response) <= 200 else response[:197] + "...",  # Truncate long responses
        }

        self.memory["conversation_history"].append(history_entry)

        # Keep only the last 50 conversations in memory
        if len(self.memory["conversation_history"]) > 50:
            self.memory["conversation_history"] = self.memory["conversation_history"][-50:]

        # Force synchronize with memory file to ensure persistence
        task_id = str(uuid.uuid4())
        self.memory_manager.task_queue.put({
            "type": "add_history",
            "history_entry": history_entry,
            "task_id": task_id
        })

        self.pending_memory_tasks[task_id] = time.time()
        print(f"Added conversation to history and sent to memory manager")

        # Debug print
        print(f"Current memory has {len(self.memory['conversation_history'])} history entries and {len(self.memory['facts'])} facts")

    def handle_command(self, command):
        """Handle special commands."""
        # Reset active agent
        self.current_agent = None

        # Handle existing commands
        if command == "/list_agents":
            return "Available Agents:\n" + "\n".join(
                [f"- {name}" for name in self.registry.keys()]
            )

        elif command == "/ea":
            return "You are now talking to the Executive Assistant. How can I help you?"

        # Add command to show training status
        elif command == "/training_status":
            # Format a report of training status for all agents
            if not self.training_status:
                return "No agent training data available yet."

            status_report = "=== AGENT TRAINING STATUS ===\n\n"

            for agent_name, status_data in self.training_status.items():
                if agent_name not in self.registry:
                    continue  # Skip if agent doesn't exist anymore

                status = status_data.get("status", "idle")
                last_updated = status_data.get("last_updated", 0)
                time_since = time.time() - last_updated

                status_report += f"Agent: {agent_name}\n"
                status_report += f"  Current status: {status}\n"
                status_report += f"  Interactions recorded: {self.agent_interaction_counts.get(agent_name, 0)}\n"

                if status == "improved":
                    improvement_time = status_data.get("improvement_time", "unknown")
                    status_report += f"  Last improved: {improvement_time}\n"

                if time_since < 3600:
                    status_report += (
                        f"  Last activity: {int(time_since / 60)} minutes ago\n"
                    )
                else:
                    status_report += (
                        f"  Last activity: {int(time_since / 3600)} hours ago\n"
                    )

                status_report += "\n"

            return status_report

        # Add new command for providing direct agent feedback
        elif command.startswith("/feedback"):
            # Format: /feedback agent_name "feedback text"
            parts = re.match(r"/feedback\s+(\w+)\s+\"(.+)\"", command)
            if parts:
                agent_name = parts.group(1)
                feedback_text = parts.group(2)

                if agent_name in self.registry:
                    # Send feedback to agent trainer
                    return self._provide_direct_feedback(agent_name, feedback_text)
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return 'Invalid format. Use: /feedback agent_name "your feedback here"'

        # Add command to toggle automatic agent improvement
        elif command == "/toggle_auto_improve":
            self.auto_improve_agents = not self.auto_improve_agents
            return f"Automatic agent improvement {'enabled' if self.auto_improve_agents else 'disabled'}"

        # Add command to force analyze an agent
        elif command.startswith("/analyze_agent"):
            parts = command.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                if agent_name in self.registry:
                    # Trigger analysis
                    self._analyze_agent_performance(agent_name)
                    return f"Analysis of agent '{agent_name}' has been triggered. Results will be available soon."
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return "Invalid format. Use: /analyze_agent <agent_name>"

        # Add command to force improve an agent
        elif command.startswith("/improve_agent"):
            parts = command.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                if agent_name in self.registry:
                    # Trigger improvement
                    return self._trigger_agent_improvement(agent_name)
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return "Invalid format. Use: /improve_agent <agent_name>"

        elif command.startswith("/add_agent"):
            # Format: /add_agent name "system prompt"
            parts = command.split(" ", 2)
            if len(parts) >= 3:
                agent_name = parts[1]
                system_prompt = parts[2].strip('"')

                # Get enhanced prompt in Alpaca format
                enhanced_prompt = self._generate_agent_system_prompt(
                    agent_name, system_prompt
                )

                # Add to registry - store the original prompt
                self.registry[agent_name] = {
                    "system_prompt": system_prompt,
                    "model": DEFAULT_MODEL,
                    "temperature": 0.7,
                }
                self.save_registry()

                # Start the new agent process with the enhanced prompt
                parent_conn, child_conn = Pipe()
                agent_process = AgentProcess(
                    agent_name=agent_name,
                    system_prompt=enhanced_prompt,
                    model_name=DEFAULT_MODEL,
                    pipe=child_conn,
                )
                self.agent_processes[agent_name] = agent_process
                self.agent_pipes[agent_name] = parent_conn
                agent_process.start()

                return f"Agent '{agent_name}' added to registry and started with enhanced Alpaca-style prompt"
            else:
                return 'Invalid format. Use: /add_agent name "system prompt"'

        elif command.startswith("/delete_agent"):
            # Format: /delete_agent name
            parts = command.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                if agent_name in self.registry:
                    # Shutdown the process
                    if agent_name in self.agent_processes:
                        try:
                            self.agent_pipes[agent_name].send({"type": "shutdown"})
                            self.agent_processes[agent_name].join(timeout=2)
                            if self.agent_processes[agent_name].is_alive():
                                self.agent_processes[agent_name].terminate()
                            del self.agent_processes[agent_name]
                            del self.agent_pipes[agent_name]
                        except Exception as e:
                            print(f"Error shutting down agent '{agent_name}': {e}")

                    # Remove from registry
                    del self.registry[agent_name]
                    self.save_registry()
                    return f"Agent '{agent_name}' deleted from registry"
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return "Invalid format. Use: /delete_agent name"

        elif command.startswith("/view_agent"):
            # Format: /view_agent name
            parts = command.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                if agent_name in self.registry:
                    return (
                        f"Agent: {agent_name}\n"
                        + f"-" * 40
                        + "\n"
                        + f"System Prompt: {self.registry[agent_name].get('system_prompt', 'No system prompt')}\n"
                        + f"-" * 40
                    )
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return "Invalid format. Use: /view_agent name"

        elif command.startswith("/edit_agent"):
            # Format: /edit_agent name "new system prompt"
            parts = command.split(" ", 2)
            if len(parts) >= 3:
                agent_name = parts[1]
                new_system_prompt = parts[2].strip('"')

                if agent_name in self.registry:
                    # Update registry with original prompt
                    self.registry[agent_name]["system_prompt"] = new_system_prompt
                    self.save_registry()

                    # Get enhanced prompt in Alpaca format
                    enhanced_prompt = self._generate_agent_system_prompt(
                        agent_name, new_system_prompt
                    )

                    # Restart the agent process with the new prompt
                    if agent_name in self.agent_processes:
                        # Shutdown old process
                        try:
                            self.agent_pipes[agent_name].send({"type": "shutdown"})
                            self.agent_processes[agent_name].join(timeout=2)
                            if self.agent_processes[agent_name].is_alive():
                                self.agent_processes[agent_name].terminate()
                        except Exception as e:
                            print(f"Error shutting down agent '{agent_name}': {e}")

                    # Start new process with enhanced prompt
                    parent_conn, child_conn = Pipe()
                    agent_process = AgentProcess(
                        agent_name=agent_name,
                        system_prompt=enhanced_prompt,
                        model_name=self.registry[agent_name].get(
                            "model", DEFAULT_MODEL
                        ),
                        pipe=child_conn,
                    )
                    self.agent_processes[agent_name] = agent_process
                    self.agent_pipes[agent_name] = parent_conn
                    agent_process.start()

                    return f"Updated system prompt for agent '{agent_name}' and restarted the agent process with enhanced Alpaca-style prompt"
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return 'Invalid format. Use: /edit_agent name "new system prompt"'

        elif command.startswith("/memory"):
            # Format: /memory [query]
            parts = command.split(" ", 1)
            query = parts[1] if len(parts) > 1 else None

            if not self.memory["facts"]:
                return "No facts have been stored in memory yet."

            if query:
                # Filter facts by query terms
                query_terms = set(query.lower().split())
                matching_facts = []

                for fact in self.memory["facts"]:
                    fact_terms = set(fact.lower().split())
                    # Check for any overlap
                    if query_terms.intersection(fact_terms):
                        matching_facts.append(fact)

                if not matching_facts:
                    return f"No facts matching '{query}' found in memory."

                result = f"Found {len(matching_facts)} facts matching '{query}':\n\n"
                for i, fact in enumerate(matching_facts[:10], 1):
                    result += f"{i}. {fact}\n"

                if len(matching_facts) > 10:
                    result += f"\n...and {len(matching_facts) - 10} more facts."

                return result
            else:
                # Show all facts
                result = (
                    f"There are {len(self.memory['facts'])} stored facts in memory:\n\n"
                )
                for i, fact in enumerate(self.memory["facts"][:10], 1):
                    result += f"{i}. {fact}\n"

                if len(self.memory["facts"]) > 10:
                    result += f"\n...and {len(self.memory['facts']) - 10} more facts."

                return result

        elif command == "/clear_memory":
            task_id = str(uuid.uuid4())
            self.memory_manager.task_queue.put({"type": "clear", "task_id": task_id})
            print("Memory clear in progress...")

            # Also clear the in-memory copy
            self.memory = {"facts": [], "conversation_history": []}
            return "Memory clearing in progress..."

        elif command == "/deduplicate_memory":
            original_count = len(self.memory["facts"])
            if original_count == 0:
                return "Memory is empty. Nothing to deduplicate."

            # First, ensure we have the latest memory
            self._force_memory_sync()
            original_count = len(self.memory["facts"])  # Update count after sync

            # Create a new list to hold unique facts
            unique_facts = []

            # Process each fact to remove duplicates and similar facts
            for fact in self.memory["facts"]:
                should_add = True
                facts_to_remove = []

                # Compare with facts we've already decided to keep
                for i, existing_fact in enumerate(unique_facts):
                    similarity = self._calculate_fact_similarity(
                        fact.lower(), existing_fact.lower()
                    )

                    # If very similar or one contains the other
                    if (
                        similarity > 0.7
                        or existing_fact.lower() in fact.lower()
                        or fact.lower() in existing_fact.lower()
                    ):
                        # Keep the longer/more detailed one
                        if len(fact) > len(existing_fact):
                            facts_to_remove.append(i)
                        else:
                            should_add = False
                            break

                # Remove any facts that this one supersedes
                for idx in sorted(facts_to_remove, reverse=True):
                    unique_facts.pop(idx)

                # Add this fact if it's not redundant
                if should_add:
                    unique_facts.append(fact)

            # Update our memory
            self.memory["facts"] = unique_facts

            # Send the deduplicated facts to the memory manager - first clear and wait
            task_id = str(uuid.uuid4())
            self.memory_manager.task_queue.put({"type": "clear", "task_id": task_id})
            self.pending_memory_tasks[task_id] = time.time()

            # Wait for clear operation to complete
            clear_completed = False
            start_time = time.time()
            while time.time() - start_time < 2.0:  # 2 second timeout
                if task_id not in self.pending_memory_tasks:
                    clear_completed = True
                    break
                self.check_memory_updates()  # Process any results
                time.sleep(0.1)

            if not clear_completed:
                print("Warning: Memory clear operation timeout - proceeding anyway")

            # Now re-add all the unique facts in bulk
            # We'll use multiple batches to avoid overwhelming the queue
            batch_size = 10
            for i in range(0, len(unique_facts), batch_size):
                batch = unique_facts[i : i + batch_size]
                task_id = str(uuid.uuid4())
                self.memory_manager.task_queue.put(
                    {
                        "type": "extract",
                        "query": "Memory deduplication batch",
                        "response": "System operation",
                        "agent": "system",
                        "facts": batch,
                        "task_id": task_id,
                    }
                )
                self.pending_memory_tasks[task_id] = time.time()

            # Return a summary
            removed = original_count - len(unique_facts)
            if removed > 0:
                return f"Memory deduplicated. Removed {removed} redundant facts. {len(unique_facts)} unique facts remain."
            else:
                return "No duplicate facts found in memory."

        # Add new commands for agent switching
        elif command.startswith("/switch_to"):
            parts = command.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                if agent_name in self.registry:
                    self.current_agent = agent_name
                    return f"Switched conversation to agent '{agent_name}'. What would you like to discuss?"
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return "Invalid format. Use: /switch_to <agent_name>"

        elif command == "/who":
            if self.current_agent:
                return f"You are currently talking to: {self.current_agent}"
            else:
                return "You are currently talking to the Executive Assistant"

        # Add a command to simulate negative feedback for testing the training process
        elif command.startswith("/simulate_dissatisfaction"):
            parts = command.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                if agent_name in self.registry:
                    # Create a simulated dissatisfaction event
                    print(f"\n{'!'*60}")
                    print(
                        f"! AGENT TRAINER: Simulating user dissatisfaction with {agent_name}"
                    )
                    print(
                        f"! This will trigger the training process for demonstration purposes"
                    )
                    print(f"{'!'*60}")

                    # Add a direct training note
                    self.agent_trainer.task_queue.put(
                        {
                            "type": "direct_feedback",
                            "agent_name": agent_name,
                            "feedback": "The agent responses need improvement with more specific examples and better clarity",
                            "task_id": str(uuid.uuid4()),
                        }
                    )

                    # Also create artificial dissatisfaction records
                    for i in range(3):  # Create 3 dissatisfaction records
                        self.agent_trainer.task_queue.put(
                            {
                                "type": "record_interaction",
                                "agent_name": agent_name,
                                "query": f"that's not what I asked (simulated dissatisfaction {i+1})",
                                "response": "I apologize for misunderstanding your question.",
                                "timestamp": time.time()
                                - (i * 60),  # Spread them out a bit
                                "task_id": str(uuid.uuid4()),
                            }
                        )

                    # Trigger the analysis
                    self._analyze_agent_performance(agent_name)

                    return f"Simulated dissatisfaction with agent '{agent_name}'. The training process has been initiated."
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return "Invalid format. Use: /simulate_dissatisfaction <agent_name>"

        # Add command to force update the registry for direct testing
        elif command.startswith("/force_update_registry"):
            parts = command.split(" ", 2)
            if len(parts) >= 3:
                agent_name = parts[1]
                test_addition = parts[2].strip('"')

                if agent_name in self.registry:
                    # Get the current prompt
                    current_prompt = self.registry[agent_name]["system_prompt"]

                    # Create an enhanced prompt
                    improved_prompt = current_prompt
                    if "### Guidelines:" in current_prompt:
                        # Add to the guidelines section
                        improved_prompt = current_prompt.replace(
                            "### Guidelines:", f"### Guidelines:\n- {test_addition}\n"
                        )
                    else:
                        # Just append to the end if no guidelines section
                        improved_prompt = current_prompt + f"\n\n- {test_addition}"

                    # Apply the improved prompt
                    success = self._apply_improved_prompt(agent_name, improved_prompt)

                    if success:
                        return f"Forced registry update for '{agent_name}'. Registry file has been updated."
                    else:
                        return f"Failed to update registry for '{agent_name}'."
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                return 'Invalid format. Use: /force_update_registry <agent_name> "test addition text"'

        else:
            return (
                "Unknown command. Available commands:\n"
                + "/list_agents - Show all available agents\n"
                + '/add_agent <name> "<system prompt>" - Create a new agent\n'
                + "/delete_agent <name> - Remove an agent\n"
                + "/view_agent <name> - View an agent's system prompt\n"
                + '/edit_agent <name> "<new system prompt>" - Update an agent\n'
                + "/memory - Show all facts stored in memory\n"
                + "/memory <query> - Show facts matching a specific query\n"
                + "/clear_memory - Delete all stored memories\n"
                + "/deduplicate_memory - Remove redundant facts\n"
                + '/feedback <agent_name> "your feedback" - Provide direct feedback about an agent\n'
                + "/toggle_auto_improve - Enable/disable automatic agent improvement\n"
                + "/analyze_agent <agent_name> - Analyze an agent's performance\n"
                + "/improve_agent <agent_name> - Trigger improvement for an agent\n"
                + "/training_status - Show current training status for all agents\n"
                + "/simulate_dissatisfaction <agent_name> - Simulate negative feedback for testing\n"
                + '/force_update_registry <agent_name> "test text" - Force update registry for testing\n'
                + "\nConversation Session Commands:\n"
                + "/who - Shows which agent you're currently talking to\n"
                + "/switch_to <agent> - Explicitly switch to talking with a specific agent\n"
                + "/ea - Switch back to talking with the Executive Assistant"
            )

    def check_memory_updates(self):
        """Check for and display memory updates from the memory manager."""
        # Non-blocking check for memory updates
        try:
            if self.memory_manager.result_queue.empty():
                return

            while not self.memory_manager.result_queue.empty():
                result = self.memory_manager.result_queue.get_nowait()

                # Support both old and new formats
                operation = result.get("operation", result.get("type"))
                status = result.get(
                    "status", "success"
                )  # Default to success for old format

                if (
                    operation in ["extract_facts", "extract_result"]
                    and status == "success"
                ):
                    facts = result.get("facts", [])
                    if facts:
                        print("\n--- Memory Update ---")
                        print(f"Added {len(facts)} new facts to memory:")
                        for i, fact in enumerate(facts, 1):
                            print(f"  {i}. {fact}")
                        print("--------------------")

                        # Update the in-memory copy with the same logic as the memory manager
                        merge_results = self._merge_facts_to_memory(facts)

                        # If these facts were already skipped by the memory manager but we're adding them here,
                        # we need to force a synchronization to get the correct memory state
                        if len(merge_results["added"]) != len(facts):
                            self._force_memory_sync()

                elif operation in ["get_memory", "read_result"] and status == "success":
                    # Memory retrieval - update our local copy to ensure consistency
                    if "memory" in result:
                        retrieved_memory = result.get("memory")
                        if retrieved_memory and "facts" in retrieved_memory:
                            # Only update if we got a valid memory object
                            self.memory = retrieved_memory

                elif status == "error" or operation == "error":
                    error_msg = result.get(
                        "message", result.get("error", "Unknown error")
                    )
                    print(f"\n--- Memory Error: {error_msg} ---")

                # Clean up completed tasks
                task_id = result.get("task_id")
                if task_id and task_id in self.pending_memory_tasks:
                    del self.pending_memory_tasks[task_id]

        except Exception as e:
            print(f"\n--- Error checking memory updates: {str(e)} ---")
            import traceback

            print(traceback.format_exc())

    def _merge_facts_to_memory(self, new_facts):
        """Merge new facts into memory with deduplication."""
        if not new_facts:
            return

        added_facts = []
        updated_facts = []
        skipped_facts = []

        # For each new fact, check if it's already in memory or similar to existing facts
        for new_fact in new_facts:
            # Skip empty or very short facts
            if not new_fact or len(new_fact) < 5:
                continue

            new_fact_lower = new_fact.lower()

            # Check for duplicates or similar facts
            should_add = True
            facts_to_remove = []
            best_match = None
            best_similarity = 0

            for i, existing_fact in enumerate(self.memory["facts"]):
                existing_fact_lower = existing_fact.lower()

                # Calculate simple similarity
                similarity = self._calculate_fact_similarity(
                    new_fact_lower, existing_fact_lower
                )

                # Keep track of the best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing_fact

                # If facts are very similar or one contains the other
                if (
                    similarity > 0.7
                    or existing_fact_lower in new_fact_lower
                    or new_fact_lower in existing_fact_lower
                ):
                    # Replace the old fact if the new one is more detailed
                    if len(new_fact) >= len(existing_fact):
                        facts_to_remove.append(i)
                    else:
                        # Skip adding if the existing fact is better
                        should_add = False
                        skipped_facts.append(new_fact)
                        break

            # Remove outdated facts (in reverse order to avoid index issues)
            if should_add and facts_to_remove:
                for idx in sorted(facts_to_remove, reverse=True):
                    updated_facts.append(self.memory["facts"].pop(idx))

            # Add the new fact if it's not redundant
            if should_add:
                self.memory["facts"].append(new_fact)
                if not facts_to_remove:  # Only if it's truly new, not a replacement
                    added_facts.append(new_fact)

        return {
            "added": added_facts,
            "updated": updated_facts,
            "skipped": skipped_facts,
        }

    def _calculate_fact_similarity(self, text1, text2):
        """Calculate similarity between two text strings using the improved algorithm."""
        # Simple word overlap calculation using Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0

        # Check for one being a substring of the other
        substring_factor = 0
        if text1 in text2:
            substring_factor = len(text1) / len(text2)
        elif text2 in text1:
            substring_factor = len(text2) / len(text1)

        # Check for key phrase matches (like dates, times, names)
        key_phrases_similarity = 0
        # Look for patterns like dates, times, numbers
        patterns = [
            r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b",  # Time
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",  # Month day
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",  # Day month
            r"\b(?:tomorrow|today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",  # Day references
            r"\b\d+\s+(?:meeting|appointment|call|session)s?\b",  # Number of events
        ]

        for pattern in patterns:
            matches1 = set(re.findall(pattern, text1, re.IGNORECASE))
            matches2 = set(re.findall(pattern, text2, re.IGNORECASE))
            if matches1 and matches2:
                common_matches = matches1.intersection(matches2)
                if common_matches:
                    key_phrases_similarity = (
                        0.3  # Boost similarity if key patterns match
                    )

        # Combine the different similarity measures with weights
        combined_similarity = max(
            jaccard * 0.6 + substring_factor * 0.3 + key_phrases_similarity,
            substring_factor * 0.7,  # Give substantial weight to substring matching
        )

        return combined_similarity

    def _force_memory_sync(self):
        """Force synchronization with the memory file to ensure consistency."""
        # Request the current memory from the manager
        task_id = str(uuid.uuid4())
        self.memory_manager.task_queue.put({"type": "read", "task_id": task_id})
        self.pending_memory_tasks[task_id] = time.time()

        # Wait briefly for a response
        start_time = time.time()
        while time.time() - start_time < 1.0:  # 1 second timeout
            if not self.memory_manager.result_queue.empty():
                result = self.memory_manager.result_queue.get()
                if result.get("task_id") == task_id and "memory" in result:
                    self.memory = result["memory"]
                    return True
            time.sleep(0.1)

        return False

    # Agent Trainer Methods

    def _record_interaction(self, agent_name, query, response):
        """Record an interaction with an agent for training purposes."""
        task_id = str(uuid.uuid4())

        # Skip recording if the agent isn't in the registry (like executive_assistant)
        if agent_name not in self.registry and agent_name != EA_AGENT_NAME:
            return

        # Determine if this is a potentially problematic interaction
        is_problematic = self._check_query_for_dissatisfaction(query)

        # Update interaction count
        self.agent_interaction_counts[agent_name] += 1
        interaction_count = self.agent_interaction_counts[agent_name]

        # Only print feedback for non-problematic interactions (problematic ones already print)
        if not is_problematic and self.auto_improve_agents:
            print(
                f"\n--- Agent Trainer: Recording interaction #{interaction_count} with {agent_name} ---"
            )
            print(
                f"- Interaction length: Query ({len(query)}) chars, Response (~{len(response.split())}) words"
            )
            print(f"- This data will be used to analyze agent performance")

        # Send the interaction to the trainer
        self.agent_trainer.task_queue.put(
            {
                "type": "record_interaction",
                "agent_name": agent_name,
                "query": query,
                "response": response,
                "timestamp": time.time(),
                "task_id": task_id,
            }
        )
        self.pending_trainer_tasks[task_id] = time.time()

        # Update training status
        self.training_status[agent_name]["status"] = "recording_data"
        self.training_status[agent_name]["last_updated"] = time.time()

        # If we've recorded many interactions, trigger analysis
        if (
            interaction_count > 0
            and interaction_count % 5 == 0
            and self.auto_improve_agents
        ):
            print(
                f"\n--- Agent Trainer: Triggering periodic analysis for '{agent_name}' ---"
            )
            print(f"- {interaction_count} interactions recorded since last reset")
            self._analyze_agent_performance(agent_name)

    def _check_query_for_dissatisfaction(self, query):
        """Check if a query shows dissatisfaction with previous response."""
        # Common dissatisfaction patterns
        patterns = [
            r"that's not what i( was)? ask(ed|ing)",
            r"you didn't answer my question",
            r"that's not (correct|right|accurate)",
            r"you misunderstood",
            r"that's not helpful",
            r"can you try again",
            r"that's wrong",
            r"not what i meant",
            r"you're confused",
        ]

        query_lower = query.lower()

        for pattern in patterns:
            if re.search(pattern, query_lower):
                # Print immediate feedback in the terminal when dissatisfaction is detected
                print("\n" + "!" * 60)
                print(
                    "! AGENT TRAINER: Detected user dissatisfaction with the response"
                )
                print("! Query showing friction:", query)
                print("! Recording this interaction for agent improvement")
                print("!" * 60)
                return True

        return False

    def _analyze_agent_performance(self, agent_name):
        """Trigger analysis of an agent's performance."""
        task_id = str(uuid.uuid4())
        self.agent_trainer.task_queue.put(
            {"type": "analyze_agent", "agent_name": agent_name, "task_id": task_id}
        )
        self.pending_trainer_tasks[task_id] = time.time()
        print(f"Analyzing {agent_name}'s performance...")

        # Update training status
        self.training_status[agent_name]["status"] = "analyzing"
        self.training_status[agent_name]["last_updated"] = time.time()

    def _provide_direct_feedback(self, agent_name, feedback):
        """Send direct feedback about an agent."""
        task_id = str(uuid.uuid4())
        self.agent_trainer.task_queue.put(
            {
                "type": "direct_feedback",
                "agent_name": agent_name,
                "feedback": feedback,
                "task_id": task_id,
            }
        )
        self.pending_trainer_tasks[task_id] = time.time()

        # Update training status
        self.training_status[agent_name]["status"] = "feedback_received"
        self.training_status[agent_name]["last_updated"] = time.time()

        return f"Thank you for your feedback about {agent_name}. I'll use this to improve their responses in the future."

    def _trigger_agent_improvement(self, agent_name):
        """Trigger improvement of an agent's system prompt."""
        if agent_name not in self.registry:
            return f"Agent '{agent_name}' not found in registry"

        # Get current prompt
        current_prompt = self.registry[agent_name]["system_prompt"]

        # Send to trainer for improvement
        task_id = str(uuid.uuid4())
        self.agent_trainer.task_queue.put(
            {
                "type": "improve_prompt",
                "agent_name": agent_name,
                "current_prompt": current_prompt,
                "task_id": task_id,
            }
        )
        self.pending_trainer_tasks[task_id] = time.time()

        # Update training status
        self.training_status[agent_name]["status"] = "improving"
        self.training_status[agent_name]["last_updated"] = time.time()

        return f"Prompt improvement for agent '{agent_name}' has been triggered. This may take a moment..."

    def _apply_improved_prompt(self, agent_name, improved_prompt):
        """Apply an improved prompt to an agent."""
        if agent_name not in self.registry:
            print(
                f"Warning: Agent '{agent_name}' not found when trying to apply improved prompt"
            )
            return False

        # Print original and improved prompts for comparison
        original_prompt = self.registry[agent_name]["system_prompt"]
        print("\n==== PROMPT IMPROVEMENT DETAILS ====")
        print(f"Agent: {agent_name}")
        print(f"Original length: {len(original_prompt)} chars")
        print(f"Improved length: {len(improved_prompt)} chars")

        # Show a simple diff of key changes
        print("\nKey improvements:")

        # Find some sample differences
        orig_paragraphs = original_prompt.split("\n\n")
        new_paragraphs = improved_prompt.split("\n\n")

        # Compare number of paragraphs/sections
        print(f"- Original prompt had {len(orig_paragraphs)} sections")
        print(f"- Improved prompt has {len(new_paragraphs)} sections")

        # Look for Alpaca formatting
        if (
            "### Instruction:" in improved_prompt
            and "### Instruction:" not in original_prompt
        ):
            print("- Added standardized Alpaca-style formatting with clear sections")

        # Check for examples
        orig_examples = original_prompt.count("example")
        new_examples = improved_prompt.count("example")
        if new_examples > orig_examples:
            print(f"- Added more examples ({new_examples} vs {orig_examples})")

        # Track training status
        self.training_status[agent_name]["status"] = "improved"
        self.training_status[agent_name]["last_updated"] = time.time()
        self.training_status[agent_name]["improvement_time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Update registry with the improved prompt
        self.registry[agent_name]["system_prompt"] = improved_prompt
        self.save_registry()
        print(f"\n>> Updated agent registry file: {self.registry_file}")

        # For debugging, create a backup of the registry
        backup_file = f"{self.registry_file}.{int(time.time())}.bak"
        try:
            with open(backup_file, "w") as f:
                json.dump(self.registry, f, indent=2)
            print(f">> Created backup of registry: {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create registry backup: {e}")

        # If the agent is currently running, restart it with the new prompt
        if agent_name in self.agent_processes:
            # Get enhanced prompt in Alpaca format
            enhanced_prompt = self._generate_agent_system_prompt(
                agent_name, improved_prompt
            )

            # Shutdown old process
            try:
                self.agent_pipes[agent_name].send({"type": "shutdown"})
                self.agent_processes[agent_name].join(timeout=2)
                if self.agent_processes[agent_name].is_alive():
                    self.agent_processes[agent_name].terminate()
            except Exception as e:
                print(f"Error shutting down agent '{agent_name}': {e}")

            # Start new process with enhanced prompt
            parent_conn, child_conn = Pipe()
            agent_process = AgentProcess(
                agent_name=agent_name,
                system_prompt=enhanced_prompt,
                model_name=self.registry[agent_name].get("model", DEFAULT_MODEL),
                pipe=child_conn,
            )
            self.agent_processes[agent_name] = agent_process
            self.agent_pipes[agent_name] = parent_conn
            agent_process.start()

            print(f">> Agent '{agent_name}' restarted with improved prompt")

            # Reset interaction count for this agent
            self.agent_interaction_counts[agent_name] = 0

            return True

        return False

    def check_trainer_updates(self):
        """Check for and process updates from the agent trainer."""
        # Check if it's time to check for updates
        now = time.time()
        if now - self.last_trainer_check < self.trainer_check_interval:
            return

        self.last_trainer_check = now

        # Non-blocking check for trainer updates
        try:
            if self.agent_trainer.result_queue.empty():
                return

            while not self.agent_trainer.result_queue.empty():
                result = self.agent_trainer.result_queue.get_nowait()

                # Process different result types
                result_type = result.get("type", "")

                if (
                    result_type == "analyze_agent_result"
                    and result.get("status") == "success"
                ):
                    # Handle agent analysis results
                    analysis = result.get("analysis", {})

                    if analysis.get("needs_improvement", False):
                        agent_name = analysis.get("agent_name")
                        print(f"\n{'='*60}")
                        print(
                            f"=== AGENT TRAINER: Analysis Results for '{agent_name}' ==="
                        )
                        print(
                            f"= Dissatisfaction detected in {analysis.get('dissatisfaction_count')} out of {analysis.get('total_interactions')} interactions"
                        )
                        print(
                            f"= Problem percentage: {analysis.get('problem_percentage')}%"
                        )

                        # Patterns detected
                        if "patterns" in analysis and analysis["patterns"]:
                            print("= Identified Problem Patterns:")
                            for pattern in analysis["patterns"]:
                                print(f"=   - {pattern}")

                        print(f"= CONCLUSION: Agent '{agent_name}' needs improvement")
                        print(f"{'='*60}")

                        # Trigger automatic improvement if enabled
                        if self.auto_improve_agents:
                            print(
                                f"\n>> Automatically triggering improvement for agent '{agent_name}'..."
                            )
                            self._trigger_agent_improvement(agent_name)
                    else:
                        agent_name = analysis.get("agent_name", "unknown")
                        print(
                            f"\n--- Agent Trainer: '{agent_name}' is performing adequately ---"
                        )
                        print(
                            f"- Only {analysis.get('dissatisfaction_count', 0)} out of {analysis.get('total_interactions', 0)} interactions showed signs of dissatisfaction"
                        )
                        print("- No improvements needed at this time")
                        print("-------------------------------------------")

                elif (
                    result_type == "improve_prompt_result"
                    and result.get("status") == "success"
                ):
                    # Handle prompt improvement results
                    agent_name = result.get("agent_name")
                    improved_prompt = result.get("improved_prompt")

                    if agent_name and improved_prompt:
                        # Apply the improved prompt
                        success = self._apply_improved_prompt(
                            agent_name, improved_prompt
                        )

                        if success:
                            # Show what was improved
                            notes = result.get("notes", [])
                            note_texts = [n.get("note", "") for n in notes]

                            print(f"\n{'*'*60}")
                            print(
                                f"*** AGENT TRAINER: '{agent_name}' Has Been Improved ***"
                            )
                            print("* Improvements based on the following feedback:")
                            for i, note in enumerate(note_texts, 1):
                                print(f"*  {i}. {note}")

                            print("*\n* The agent's prompt has been enhanced with:")
                            # Show key differences or improvements
                            print("*  - More specific instructions and context")
                            print("*  - Better handling of user queries")
                            print("*  - Improved response guidelines")

                            print(f"* Updated prompt saved to {self.registry_file}")
                            print(
                                f"* Agent has been restarted with the improved system prompt"
                            )
                            print(f"{'*'*60}")
                elif (
                    result_type == "direct_feedback_result"
                    and result.get("status") == "success"
                ):
                    # Show confirmation of feedback processing
                    agent_name = result.get("agent_name", "unknown")
                    improvement_type = result.get("improvement_type", "unknown")

                    print(
                        f"\n--- Agent Trainer: Feedback for '{agent_name}' recorded ---"
                    )
                    print(f"- Improvement area identified: {improvement_type}")
                    print(
                        f"- This feedback will be used for future prompt improvements"
                    )
                    print("-------------------------------------------")

                # Clean up completed tasks
                task_id = result.get("task_id")
                if task_id and task_id in self.pending_trainer_tasks:
                    del self.pending_trainer_tasks[task_id]

        except Exception as e:
            print(f"\n--- Error checking trainer updates: {str(e)} ---")
            import traceback

            print(traceback.format_exc())


if __name__ == "__main__":
    print("Multi-Process Llama 3.1 AI Office")
    print("=" * 40)
    print("Enter your prompt below (or 'quit' to exit)")
    print("Agent commands:")
    print("  /list_agents - Show all available agents")
    print('  /add_agent <name> "<system prompt>" - Create a new specialized agent')
    print("  /delete_agent <name> - Remove an agent")
    print("  /view_agent <name> - View an agent's system prompt")
    print(
        '  /edit_agent <name> "<new system prompt>" - Update an agent\'s system prompt'
    )
    print("  /memory - Show all facts stored in memory")
    print("  /memory <query> - Show facts matching a specific query")
    print("  /clear_memory - Delete all stored memories")
    print("  /deduplicate_memory - Remove redundant facts")
    print(
        '  /feedback <agent_name> "your feedback" - Provide direct feedback about an agent'
    )
    print("  /toggle_auto_improve - Enable/disable automatic agent improvement")
    print("  /analyze_agent <agent_name> - Analyze an agent's performance")
    print("  /improve_agent <agent_name> - Trigger improvement for an agent")

    # Create the executive assistant
    office = MultiAgentOffice()

    try:
        # Print available agents
        print("\nAvailable Agents:")
        for agent_name in office.registry.keys():
            print(f"- {agent_name}")

        print("\nConversation Session Commands:")
        print("- /who - Shows which agent you're currently talking to")
        print(
            "- /switch_to <agent> - Explicitly switch to talking with a specific agent"
        )
        print("- /ea - Switch back to talking with the Executive Assistant")
        print(
            "Note: The system will automatically maintain conversation continuity with agents"
        )

        while True:
            user_prompt = input("\n> ")

            if user_prompt.lower() in ["quit", "exit", "q"]:
                break

            # Process the user's query
            # For non-command queries, responses are streamed directly by the handler methods
            # For command queries, we need to print the response here
            response = office.process_query(user_prompt)

            # Only print responses for special commands (which don't stream)
            # Or if we received a non-empty response from a regular query
            if user_prompt.startswith("/"):
                print(response)
            elif response:  # Only print non-empty responses for regular queries
                print(response)

    finally:
        # Ensure we shut down all agent processes when exiting
        print("\nShutting down agent processes...")
        office.shutdown_agent_processes()
        print("Goodbye!")
