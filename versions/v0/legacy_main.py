#!/usr/bin/env python3

"""
AI Office - Main Entry Point
This script can either run the original single-process AI Office or the new multi-process version.
"""

import os
import sys
import requests
import json
import re
import time
import datetime

# Model configuration
MODEL_NAME = "llama3.1:latest"
# Use a smaller, faster model for agent selection
SELECTOR_MODEL = "tinyllama"  # A much smaller model for quick classification
API_URL = "http://localhost:11434/api/generate"


# Memory System
class MemorySystem:
    def __init__(self, memory_file="office_memory.json"):
        self.memory_file = memory_file
        self.memories = self.load_memories()

    def load_memories(self):
        """Load memories from the file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading memories: {e}")
                return {"facts": [], "conversation_history": []}
        else:
            return {"facts": [], "conversation_history": []}

    def save_memories(self):
        """Save memories to a file."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            print(f"Error saving memories: {e}")

    def add_conversation(self, user_input, agent_name, response):
        """Add a conversation to the history."""
        timestamp = datetime.datetime.now().isoformat()
        self.memories["conversation_history"].append(
            {
                "timestamp": timestamp,
                "user_input": user_input,
                "agent": agent_name,
                "response": (
                    response[:200] + "..." if len(response) > 200 else response
                ),  # Truncate long responses
            }
        )

        # Keep only the last 50 conversations
        if len(self.memories["conversation_history"]) > 50:
            self.memories["conversation_history"] = self.memories[
                "conversation_history"
            ][-50:]

        self.save_memories()

    def extract_facts(self, user_input, response):
        """Extract key facts from a conversation using Llama."""
        # Only extract facts for substantive conversations
        if len(user_input) < 10:  # Skip short inputs
            return []

        # Create a prompt to extract facts
        extraction_prompt = f"""
From the following conversation, extract 0-3 key factual pieces of information that would be useful to remember for future reference.
Only extract specific, concrete facts mentioned by the user or established in the conversation.
Focus on personal preferences, important dates, key information, or specific requirements mentioned.

Examples of good facts to extract:
- "User has three meetings scheduled for tomorrow"
- "User is allergic to peanuts"
- "User's project deadline is March 15th"
- "User prefers vegetarian meals for dinner"

Do not include opinions, general knowledge, common information, or placeholder text like "fact 1" or "fact 2".
If no clear facts are present, return an empty list.

User: {user_input}

Response: {response}

Extract only clear, specific factual information that would be useful to remember about the user or their situation.
Format the output as valid JSON like this: ["fact 1", "fact 2"]
"""

        try:
            # Try to use the smaller model for extraction if available
            model_to_use = (
                MODEL_NAME  # Always use the main model for accurate fact extraction
            )

            payload = {
                "model": model_to_use,
                "prompt": extraction_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "max_tokens": 300},
                "system": "You are a fact extraction assistant. Extract only clear factual information that would be useful to remember about the user. If no clear facts are present, respond with an empty JSON array [].",
            }

            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            result = ""
            for line in response.text.splitlines():
                if not line:
                    continue

                data = json.loads(line)
                if "response" in data:
                    result += data["response"]

                if data.get("done", False):
                    break

            # Parse the response as JSON
            try:
                # Find JSON list in the response using regex
                match = re.search(r"\[(.*?)\]", result, re.DOTALL)
                if match:
                    json_str = f"[{match.group(1)}]"
                    facts = json.loads(json_str)

                    # Validate facts - ensure they're not just placeholders
                    valid_facts = []
                    for fact in facts:
                        if fact and len(fact) > 5 and "fact" not in fact.lower():
                            # Only add if it seems like a real fact
                            valid_facts.append(fact)

                    # Add new facts to memory
                    if valid_facts:
                        self.memories["facts"].extend(valid_facts)
                        print(f"Added {len(valid_facts)} new facts to memory")
                        self.save_memories()
                    return valid_facts
            except Exception as e:
                print(f"Error parsing extracted facts: {e}")

            return []

        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []

    def get_relevant_context(self, query, max_facts=5):
        """Get relevant facts and conversation history for a query."""
        if not self.memories["facts"]:
            return ""

        # Simple keyword matching for now - could be improved with embeddings
        query_terms = set(query.lower().split())
        scored_facts = []

        for fact in self.memories["facts"]:
            fact_terms = set(fact.lower().split())
            # Calculate word overlap
            overlap = len(query_terms.intersection(fact_terms))
            if overlap > 0:
                scored_facts.append((fact, overlap))

        # Sort by relevance score
        scored_facts.sort(key=lambda x: x[1], reverse=True)

        # Get top facts
        top_facts = [fact for fact, score in scored_facts[:max_facts]]

        if not top_facts:
            return ""

        context = "Based on previous conversations, I know the following facts that may be relevant:\n"
        for fact in top_facts:
            context += f"- {fact}\n"

        return context


def run_llama(prompt, system_prompt=None, temperature=0.7, max_tokens=1000):
    """
    Run the Llama 3.1 model with the provided prompt and stream the output.

    Args:
        prompt (str): The prompt to send to the model
        system_prompt (str, optional): Optional system prompt to guide the model
        temperature (float): Controls randomness (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate
    """
    # Prepare the payload
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "max_tokens": max_tokens},
    }

    if system_prompt:
        payload["system"] = system_prompt

    try:
        # Check if Llama 3.1 is available
        check_model()

        # Make the API request
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line)
            if "response" in data:
                chunk = data["response"]
                full_response += chunk
                print(chunk, end="", flush=True)

            # Check if this is the final response
            if data.get("done", False):
                print()
                print("-" * 40)
                break

        return full_response

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return ""


def check_model():
    """Check if Llama 3.1 is available and pull it if necessary."""
    try:
        # Check available models
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()

        data = response.json()
        models = [model["name"] for model in data.get("models", [])]

        if MODEL_NAME not in models:
            print(f"{MODEL_NAME} not found in your Ollama installation.")
            print(f"Please run: ollama pull {MODEL_NAME.split(':')[0]}")
            print("Then try running this script again.")
            exit(1)

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        print("Make sure Ollama is installed and running.")
        print("You can download it from: https://ollama.ai/download")
        exit(1)


def check_selector_model():
    """Check if the selector model is available and suggest pulling it if not."""
    try:
        # Check available models
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()

        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        selector_with_tag = f"{SELECTOR_MODEL}:latest"

        if selector_with_tag not in models and SELECTOR_MODEL not in models:
            print(f"Notice: Faster selector model '{SELECTOR_MODEL}' not found.")
            print(f"You can install it with: /pull_selector")
            print(f"Using {MODEL_NAME} for agent selection until then.")
            return False
        return True

    except requests.exceptions.RequestException:
        # If error, just silently return False
        return False


def pull_selector_model():
    """Pull the selector model for faster agent selection."""
    try:
        print(f"Pulling {SELECTOR_MODEL} model for faster agent selection...")

        # Execute the Ollama pull command through the API
        response = requests.post(
            "http://localhost:11434/api/pull", json={"name": SELECTOR_MODEL}
        )

        # Stream the download progress
        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line)
            if "status" in data:
                status = data["status"]
                if "progress" in data:
                    print(f"\r{status} {data['progress']:.1f}%", end="", flush=True)
                else:
                    print(f"\r{status}", end="", flush=True)

            if data.get("completed", False):
                print("\nDownload completed!")
                break

        return True
    except Exception as e:
        print(f"Error pulling selector model: {e}")
        return False


# Agent registry system
class AgentRegistry:
    def __init__(self, registry_file="agent_registry.json"):
        self.agents = {}
        self.registry_file = registry_file
        self.load_registry()

    def register_agent(self, name, system_prompt, model=MODEL_NAME, temperature=0.7):
        self.agents[name] = {
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature,
            "status": "idle",
        }
        self.save_registry()

    def get_agent(self, name):
        return self.agents.get(name)

    def list_available_agents(self):
        return [
            name for name, details in self.agents.items() if details["status"] == "idle"
        ]

    def save_registry(self):
        """Save the registry to a file."""
        try:
            with open(self.registry_file, "w") as f:
                json.dump(self.agents, f, indent=2)
            print(f"Registry saved to {self.registry_file}")
        except Exception as e:
            print(f"Error saving registry: {e}")

    def load_registry(self):
        """Load the registry from a file if it exists."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    self.agents = json.load(f)
                print(f"Loaded {len(self.agents)} agents from registry")
            except Exception as e:
                print(f"Error loading registry: {e}")
                # Initialize with empty registry if loading fails
                self.agents = {}
        else:
            print("No registry file found. Starting with empty registry.")

    def delete_agent(self, name):
        """Delete an agent from the registry."""
        if name in self.agents:
            del self.agents[name]
            self.save_registry()
            return True
        return False


# Executive Assistant class
class ExecutiveAssistant:
    def __init__(
        self, registry_file="agent_registry.json", memory_file="office_memory.json"
    ):
        self.registry = AgentRegistry(registry_file)
        self.memory = MemorySystem(memory_file)

        # Only setup default agents if the registry is empty
        if not self.registry.agents:
            self.setup_default_agents()

    def setup_default_agents(self):
        # Register some default specialized agents
        self.registry.register_agent(
            "calendar_manager",
            "You are a calendar management specialist. Your job is to organize schedules, create agendas, and manage time efficiently. You help with scheduling meetings, tracking appointments, and planning events. When asked about calendar-related tasks, provide detailed and organized responses.",
        )
        self.registry.register_agent(
            "research_assistant",
            "You are a research specialist. Your job is to find information, summarize content, and provide well-researched answers. You excel at answering factual questions, synthesizing complex topics, and providing balanced perspectives on various subjects.",
        )
        self.registry.register_agent(
            "creative_writer",
            "You are a creative writing specialist. Your job is to generate engaging, original content including stories, poems, scripts, and marketing copy. You can adapt your writing style to different tones and formats as requested.",
        )
        self.registry.register_agent(
            "technical_expert",
            "You are a technical expert specialized in software, programming, and technology. You provide detailed explanations of technical concepts, help debug code issues, and offer guidance on best practices in software development and IT.",
        )
        self.registry.register_agent(
            "personal_assistant",
            "You are a personal life assistant. You help with personal tasks, offer advice on daily challenges, suggest self-improvement strategies, and provide thoughtful recommendations for health, wellness, and lifestyle choices.",
        )

    def delegate_task(self, task_description):
        """Intelligently delegate a task to the most appropriate agent."""
        # If no agents are available, handle it directly
        available_agents = self.registry.list_available_agents()
        if not available_agents:
            response = self.handle_with_ea(task_description)
            self.memory.add_conversation(
                task_description, "executive_assistant", response
            )
            self.memory.extract_facts(task_description, response)
            return response

        # First try quick keyword matching for common tasks
        start_time = time.time()
        quick_match = self.quick_agent_match(task_description, available_agents)
        if quick_match:
            print(f"Quick agent match found in {time.time() - start_time:.2f}s")
            response = self.run_agent_task(quick_match, task_description)
            self.memory.add_conversation(task_description, quick_match, response)
            self.memory.extract_facts(task_description, response)
            return response

        # Use a smaller model to decide which agent is best suited for this task
        best_agent = self.determine_best_agent(task_description, available_agents)

        if best_agent:
            response = self.run_agent_task(best_agent, task_description)
            self.memory.add_conversation(task_description, best_agent, response)
            self.memory.extract_facts(task_description, response)
            return response
        else:
            # If no suitable agent was found, handle it directly
            response = self.handle_with_ea(task_description)
            self.memory.add_conversation(
                task_description, "executive_assistant", response
            )
            self.memory.extract_facts(task_description, response)
            return response

    def quick_agent_match(self, task, available_agents):
        """Use keyword matching for fast agent selection for common tasks."""
        task_lower = task.lower()

        # Define keyword patterns for common agent tasks
        agent_keywords = {
            "calendar_manager": [
                "calendar",
                "schedule",
                "meeting",
                "appointment",
                "remind",
                "agenda",
                "event",
                "time",
            ],
            "research_assistant": [
                "research",
                "find",
                "information",
                "article",
                "search",
                "learn about",
                "tell me about",
            ],
            "creative_writer": [
                "write",
                "story",
                "poem",
                "creative",
                "fiction",
                "script",
                "content",
                "generate",
            ],
            "technical_expert": [
                "code",
                "program",
                "technical",
                "software",
                "computer",
                "programming",
                "bug",
                "debug",
            ],
            "personal_assistant": [
                "advice",
                "help me with",
                "personal",
                "life",
                "recommend",
                "suggestion",
                "plan",
            ],
        }

        # Only consider agents that are available
        for agent, keywords in agent_keywords.items():
            if agent in available_agents and any(
                keyword in task_lower for keyword in keywords
            ):
                return agent

        return None  # No quick match found

    def determine_best_agent(self, task, available_agents):
        """Use a smaller model to determine which agent is best suited for a task."""
        if not available_agents:
            return None

        # Create a description of each available agent
        agent_descriptions = []
        for name in available_agents:
            agent = self.registry.get_agent(name)
            if agent:
                prompt = agent.get("system_prompt", "No description available")
                agent_descriptions.append(f"Agent '{name}': {prompt}")

        # Create a prompt for the model to determine the best agent
        selection_prompt = f"""Task: "{task}"

Available agents:
{chr(10).join(agent_descriptions)}

Which single agent would be best suited to handle this task? Reply with ONLY the agent name.
"""

        # Use a smaller model for agent selection
        try:
            start_time = time.time()
            print("Selecting best agent...")

            # Try to use a smaller, faster model if available
            try:
                model_to_use = SELECTOR_MODEL
                # Check if the model exists
                response = requests.get("http://localhost:11434/api/tags")
                models = [model["name"] for model in response.json().get("models", [])]
                if (
                    f"{SELECTOR_MODEL}:latest" not in models
                    and SELECTOR_MODEL not in models
                ):
                    print(
                        f"Selection model {SELECTOR_MODEL} not found, using {MODEL_NAME} instead"
                    )
                    model_to_use = MODEL_NAME
            except Exception:
                # If any error, fall back to the main model
                model_to_use = MODEL_NAME

            payload = {
                "model": model_to_use,
                "prompt": selection_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 20,
                },  # Reduced tokens, we just need the name
                "system": "You are a task classifier. Respond with ONLY the name of the most appropriate agent for the task.",
            }

            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            result = ""
            for line in response.text.splitlines():
                if not line:
                    continue

                data = json.loads(line)
                if "response" in data:
                    result += data["response"]

                if data.get("done", False):
                    break

            # Clean and normalize the result
            result = result.strip().lower()

            # Clean up the result to extract just the agent name by removing punctuation and quotes
            cleaned_result = re.sub(r"[^\w\s]", "", result).strip()

            # Try to find a valid agent name in the result
            for agent_name in available_agents:
                if agent_name.lower() in cleaned_result:
                    print(
                        f"Selected agent: {agent_name} in {time.time() - start_time:.2f}s"
                    )
                    return agent_name

            # Check if "none" is in the response
            if "none" in cleaned_result:
                print(f"No suitable agent found in {time.time() - start_time:.2f}s")
                return None

            # If no match was found, try to use the first word as the agent name
            first_word = cleaned_result.split()[0] if cleaned_result.split() else None
            if first_word and first_word in available_agents:
                print(
                    f"Selected agent: {first_word} in {time.time() - start_time:.2f}s"
                )
                return first_word

            # If we still can't determine the agent, handle with EA
            print(
                f"Agent selection failed in {time.time() - start_time:.2f}s: '{result}'"
            )
            return None

        except Exception as e:
            print(f"Error selecting agent: {e}")
            # Default to the first available agent if there's an error
            return available_agents[0] if available_agents else None

    def run_agent_task(self, agent_name, task):
        agent = self.registry.get_agent(agent_name)
        if agent:
            print(f"Delegating to {agent_name}...")

            # Get relevant context from memory
            context = self.memory.get_relevant_context(task)
            if context:
                print("Adding relevant context from memory")

                # Add context to the prompt
                enhanced_task = f"{context}\n\nUser request: {task}"

                # Enhance the system prompt
                system_prompt = agent.get("system_prompt", "")
                enhanced_system = f"{system_prompt}\n\nUse the provided context when it's relevant to the user's request. Don't explicitly mention that you're using previous conversation data unless directly asked about your memory."

                return run_llama(enhanced_task, system_prompt=enhanced_system)
            else:
                return run_llama(task, system_prompt=agent["system_prompt"])
        return "Agent not found."

    def handle_with_ea(self, task_description):
        """Handle a task directly with the executive assistant."""
        return run_llama(
            task_description,
            system_prompt="You are an executive assistant that manages other specialized AI agents.",
        )

    def show_memory(self, query=None, max_facts=10):
        """Display stored memories/facts, optionally filtered by a query."""
        facts = self.memory.memories["facts"]
        if not facts:
            return "No facts have been stored in memory yet."

        if query:
            # Filter facts by query terms
            query_terms = set(query.lower().split())
            matching_facts = []

            for fact in facts:
                fact_terms = set(fact.lower().split())
                # Check for any overlap
                if query_terms.intersection(fact_terms):
                    matching_facts.append(fact)

            if not matching_facts:
                return f"No facts matching '{query}' found in memory."

            result = f"Found {len(matching_facts)} facts matching '{query}':\n\n"
            for i, fact in enumerate(matching_facts[:max_facts], 1):
                result += f"{i}. {fact}\n"

            if len(matching_facts) > max_facts:
                result += f"\n...and {len(matching_facts) - max_facts} more facts."

            return result
        else:
            # Show all facts
            result = f"There are {len(facts)} stored facts in memory:\n\n"
            for i, fact in enumerate(facts[:max_facts], 1):
                result += f"{i}. {fact}\n"

            if len(facts) > max_facts:
                result += f"\n...and {len(facts) - max_facts} more facts."

            return result

    def clear_memory(self):
        """Clear all stored memories."""
        self.memory.memories = {"facts": [], "conversation_history": []}
        self.memory.save_memories()
        return "Memory has been cleared."


def main():
    print(f"Llama 3.1 AI Office")
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
    print("  /pull_selector - Pull the smaller model for faster agent selection")
    print("  /memory - Show all facts stored in memory")
    print("  /memory <query> - Show facts matching a specific query")
    print("  /clear_memory - Delete all stored memories")

    # Create the executive assistant
    ea = ExecutiveAssistant()

    # Check if selector model is available
    check_selector_model()

    # Print available agents
    print("\nAvailable Agents:")
    for agent_name in ea.registry.list_available_agents():
        print(f"- {agent_name}")

    while True:
        user_prompt = input("\n> ")

        if user_prompt.lower() in ["quit", "exit", "q"]:
            break

        # Check for agent management commands
        if user_prompt.startswith("/add_agent"):
            # Format: /add_agent name "system prompt"
            parts = user_prompt.split(" ", 2)
            if len(parts) >= 3:
                agent_name = parts[1]
                system_prompt = parts[2].strip('"')
                ea.registry.register_agent(agent_name, system_prompt)
                print(f"Agent '{agent_name}' added to registry")
            else:
                print('Invalid format. Use: /add_agent name "system prompt"')
        elif user_prompt.startswith("/list_agents"):
            print("\nAvailable Agents:")
            for agent_name in ea.registry.list_available_agents():
                print(f"- {agent_name}")
        elif user_prompt.startswith("/delete_agent"):
            # Format: /delete_agent name
            parts = user_prompt.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                if ea.registry.delete_agent(agent_name):
                    print(f"Agent '{agent_name}' deleted from registry")
                else:
                    print(f"Agent '{agent_name}' not found")
            else:
                print("Invalid format. Use: /delete_agent name")
        elif user_prompt.startswith("/view_agent"):
            # Format: /view_agent name
            parts = user_prompt.split(" ", 1)
            if len(parts) >= 2:
                agent_name = parts[1]
                agent = ea.registry.get_agent(agent_name)
                if agent:
                    print(f"\nAgent: {agent_name}")
                    print("-" * 40)
                    print(
                        f"System Prompt: {agent.get('system_prompt', 'No system prompt')}"
                    )
                    print("-" * 40)
                else:
                    print(f"Agent '{agent_name}' not found")
            else:
                print("Invalid format. Use: /view_agent name")
        elif user_prompt.startswith("/edit_agent"):
            # Format: /edit_agent name "new system prompt"
            parts = user_prompt.split(" ", 2)
            if len(parts) >= 3:
                agent_name = parts[1]
                new_system_prompt = parts[2].strip('"')
                agent = ea.registry.get_agent(agent_name)
                if agent:
                    agent["system_prompt"] = new_system_prompt
                    ea.registry.save_registry()
                    print(f"Updated system prompt for agent '{agent_name}'")
                else:
                    print(f"Agent '{agent_name}' not found")
            else:
                print('Invalid format. Use: /edit_agent name "new system prompt"')
        elif user_prompt.startswith("/pull_selector"):
            # Pull the smaller model for faster agent selection
            pull_selector_model()
        elif user_prompt.startswith("/memory"):
            # Format: /memory [query]
            parts = user_prompt.split(" ", 1)
            query = parts[1] if len(parts) > 1 else None
            print(ea.show_memory(query))
        elif user_prompt.startswith("/clear_memory"):
            print(ea.clear_memory())
        else:
            # Try to delegate the task to a specialized agent, or handle it with the default assistant
            ea.delegate_task(user_prompt)


if __name__ == "__main__":
    main()
