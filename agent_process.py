#!/usr/bin/env python3

import os
import sys
import json
import time
import requests
from multiprocessing import Process, Pipe
import datetime
import re

API_URL = "http://localhost:11434/api/generate"


class AgentProcess(Process):
    def __init__(
        self, agent_name, system_prompt, model_name="llama3.1:latest", pipe=None
    ):
        super().__init__()
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.pipe = pipe
        self.local_memory = []  # Agent's private memory
        self.daemon = True  # Process will exit when main process exits

    def run(self):
        """Main process loop that waits for and handles incoming requests."""
        print(f"Agent '{self.agent_name}' started in process {os.getpid()}")

        while True:
            try:
                if self.pipe.poll(timeout=None):  # Wait for messages
                    message = self.pipe.recv()

                    if message["type"] == "request":
                        # Process the request
                        query = message["query"]
                        context = message.get("context", [])

                        # Log the request
                        print(
                            f"[Agent: {self.agent_name}] Received request: {query[:50]}..."
                        )

                        # Process query and get response
                        response = self.process_query(query, context)

                        # Update local memory
                        self.update_local_memory(query, response)

                        # Extract facts to share with EA
                        extracted_facts = self.extract_facts(query, response)

                        # Send final response back to EA with facts
                        self.pipe.send(
                            {
                                "type": "complete",
                                "agent": self.agent_name,
                                "response": response,
                                "extracted_facts": extracted_facts,
                            }
                        )

                    elif message["type"] == "shutdown":
                        print(f"Agent '{self.agent_name}' shutting down...")
                        break
            except EOFError:
                # Parent process closed the pipe
                print(f"Agent '{self.agent_name}' pipe closed. Shutting down...")
                break
            except Exception as e:
                print(f"Error in agent '{self.agent_name}': {e}")
                # Send error back to EA
                self.pipe.send(
                    {"type": "error", "agent": self.agent_name, "error": str(e)}
                )

    def process_query(self, query, context):
        """Process a query using the agent's model and system prompt."""
        # Combine context with query
        if context:
            context_str = "Based on our previous conversations, I know the following facts that may be relevant:\n"
            for fact in context:
                context_str += f"- {fact}\n"
            full_prompt = f"{context_str}\n\nUser request: {query}"
        else:
            full_prompt = query

        # Create the LLM payload
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True,  # Changed to True for streaming
            "options": {"temperature": 0.7, "max_tokens": 1000},
        }

        # Add system prompt if available
        enhanced_system = self.system_prompt
        if context:
            enhanced_system += "\n\nUse the provided context when it's relevant to the user's request. Don't explicitly mention that you're using previous conversation data unless directly asked about your memory."

        payload["system"] = enhanced_system

        try:
            # Call the LLM API with streaming
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

                    # Stream the chunk back to the main process
                    self.pipe.send(
                        {"type": "chunk", "agent": self.agent_name, "chunk": chunk}
                    )

                if data.get("done", False):
                    break

            return full_response
        except Exception as e:
            print(f"Error querying LLM: {e}")
            error_msg = f"Error processing your request: {e}"

            # Send the error as a chunk
            self.pipe.send(
                {"type": "chunk", "agent": self.agent_name, "chunk": error_msg}
            )

            return error_msg

    def update_local_memory(self, query, response):
        """Update the agent's local memory with new information from the conversation."""
        # Extract key information from the conversation
        timestamp = datetime.datetime.now().isoformat()

        # Store the conversation
        self.local_memory.append(
            {
                "timestamp": timestamp,
                "query": query,
                "response_summary": (
                    response[:100] + "..." if len(response) > 100 else response
                ),
            }
        )

        # Limit memory size
        if len(self.local_memory) > 20:
            self.local_memory = self.local_memory[-20:]

    def extract_facts(self, query, response):
        """Extract facts from the conversation to share with the EA."""
        # Skip extraction for short queries
        if len(query) < 10:
            return []

        # Create a prompt to extract facts
        extraction_prompt = f"""
From the following conversation, extract 0-3 key factual pieces of information that would be useful to remember.
Focus only on specific, concrete facts mentioned by the user or established in the conversation.

User: {query}

Response: {response}

Extract only clear, specific factual information that would be useful to remember.
Format your response as a JSON list: ["fact 1", "fact 2"]
"""

        # Create extraction payload
        payload = {
            "model": self.model_name,
            "prompt": extraction_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "max_tokens": 300},
            "system": "You are a fact extraction assistant. Extract only clear factual information that would be useful to remember. Respond with a JSON array of facts.",
        }

        try:
            # Call the LLM API
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            # Parse the response
            result = ""
            for line in response.text.splitlines():
                if not line:
                    continue

                data = json.loads(line)
                if "response" in data:
                    result += data["response"]

                if data.get("done", False):
                    break

            # Extract JSON array
            match = re.search(r"\[(.*?)\]", result, re.DOTALL)
            if match:
                json_str = f"[{match.group(1)}]"
                facts = json.loads(json_str)

                # Filter valid facts
                valid_facts = []
                for fact in facts:
                    if fact and len(fact) > 5 and "fact" not in fact.lower():
                        valid_facts.append(fact)

                return valid_facts

            return []
        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []


if __name__ == "__main__":
    # This file shouldn't be run directly, but we can add a test
    print("This module is designed to be imported, not run directly.")
