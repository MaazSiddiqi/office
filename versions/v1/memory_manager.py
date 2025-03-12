#!/usr/bin/env python3

import os
import json
import requests
import time
import datetime
import re
import signal
import fcntl
from multiprocessing import Process, Queue
import traceback
import hashlib

# Configuration
MODEL_NAME = "llama3.1:latest"
API_URL = "http://localhost:11434/api/generate"
DEFAULT_MEMORY_FILE = "office_memory.json"


class MemoryManagerProcess(Process):
    """
    A separate process that handles memory operations asynchronously.
    This class is responsible for:
    1. Extracting facts from conversations
    2. Storing facts in the memory file
    3. Handling read requests for memory
    4. Managing file locks to prevent race conditions
    """

    def __init__(self, memory_file=DEFAULT_MEMORY_FILE, model_name=MODEL_NAME):
        super().__init__()
        self.memory_file = memory_file
        self.model_name = model_name
        self.task_queue = Queue()  # Queue for incoming tasks
        self.result_queue = Queue()  # Queue for sending results back
        self.should_exit = False
        self.daemon = True  # Process will exit when main process exits

        # Ensure memory file exists
        if not os.path.exists(memory_file):
            with open(memory_file, "w") as f:
                json.dump({"facts": [], "conversation_history": []}, f)

    def run(self):
        """Main process loop that handles memory operations."""
        print(f"Memory Manager started in process {os.getpid()}")

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        while not self.should_exit:
            try:
                # Get task from queue (wait up to 1 second)
                try:
                    if self.task_queue.empty():
                        time.sleep(0.1)  # Short sleep to reduce CPU usage
                        continue

                    task = self.task_queue.get(timeout=1)
                except Exception:
                    # No task available
                    continue

                task_type = task.get("type")

                # Process different task types
                if task_type == "extract":
                    # Extract facts from a conversation
                    self._handle_extract_task(task)
                elif task_type == "read":
                    # Read the entire memory file
                    self._handle_read_task(task)
                elif task_type == "query":
                    # Query memory for relevant facts
                    self._handle_query_task(task)
                elif task_type == "clear":
                    # Clear memory
                    self._handle_clear_task(task)
                elif task_type == "add_history":
                    # Add history entry directly
                    self._handle_add_history_task(task)
                elif task_type == "shutdown":
                    print("Memory Manager shutting down...")
                    self.should_exit = True
                else:
                    print(f"Unknown task type: {task_type}")

            except Exception as e:
                print(f"Error in Memory Manager: {e}")
                import traceback
                print(traceback.format_exc())

    def _handle_signal(self, signum, frame):
        """Handle termination signals gracefully."""
        print(f"Memory Manager received signal {signum}, shutting down...")
        self.should_exit = True

    def _handle_extract_task(self, task):
        """Handle extraction of facts from conversation."""
        query = task.get("query", "")
        response = task.get("response", "")
        agent = task.get("agent", "unknown")
        task_id = task.get("task_id")

        # Check if facts are already provided (from agent)
        pre_extracted_facts = task.get("facts", None)

        if pre_extracted_facts:
            facts = pre_extracted_facts
            print(f"[Memory] Using {len(facts)} pre-extracted facts from {agent}")
        else:
            # Check for direct contact information first - this is high priority info
            # that should always be extracted when present
            contact_facts = self._extract_contact_information(query, response)
            if contact_facts:
                facts = contact_facts
                print(f"[Memory] Extracted {len(facts)} contact information facts")
            else:
                # Only skip for very short interactions (reduced length threshold)
                if len(query) < 5 and len(response) < 10:
                    print(f"[Memory] Skipping extraction for very short interaction: {query}")
                    self.result_queue.put(
                        {
                            "operation": "extract_facts",
                            "status": "success",
                            "facts": [],
                            "task_id": task_id,
                        }
                    )
                    return

                # Skip extraction only for the most basic greetings
                common_trivial_phrases = [
                    "hello",
                    "hi",
                    "thanks",
                    "thank you",
                    "bye",
                ]

                query_lower = query.lower()
                if query_lower in common_trivial_phrases and len(query) < 10:
                    print(f"[Memory] Skipping extraction for basic greeting: {query}")
                    self.result_queue.put(
                        {
                            "operation": "extract_facts",
                            "status": "success",
                            "facts": [],
                            "task_id": task_id,
                        }
                    )
                    return

                # Extract facts using LLM
                facts = self._extract_facts(query, response)

        # Add debug prints
        print(f"[Memory] Pre-update facts count: {len(facts) if facts else 0}")

        # Update memory file with extracted facts
        if facts:
            print(f"[Memory] Extracted {len(facts)} facts")
            try:
                # Update memory and get list of facts that were actually added or updated
                effective_changes = self._update_memory_file(facts, query, response, agent)

                # Debug info
                print(f"[Memory] Effective changes count: {len(effective_changes)}")
                for i, fact in enumerate(effective_changes, 1):
                    print(f"[Memory] Added new fact: {fact}")

                # For better feedback, format facts nicely
                formatted_facts = []
                for i, fact in enumerate(effective_changes, 1):
                    formatted_facts.append(f"  {i}. {fact}")

                # Include formatted facts in result
                self.result_queue.put(
                    {
                        "operation": "extract_facts",
                        "status": "success",
                        "facts": effective_changes,  # Only return facts that were actually added/updated
                        "formatted_facts": formatted_facts,
                        "agent": agent,
                        "task_id": task_id,
                    }
                )

                # Extra debug info
                print(f"[Memory] Sent {len(effective_changes)} facts to result queue")
            except Exception as e:
                print(f"[Memory] Error in memory update: {e}")
                import traceback
                print(traceback.format_exc())

                # Report error
                self.result_queue.put(
                    {
                        "operation": "error",
                        "status": "error",
                        "message": f"Error updating memory: {str(e)}",
                        "task_id": task_id,
                    }
                )
        else:
            print("[Memory] No facts extracted")
            self.result_queue.put(
                {
                    "operation": "extract_facts",
                    "status": "success",
                    "facts": [],
                    "task_id": task_id,
                }
            )

    def _extract_contact_information(self, query, response):
        """Extract contact information directly using regex patterns.

        This is a specialized extraction method that looks specifically for contact information
        like email addresses, phone numbers, and addresses, which are always important to remember.
        """
        combined_text = f"{query} {response}"
        facts = []

        # Email extraction
        email_pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
        emails = re.findall(email_pattern, combined_text)

        for email in emails:
            # Look for various context patterns
            email_context_patterns = [
                r"(\w+)\s+(?:email|e-mail)\s+is\s+" + re.escape(email),
                r"(?:email|e-mail)\s+(?:is|at)\s+" + re.escape(email),
                r"(?:my|new)\s+(\w+)\s+(?:email|e-mail)\s+is\s+" + re.escape(email),
            ]

            context_found = False
            for pattern in email_context_patterns:
                email_context_match = re.search(pattern, combined_text, re.IGNORECASE)
                if email_context_match:
                    try:
                        context = email_context_match.group(1)
                        if context.lower() in ["my", "the", "an", "your"]:
                            facts.append(f"User's email is {email}")
                        else:
                            facts.append(f"User's {context} email is {email}")
                        context_found = True
                        break
                    except IndexError:
                        # Pattern matched but no capture group
                        facts.append(f"User's email is {email}")
                        context_found = True
                        break

            if not context_found:
                facts.append(f"User's email is {email}")

        # Phone extraction
        phone_pattern = r"\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b"
        phones = re.findall(phone_pattern, combined_text)

        for phone in phones:
            # Look for various context patterns
            phone_context_patterns = [
                r"(\w+)\s+(?:phone|number|cell|mobile)\s+is\s+" + re.escape(phone),
                r"(?:phone|number|cell|mobile)\s+(?:is|at)\s+" + re.escape(phone),
                r"(?:my|new)\s+(\w+)\s+(?:phone|number)\s+is\s+" + re.escape(phone),
            ]

            context_found = False
            for pattern in phone_context_patterns:
                phone_context_match = re.search(pattern, combined_text, re.IGNORECASE)
                if phone_context_match:
                    try:
                        context = phone_context_match.group(1)
                        if context.lower() in ["my", "the", "a", "your"]:
                            facts.append(f"User's phone number is {phone}")
                        else:
                            facts.append(f"User's {context} phone number is {phone}")
                        context_found = True
                        break
                    except IndexError:
                        # Pattern matched but no capture group
                        facts.append(f"User's phone number is {phone}")
                        context_found = True
                        break

            if not context_found:
                facts.append(f"User's phone number is {phone}")

        # Address extraction - this is more complex and might need LLM assistance
        # For now, just check for keywords and extract simple patterns
        if "address" in combined_text.lower():
            address_patterns = [
                r"(?:address|location)\s+is\s+(.+?)(?:\.|,|\n|$)",
                r"(?:live|locate)\s+at\s+(.+?)(?:\.|,|\n|$)",
            ]

            for pattern in address_patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    address = match.group(1).strip()
                    if len(address) > 10:  # Ensure it's a substantial address
                        facts.append(f"User's address is {address}")

        return facts

    def _conversation_likely_important(self, query, response):
        """Determine if a conversation is likely to contain important information worth remembering."""
        # Check for indicators of important information in the conversation
        combined_text = (query + " " + response).lower()

        # Definite importance markers - scheduling, contact info, etc.
        important_indicators = [
            # Time/scheduling related
            r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b",  # Time
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",  # Month day
            r"\b(?:tomorrow|today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",  # Day references
            r"\bschedul(?:e|ed|ing)\b",  # Schedule-related
            r"\bmeeting\b",  # Meetings
            r"\bappointment\b",  # Appointments
            r"\bdeadline\b",  # Deadlines
            # Contact information
            r"\b(?:phone|email|address|contact)\b",  # Contact methods
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",  # Email addresses
            r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b",  # Phone numbers like 555-123-4567
            r"\bmy (?:phone|email|address) is\b",  # "My phone/email/address is" patterns
            r"\bnew (?:phone|email|address)\b",  # "New phone/email/address" patterns
            # Preferences and commitments
            r"\bprefer(?:s|red)?\b",  # Preferences
            r"\bagree(?:d|s)?\b",  # Agreements
            r"\bwill\b(?!.*\?)",  # Commitments (not in questions)
        ]

        for pattern in important_indicators:
            if re.search(pattern, combined_text):
                return True

        # Look for numeric information which often indicates specificity
        # But only counts if there are also other indicators of importance
        has_numbers = bool(re.search(r"\b\d+\b", combined_text))

        # Keywords that suggest importance
        important_keywords = [
            "important",
            "critical",
            "essential",
            "key",
            "priority",
            "urgent",
            "remember",
            "note",
            "save",
            "project",
            "task",
            "reminder",
            "due",
            "need",
            "must",
            "required",
            "contact",
            "reach me",
        ]

        keyword_count = sum(1 for word in important_keywords if word in combined_text)

        # If it has both numbers and multiple important keywords, it's likely important
        if has_numbers and keyword_count >= 1:
            return True

        # Or if it has multiple important keywords, it's worth checking
        if keyword_count >= 2:
            return True

        # Check for question-answer patterns that signal important information exchange
        info_seeking_patterns = [
            r"what (?:is|are) (?:my|the|your)",
            r"when (?:is|are|will|should)",
            r"where (?:is|are|will|should)",
            r"how (?:many|much|do|does|can|should)",
        ]

        for pattern in info_seeking_patterns:
            if re.search(pattern, query.lower()):
                # If the response is substantive, it might contain useful information
                if len(response) > 100:
                    return True

        # Default to not important if none of the above conditions are met
        return False

    def _handle_read_task(self, task):
        """Handle reading all facts from memory."""
        task_id = task.get("task_id")

        # Read memory file with shared lock
        memory = self._read_memory_file()

        # Send result back
        self.result_queue.put(
            {
                "operation": "get_memory",
                "status": "success",
                "memory": memory,
                "task_id": task_id,
            }
        )

    def _handle_query_task(self, task):
        """Handle querying facts from memory by keywords."""
        query_terms = task.get("query_terms", [])
        max_facts = task.get("max_facts", 5)
        task_id = task.get("task_id")

        # Read memory with shared lock
        memory = self._read_memory_file()

        # Get relevant facts
        relevant_facts = []
        if memory["facts"] and query_terms:
            query_terms_set = set(query_terms)

            # Create pattern matchers for special types of information
            meeting_pattern = re.compile(r"(\d+)\s+meeting", re.IGNORECASE)
            time_pattern = re.compile(r"\b(\d{1,2})[:\.](\d{2})\s*(am|pm|AM|PM)?\b")
            date_pattern = re.compile(
                r"\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                re.IGNORECASE,
            )

            # Score facts by relevance with improved scoring
            scored_facts = []
            for fact in memory["facts"]:
                fact_lower = fact.lower()
                fact_words = set(fact_lower.split())

                # Base score - calculate different aspects of relevance
                term_overlap = len(query_terms_set.intersection(fact_words))

                # Additional scoring for partial matches (substrings)
                substring_matches = 0
                for term in query_terms_set:
                    if term in fact_lower and term not in fact_words:
                        substring_matches += (
                            0.5  # Partial match is worth less than exact word match
                        )

                # Special pattern matching with higher weights for important information
                pattern_score = 0

                # Check for meeting counts (e.g., "3 meetings")
                meeting_mentions_query = meeting_pattern.findall(" ".join(query_terms))
                meeting_mentions_fact = meeting_pattern.findall(fact_lower)
                if meeting_mentions_query and meeting_mentions_fact:
                    # Extra points for exact meeting count match
                    if meeting_mentions_query[0] == meeting_mentions_fact[0]:
                        pattern_score += 2.0
                    else:
                        pattern_score += 1.0

                # Check for times (e.g., "9:00 AM")
                time_mentions_query = time_pattern.findall(" ".join(query_terms))
                time_mentions_fact = time_pattern.findall(fact_lower)
                if time_mentions_query and time_mentions_fact:
                    for q_time in time_mentions_query:
                        for f_time in time_mentions_fact:
                            # Compare hour component
                            if q_time[0] == f_time[0]:
                                pattern_score += 1.5
                                # If minute component also matches
                                if q_time[1] == f_time[1]:
                                    pattern_score += 0.5

                # Check for date mentions (today, tomorrow, etc.)
                date_mentions_query = date_pattern.findall(" ".join(query_terms))
                date_mentions_fact = date_pattern.findall(fact_lower)
                if date_mentions_query and date_mentions_fact:
                    for date in date_mentions_query:
                        if date.lower() in [d.lower() for d in date_mentions_fact]:
                            pattern_score += 1.5

                # Check for exact key phrases that might be multiple words
                key_phrase_matches = 0
                for i in range(len(query_terms)):
                    for j in range(i + 1, min(i + 4, len(query_terms) + 1)):
                        phrase = " ".join(list(query_terms)[i:j])
                        if len(phrase) > 3 and phrase in fact_lower:
                            key_phrase_matches += 1

                # Calculate total score with weighting
                total_score = (
                    term_overlap
                    + substring_matches
                    + (key_phrase_matches * 2)
                    + pattern_score
                )

                # Special case for schedule/calendar queries - facts containing meeting or time info are boosted
                if (
                    "meeting" in query_terms_set
                    or "schedule" in query_terms_set
                    or "calendar" in query_terms_set
                ):
                    if "meeting" in fact_lower or bool(time_pattern.search(fact_lower)):
                        total_score *= 1.5

                if total_score > 0:
                    scored_facts.append((fact, total_score))

            # Sort by score and return the most relevant facts
            scored_facts.sort(key=lambda x: x[1], reverse=True)

            # Take the top facts, ensuring we only return facts with some relevance
            relevant_facts = [
                fact for fact, score in scored_facts[:max_facts] if score > 0.5
            ]

            if relevant_facts:
                print(
                    f"[Memory] Found {len(relevant_facts)} relevant facts for query terms: {query_terms_set}"
                )

        # Send result back
        self.result_queue.put(
            {
                "operation": "query_memory",
                "status": "success",
                "facts": relevant_facts,
                "task_id": task_id,
            }
        )

    def _handle_clear_task(self, task):
        """Handle clearing all memory."""
        task_id = task.get("task_id")

        # Clear memory with exclusive lock
        empty_memory = {"facts": [], "conversation_history": []}
        self._write_memory_file(empty_memory)

        # Send result back
        self.result_queue.put(
            {"operation": "clear_memory", "status": "success", "task_id": task_id}
        )

    def _extract_facts(self, query, response):
        """Extract only important factual information from conversations that would likely be needed in future interactions."""
        # Create a more selective prompt to extract facts in Alpaca format
        extraction_prompt = f"""
### Instruction:
Extract 0-3 truly important factual pieces of information from this conversation that are *highly likely* to be needed in future interactions. Only extract information if it seems genuinely important to remember.

### Input:
User: {query}
Assistant: {response}

### Context:
- Be very selective - only extract facts that are clearly important to remember for future reference
- Focus on specific, actionable information with high future utility
- Priority information types (extract ONLY if present):
  1. Specific scheduling details (exact meeting times, dates, participants)
  2. Important user preferences that affect future interactions (e.g., "I prefer morning meetings")
  3. Critical deadlines or time-sensitive information
  4. Specific personal details explicitly shared by the user (names of people/organizations they mention)
  5. Concrete decisions or commitments made
- DO NOT extract:
  - General knowledge or facts not specific to the user
  - Hypothetical scenarios or speculative information
  - Opinions or subjective statements
  - Information with low probability of being referenced again
  - Temporary or trivial details
  - Vague or ambiguous statements
- If no clearly important information is present, return an empty array []
- Facts must be self-contained and understandable without the original conversation

### Output:
Format your response as a valid JSON array of strings containing only the extracted facts, for example:
["Fact 1", "Fact 2", "Fact 3"]
If no important facts are present, return [].
Do not include any explanations, notes, or text outside of the JSON array.
"""

        # Create extraction payload with more restrictive temperature
        payload = {
            "model": self.model_name,
            "prompt": extraction_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "max_tokens": 300},
            "system": "You are a highly selective fact extraction assistant. Your job is to identify only the most important factual information that will likely be needed in future conversations. Err on the side of not extracting information unless it's clearly important. Never invent facts or extract trivial information.",
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

            # Try multiple approaches to extract facts
            facts = []

            # Approach 1: Try to parse the entire response as JSON
            try:
                potential_facts = json.loads(result.strip())
                if isinstance(potential_facts, list):
                    facts = potential_facts
            except json.JSONDecodeError:
                # Approach 2: Use regex to find JSON array
                try:
                    # Look for anything that resembles a JSON array
                    match = re.search(r"\[(.*?)\]", result, re.DOTALL)
                    if match:
                        # Reconstruct proper JSON
                        json_str = f"[{match.group(1)}]"
                        # Fix potential issues with single quotes
                        json_str = json_str.replace("'", '"')
                        facts = json.loads(json_str)
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Warning: Regex JSON extraction failed: {e}")

                    # Approach 3: Manual line extraction as last resort
                    lines = result.strip().split("\n")
                    potential_facts = []
                    for line in lines:
                        line = line.strip()
                        # Skip empty lines or obvious non-facts
                        if (
                            not line
                            or len(line) < 5
                            or line.startswith("[")
                            or line.startswith("]")
                        ):
                            continue
                        # Clean up the line - remove quotes, numbers, etc.
                        line = line.strip("\"'").strip()
                        line = re.sub(
                            r"^[\d\-\*]\.\s*", "", line
                        )  # Remove "1. " or "- " prefixes
                        if (
                            line and len(line) > 10
                        ):  # Meaningful facts should have some length
                            potential_facts.append(line)

                    if potential_facts:
                        facts = potential_facts[:3]  # Take up to 3 facts

            # Apply additional filtering to ensure only important facts are kept
            valid_facts = []
            for fact in facts:
                # Skip if it's not a string
                if not isinstance(fact, str):
                    continue

                # Skip fact-like placeholders and empty/short strings
                if not (fact and len(fact) > 10 and "fact" not in fact.lower()):
                    continue

                # Skip generic or vague facts
                low_value_patterns = [
                    r"^the user (is|seems|appears)",
                    r"^user (is|seems|appears)",
                    r"^the user (asked|requested|wanted)",
                    r"(talked|spoke|discussed) about",
                    r"(is interested in)",
                    r"general information",
                    r"the conversation (was|is) about",
                ]

                if any(
                    re.search(pattern, fact.lower()) for pattern in low_value_patterns
                ):
                    print(f"[Memory] Skipping low-value fact: {fact}")
                    continue

                # Only keep facts that contain specific, actionable information
                high_value_indicators = [
                    # Scheduling and time
                    r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b",  # Time
                    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",  # Days
                    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",  # Months
                    r"\b(?:tomorrow|today|yesterday)\b",  # Relative days
                    r"\bmeeting\b",  # Meetings
                    r"\bschedule\b",  # Schedule
                    r"\bdeadline\b",  # Deadlines
                    # Contact information
                    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",  # Email addresses
                    r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b",  # Phone numbers like 555-123-4567
                    r"\bemail\b",  # Email keyword
                    r"\bphone\b",  # Phone keyword
                    r"\baddress\b",  # Address keyword
                    r"\bcontact\b",  # Contact keyword
                    # Preferences
                    r"\bprefer(s|red)?\b",  # Preferences
                    r"\blike(s|d)? to\b",  # Likes
                    r"\bdon't like\b",  # Dislikes
                    r"\bfavorite\b",  # Favorites
                    # Commitments
                    r"\bagree(s|d)?\b",  # Agreements
                    r"\bwill\b",  # Future commitments
                    r"\bplan(s|ned)?\b",  # Plans
                    # Personal details
                    r"\bname is\b",  # Names
                    r"\bemail is\b",  # Email
                    r"\bphone\b",  # Phone
                    r"\baddress\b",  # Address
                    r"\bbirthday\b",  # Birthday
                ]

                # Only keep the fact if it contains at least one high-value indicator
                if any(
                    re.search(pattern, fact.lower())
                    for pattern in high_value_indicators
                ):
                    valid_facts.append(fact)
                else:
                    print(
                        f"[Memory] Skipping fact without high value indicators: {fact}"
                    )

            return valid_facts

        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []

    def _read_memory_file(self):
        """Read memory file with shared lock."""
        try:
            with open(self.memory_file, "r") as f:
                # Acquire shared lock (allows other readers but blocks writers)
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    memory = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)  # Release lock
            return memory
        except FileNotFoundError:
            # Create a new memory file if it doesn't exist
            empty_memory = {"facts": [], "conversation_history": []}
            self._write_memory_file(empty_memory)
            return empty_memory
        except json.JSONDecodeError:
            # Handle corrupted memory file
            print(
                f"Warning: Memory file {self.memory_file} is corrupted. Creating new memory."
            )
            empty_memory = {"facts": [], "conversation_history": []}
            self._write_memory_file(empty_memory)
            return empty_memory

    def _write_memory_file(self, memory_data):
        """Write to memory file with exclusive lock."""
        try:
            with open(self.memory_file, "w") as f:
                # Acquire exclusive lock (blocks both readers and writers)
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(memory_data, f, indent=2)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)  # Release lock
            return True
        except Exception as e:
            print(f"Error writing memory file: {e}")
            return False

    def _update_memory_file(self, new_facts, query, response, agent):
        """Update memory file with new facts and conversation history, being more selective about what's stored."""
        # Don't bother updating if there are no facts to add
        if not new_facts:
            return []

        # Read current memory with shared lock first
        memory = self._read_memory_file()

        # Track what happened to each fact for better feedback
        fact_results = {"added": [], "merged": [], "skipped": []}

        # Apply importance decay to old facts - older facts might no longer be relevant
        # Calculate current timestamp for aging purposes
        current_time = time.time()

        # Process each new fact to avoid redundancy
        for new_fact in new_facts:
            # Skip empty or very short facts
            if not new_fact or len(new_fact) < 10:
                continue

            # Convert to lowercase for comparison
            new_fact_lower = new_fact.lower()

            # Check if this fact or a similar one already exists
            should_add = True
            facts_to_remove = []
            best_match = None
            best_similarity = 0

            # Generate a more specific hash for this fact based on key content
            # This helps better identify semantically equivalent facts
            new_fact_hash = self._generate_fact_hash(new_fact)

            for i, existing_fact in enumerate(memory["facts"]):
                existing_fact_lower = existing_fact.lower()

                # Generate hash for existing fact
                existing_fact_hash = self._generate_fact_hash(existing_fact)

                # If the hashes match, these facts are considered equivalent
                if (
                    new_fact_hash
                    and existing_fact_hash
                    and new_fact_hash == existing_fact_hash
                ):
                    # Skip this fact entirely - we already have an equivalent one
                    should_add = False
                    fact_results["skipped"].append(
                        {
                            "new_fact": new_fact,
                            "similar_to": existing_fact,
                            "similarity": 1.0,
                            "reason": "hash_match",
                        }
                    )
                    break

                # Calculate similarity ratio between facts
                similarity = self._calculate_similarity(
                    new_fact_lower, existing_fact_lower
                )

                # Keep track of the best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing_fact

                # Use a stricter similarity threshold (0.6 to 0.7 is more selective)
                if similarity > 0.6:
                    # Check which fact is more specific and valuable
                    new_fact_value = self._evaluate_fact_value(new_fact)
                    existing_fact_value = self._evaluate_fact_value(existing_fact)

                    # If existing fact contains more valuable information
                    if existing_fact_value >= new_fact_value:
                        # Skip adding the new fact
                        should_add = False
                        fact_results["skipped"].append(
                            {
                                "new_fact": new_fact,
                                "similar_to": existing_fact,
                                "similarity": similarity,
                                "reason": "lower_value",
                            }
                        )
                        break
                    else:
                        # New fact is better - mark the old one for replacement
                        facts_to_remove.append(i)

                # Also check for strict containment (one is a substring of the other)
                # This helps catch cases where one fact completely contains another
                elif existing_fact_lower in new_fact_lower:
                    # New fact contains the old one - replace old fact
                    facts_to_remove.append(i)
                elif new_fact_lower in existing_fact_lower:
                    # Old fact contains the new one - skip new fact
                    should_add = False
                    fact_results["skipped"].append(
                        {
                            "new_fact": new_fact,
                            "similar_to": existing_fact,
                            "similarity": similarity,
                            "reason": "contained_in_existing",
                        }
                    )
                    break

            # Remove outdated facts (in reverse order to avoid index issues)
            if should_add and facts_to_remove:
                removed_facts = []
                for idx in sorted(facts_to_remove, reverse=True):
                    removed_fact = memory["facts"].pop(idx)
                    removed_facts.append(removed_fact)
                    print(f"[Memory] Replaced fact: {removed_fact} with: {new_fact}")

                fact_results["merged"].append(
                    {"new_fact": new_fact, "replaced": removed_facts}
                )

            # Add the new fact if it's not redundant
            if should_add:
                # Add timestamp metadata to the fact for future reference
                # This is used for importance decay over time
                timestamp = time.time()

                memory["facts"].append(new_fact)
                if not facts_to_remove:  # Only if it's truly new, not a replacement
                    fact_results["added"].append(new_fact)
                    print(f"[Memory] Added new fact: {new_fact}")
            else:
                print(f"[Memory] Skipped redundant fact: {new_fact}")

        # Add to conversation history only if meaningful facts were exchanged
        if fact_results["added"] or fact_results["merged"]:
            timestamp = datetime.datetime.now().isoformat()

            # Limit stored conversation to just essential information
            memory["conversation_history"].append(
                {
                    "timestamp": timestamp,
                    "user_input": query[:100] + "..." if len(query) > 100 else query,
                    "agent": agent,
                    "response_summary": (
                        response[:100] + "..." if len(response) > 100 else response
                    ),
                }
            )

            # Limit conversation history more aggressively - only need recent history
            if len(memory["conversation_history"]) > 30:  # Reduced from 50
                memory["conversation_history"] = memory["conversation_history"][-30:]

        # Write updated memory with exclusive lock
        self._write_memory_file(memory)

        # Return a list of facts that were actually added or updated for better feedback
        effective_changes = fact_results["added"] + [
            item["new_fact"] for item in fact_results["merged"]
        ]
        return effective_changes

    def _generate_fact_hash(self, fact):
        """Generate a semantic hash for a fact based on key content."""
        if not fact:
            return None

        # Normalize the text
        text = fact.lower()

        # Extract key entities that determine the semantic meaning
        # This focuses on the most important parts that define uniqueness

        # Extract dates and times
        date_time_patterns = [
            r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b",  # Times
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",  # Month day
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",  # Day month
            r"\b(?:tomorrow|today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",  # Day references
        ]

        date_time_entities = []
        for pattern in date_time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            date_time_entities.extend(matches)

        # Extract numbers that might be significant (meeting counts, etc.)
        number_patterns = [
            r"\b\d+\s+(?:meeting|appointment|call|session)s?\b",  # Number of events
        ]

        number_entities = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            number_entities.extend(matches)

        # Extract key terms that represent the core subject
        # Remove common words first
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
            "will",
            "would",
            "shall",
            "should",
            "can",
            "could",
            "may",
            "might",
            "to",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
        }

        # Get the key words by removing stopwords
        words = text.split()
        key_words = [word for word in words if word not in stopwords and len(word) > 2]

        # Create a hash combining all these elements
        hash_components = []
        hash_components.extend(sorted(date_time_entities))  # Add date/time entities
        hash_components.extend(sorted(number_entities))  # Add numeric entities

        # Add a subset of the key words (the most important ones)
        if key_words:
            # Use the 3-5 most significant words for the hash
            # These are likely nouns and verbs that define the core meaning
            sorted_words = sorted(key_words, key=len, reverse=True)
            hash_components.extend(sorted_words[: min(5, len(sorted_words))])

        if not hash_components:
            return None

        # Create a deterministic hash from these components
        hash_string = "_".join(hash_components)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _evaluate_fact_value(self, fact):
        """Evaluate the importance and specificity of a fact."""
        if not fact:
            return 0

        text = fact.lower()
        value_score = 0

        # Specific dates and times are valuable
        date_time_patterns = [
            r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b",  # Times
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",  # Month day
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",  # Day month
        ]

        for pattern in date_time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            value_score += len(matches) * 2  # Times and dates are highly valuable

        # Specific days (tomorrow, Monday) are valuable but less so than exact dates
        day_patterns = [
            r"\b(?:tomorrow|today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",  # Day references
        ]

        for pattern in day_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            value_score += len(matches)  # Days are valuable

        # Numbers often indicate specificity (3 meetings, etc.)
        if re.search(r"\b\d+\b", text):
            value_score += 1

        # Length can indicate more details
        value_score += min(len(text) / 50, 2)  # Cap at 2 points for length

        # Specific keywords indicate important information
        important_keywords = [
            "meeting",
            "appointment",
            "schedule",
            "deadline",
            "email",
            "phone",
            "address",
            "contact",
            "project",
            "task",
            "reminder",
            "prefer",
            "important",
            "urgent",
            "critical",
            "required",
        ]

        for keyword in important_keywords:
            if keyword in text:
                value_score += 0.5

        return value_score

    def _calculate_similarity(self, text1, text2):
        """Calculate similarity ratio between two strings with improved accuracy."""
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

    def _handle_add_history_task(self, task):
        """Handle adding a conversation history entry directly to memory."""
        history_entry = task.get("history_entry")
        task_id = task.get("task_id")

        if not history_entry:
            self.result_queue.put({
                "operation": "add_history",
                "status": "error",
                "message": "No history entry provided",
                "task_id": task_id
            })
            return

        try:
            # Read current memory with shared lock first
            memory = self._read_memory_file()

            # Add the history entry
            memory["conversation_history"].append(history_entry)

            # Limit history size
            if len(memory["conversation_history"]) > 50:
                memory["conversation_history"] = memory["conversation_history"][-50:]

            # Write updated memory with exclusive lock
            success = self._write_memory_file(memory)

            if success:
                print(f"[Memory] Added history entry: {history_entry.get('agent')} - {history_entry.get('user_input')[:30]}...")
                self.result_queue.put({
                    "operation": "add_history",
                    "status": "success",
                    "task_id": task_id
                })
            else:
                self.result_queue.put({
                    "operation": "add_history",
                    "status": "error",
                    "message": "Failed to write memory file",
                    "task_id": task_id
                })
        except Exception as e:
            print(f"[Memory] Error adding history entry: {e}")
            self.result_queue.put({
                "operation": "add_history",
                "status": "error",
                "message": f"Error adding history entry: {str(e)}",
                "task_id": task_id
            })


# For testing stand-alone functionality
if __name__ == "__main__":
    memory_manager = MemoryManagerProcess()
    memory_manager.start()

    # Example test
    try:
        memory_manager.task_queue.put(
            {
                "type": "extract",
                "query": "I have three meetings tomorrow at 9am, 11am, and 2pm.",
                "response": "I've noted your three meetings for tomorrow. Would you like me to set reminders for these meetings?",
                "agent": "calendar_manager",
                "task_id": "test1",
            }
        )

        # Wait for result
        time.sleep(2)

        # Read results
        while not memory_manager.result_queue.empty():
            result = memory_manager.result_queue.get()
            print(f"Result: {result}")

        # Shutdown
        memory_manager.task_queue.put({"type": "shutdown"})
        memory_manager.join(timeout=5)

    except KeyboardInterrupt:
        print("Test interrupted")
        memory_manager.terminate()
