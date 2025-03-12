#!/usr/bin/env python3

import os
import json
import time
import re
import multiprocessing
import requests
import uuid
from collections import defaultdict

# Configuration
MODEL_NAME = "llama3.1:latest"
API_URL = "http://localhost:11434/api/generate"


class AgentTrainerProcess(multiprocessing.Process):
    """Process for training and improving agents based on interaction feedback."""

    def __init__(self, registry_file="agent_registry.json"):
        super().__init__()
        self.registry_file = registry_file
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.running = True

        # Store interaction data for analysis
        self.agent_interactions = defaultdict(list)
        self.training_notes = defaultdict(list)
        self.improvement_suggestions = defaultdict(list)

        # Patterns that suggest user dissatisfaction
        self.dissatisfaction_patterns = [
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

        # Types of improvements we can make
        self.improvement_types = [
            "add_specificity",
            "add_examples",
            "clarify_scope",
            "add_constraints",
            "address_misconception",
            "improve_format",
        ]

    def run(self):
        """Main process loop that handles training tasks."""
        print(f"Agent Trainer process started (PID: {os.getpid()})")

        while self.running:
            try:
                # Check if there are any tasks in the queue (non-blocking)
                if not self.task_queue.empty():
                    task = self.task_queue.get()

                    # Process different task types
                    if task["type"] == "shutdown":
                        self.running = False
                        print("Agent Trainer process shutting down...")
                        break

                    elif task["type"] == "record_interaction":
                        self._handle_record_interaction(task)

                    elif task["type"] == "analyze_agent":
                        self._handle_analyze_agent(task)

                    elif task["type"] == "improve_prompt":
                        self._handle_improve_prompt(task)

                    elif task["type"] == "direct_feedback":
                        self._handle_direct_feedback(task)

                # Sleep briefly to prevent CPU hogging
                time.sleep(0.1)

            except Exception as e:
                print(f"Error in Agent Trainer process: {e}")
                # Send error to main process
                self.result_queue.put(
                    {
                        "type": "error",
                        "error": str(e),
                        "task_id": task.get("task_id", "unknown"),
                    }
                )

    def _handle_record_interaction(self, task):
        """Record an interaction between user and agent for future analysis."""
        agent_name = task["agent_name"]
        query = task["query"]
        response = task["response"]
        timestamp = task.get("timestamp", time.time())
        task_id = task.get("task_id", str(uuid.uuid4()))

        # Store the interaction
        interaction = {
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "analyzed": False,
            "dissatisfaction_detected": False,
        }

        # Check for patterns of dissatisfaction in the query
        if self._check_for_dissatisfaction(query):
            interaction["dissatisfaction_detected"] = True
            # Add a training note about this interaction
            self._add_training_note(
                agent_name, f"User showed dissatisfaction with response to: '{query}'"
            )

        # Add interaction to history
        self.agent_interactions[agent_name].append(interaction)

        # Limit history size to prevent memory issues
        if len(self.agent_interactions[agent_name]) > 50:
            self.agent_interactions[agent_name] = self.agent_interactions[agent_name][
                -50:
            ]

        # Return success
        self.result_queue.put(
            {
                "type": "record_interaction_result",
                "status": "success",
                "agent_name": agent_name,
                "task_id": task_id,
            }
        )

    def _handle_analyze_agent(self, task):
        """Analyze an agent's performance based on recorded interactions."""
        agent_name = task["agent_name"]
        task_id = task.get("task_id", str(uuid.uuid4()))

        # Get interactions for this agent
        interactions = self.agent_interactions.get(agent_name, [])

        if not interactions:
            self.result_queue.put(
                {
                    "type": "analyze_agent_result",
                    "status": "success",
                    "message": f"No interactions found for agent '{agent_name}'",
                    "task_id": task_id,
                    "needs_improvement": False,
                }
            )
            return

        # Count dissatisfaction instances
        dissatisfaction_count = sum(
            1 for i in interactions if i.get("dissatisfaction_detected", False)
        )

        # Calculate percentage of problematic interactions
        total_interactions = len(interactions)
        problem_percentage = (
            (dissatisfaction_count / total_interactions) * 100
            if total_interactions > 0
            else 0
        )

        # Determine if improvement is needed (if >20% of recent interactions show problems)
        needs_improvement = problem_percentage >= 20 and dissatisfaction_count >= 2

        # Generate analysis result
        analysis = {
            "agent_name": agent_name,
            "total_interactions": total_interactions,
            "dissatisfaction_count": dissatisfaction_count,
            "problem_percentage": round(problem_percentage, 2),
            "needs_improvement": needs_improvement,
            "timestamp": time.time(),
        }

        # If improvement is needed, identify patterns in problematic interactions
        if needs_improvement:
            analysis["patterns"] = self._identify_problem_patterns(
                agent_name, interactions
            )

        # Send result back
        self.result_queue.put(
            {
                "type": "analyze_agent_result",
                "status": "success",
                "analysis": analysis,
                "task_id": task_id,
                "needs_improvement": needs_improvement,
            }
        )

    def _handle_improve_prompt(self, task):
        """Generate an improved system prompt for an agent based on analysis."""
        agent_name = task["agent_name"]
        current_prompt = task["current_prompt"]
        task_id = task.get("task_id", str(uuid.uuid4()))

        # Get training notes and problem patterns for this agent
        notes = self.training_notes.get(agent_name, [])

        if not notes:
            self.result_queue.put(
                {
                    "type": "improve_prompt_result",
                    "status": "error",
                    "message": f"No training notes found for agent '{agent_name}'",
                    "task_id": task_id,
                }
            )
            return

        # Prepare a prompt for the LLM to generate an improved system prompt
        improvement_prompt = self._create_improvement_prompt(
            agent_name, current_prompt, notes
        )

        # Call the LLM to get an improved prompt
        try:
            improved_prompt = self._call_llm_for_improvement(improvement_prompt)

            # Send the improved prompt back
            self.result_queue.put(
                {
                    "type": "improve_prompt_result",
                    "status": "success",
                    "agent_name": agent_name,
                    "improved_prompt": improved_prompt,
                    "notes": notes[:5],  # Include the most recent notes
                    "task_id": task_id,
                }
            )

            # Clear the training notes for this agent since they've been addressed
            self.training_notes[agent_name] = []

        except Exception as e:
            self.result_queue.put(
                {
                    "type": "improve_prompt_result",
                    "status": "error",
                    "message": f"Error generating improved prompt: {str(e)}",
                    "task_id": task_id,
                }
            )

    def _handle_direct_feedback(self, task):
        """Handle direct feedback from the user about an agent's performance."""
        agent_name = task["agent_name"]
        feedback = task["feedback"]
        task_id = task.get("task_id", str(uuid.uuid4()))

        # Add the feedback as a training note
        self._add_training_note(agent_name, f"Direct user feedback: {feedback}")

        # Analyze the feedback to determine improvement type and points
        feedback_analysis = self._analyze_feedback(feedback)
        improvement_type = feedback_analysis["type"]
        improvement_points = feedback_analysis["points"]

        # Add to improvement suggestions
        self.improvement_suggestions[agent_name].append(
            {
                "feedback": feedback,
                "improvement_type": improvement_type,
                "improvement_points": improvement_points,
                "timestamp": time.time(),
            }
        )

        # Send confirmation with more details
        self.result_queue.put(
            {
                "type": "direct_feedback_result",
                "status": "success",
                "message": f"Feedback recorded for agent '{agent_name}'",
                "agent_name": agent_name,
                "improvement_type": improvement_type,
                "improvement_points": improvement_points,
                "task_id": task_id,
            }
        )

    def _check_for_dissatisfaction(self, query):
        """Check if a user query indicates dissatisfaction with a previous response."""
        query_lower = query.lower()

        # Check for dissatisfaction patterns
        for pattern in self.dissatisfaction_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check for repeated questions (often a sign of dissatisfaction)
        # This would require comparing with previous queries

        return False

    def _add_training_note(self, agent_name, note):
        """Add a training note for an agent."""
        self.training_notes[agent_name].append({"note": note, "timestamp": time.time()})

        # Limit notes to prevent memory issues
        if len(self.training_notes[agent_name]) > 20:
            self.training_notes[agent_name] = self.training_notes[agent_name][-20:]

    def _identify_problem_patterns(self, agent_name, interactions):
        """Identify patterns in problematic interactions."""
        # Filter for problematic interactions
        problem_interactions = [
            i for i in interactions if i.get("dissatisfaction_detected", False)
        ]

        if not problem_interactions:
            return []

        # Extract queries from problematic interactions
        problem_queries = [i["query"] for i in problem_interactions]

        # Look for common patterns (simplified version)
        patterns = []

        # Check for questions about specific topics
        topics = self._extract_common_topics(problem_queries)
        if topics:
            patterns.append(f"Agent struggles with queries about: {', '.join(topics)}")

        # Check for question types (how, why, what, etc.)
        question_types = self._extract_question_types(problem_queries)
        if question_types:
            patterns.append(
                f"Agent struggles with question types: {', '.join(question_types)}"
            )

        return patterns

    def _extract_common_topics(self, queries):
        """Extract common topics from a list of queries."""
        # This is a simplified implementation
        topics = []

        # Look for common nouns or key terms
        topic_patterns = [
            (r"\b(?:schedul(e|ing)|calendar|meeting|appointment)\b", "scheduling"),
            (r"\b(?:research|find|search|information|data)\b", "research"),
            (r"\b(?:code|program|develop|software|bug|debug)\b", "coding"),
            (r"\b(?:write|draft|create|content|article)\b", "writing"),
            (r"\b(?:advice|recommend|suggest|help with)\b", "advice"),
        ]

        for pattern, topic in topic_patterns:
            for query in queries:
                if re.search(pattern, query.lower()):
                    if topic not in topics:
                        topics.append(topic)

        return topics

    def _extract_question_types(self, queries):
        """Extract types of questions from a list of queries."""
        question_types = []

        # Look for question words
        question_patterns = [
            (r"\bhow\b", "how"),
            (r"\bwhy\b", "why"),
            (r"\bwhat\b", "what"),
            (r"\bwhen\b", "when"),
            (r"\bwhere\b", "where"),
            (r"\bwhich\b", "which"),
            (r"\bcan you\b", "request"),
            (r"\bcould you\b", "request"),
        ]

        for pattern, q_type in question_patterns:
            for query in queries:
                if re.search(pattern, query.lower()):
                    if q_type not in question_types:
                        question_types.append(q_type)

        return question_types

    def _analyze_feedback(self, feedback):
        """Analyze feedback to determine what type of improvement is needed and specific improvement points."""
        feedback_lower = feedback.lower()

        # Initialize improvement type and specific points
        improvement_type = "general_improvement"
        improvement_points = []

        # Look for keywords indicating specific improvement types with detailed categorization
        if any(
            word in feedback_lower
            for word in [
                "specific",
                "detail",
                "vague",
                "unclear",
                "precise",
                "explicit",
            ]
        ):
            improvement_type = "add_specificity"
            improvement_points.append(
                "Add more specific details and precise instructions"
            )
            if "example" in feedback_lower:
                improvement_points.append(
                    "Include concrete examples to illustrate expected behavior"
                )
            if "context" in feedback_lower:
                improvement_points.append("Provide more contextual information")

        elif any(
            word in feedback_lower
            for word in ["example", "instance", "show", "demonstrate", "illustrate"]
        ):
            improvement_type = "add_examples"
            improvement_points.append(
                "Add practical examples to demonstrate correct responses"
            )
            improvement_points.append(
                "Include sample inputs and outputs to clarify expectations"
            )

        elif any(
            word in feedback_lower
            for word in ["scope", "focus", "purpose", "role", "responsibility"]
        ):
            improvement_type = "clarify_scope"
            improvement_points.append(
                "Better define the agent's role and responsibilities"
            )
            improvement_points.append(
                "Clarify the boundaries of what the agent should and should not do"
            )
            if "overlap" in feedback_lower:
                improvement_points.append(
                    "Address potential overlaps with other agents"
                )

        elif any(
            word in feedback_lower
            for word in ["constraint", "limit", "boundary", "guardrail", "restriction"]
        ):
            improvement_type = "add_constraints"
            improvement_points.append(
                "Add explicit constraints on the agent's behavior"
            )
            improvement_points.append(
                "Define clear boundaries for appropriate responses"
            )
            if "safety" in feedback_lower or "harmful" in feedback_lower:
                improvement_points.append(
                    "Add safety guidelines to prevent harmful outputs"
                )

        elif any(
            word in feedback_lower
            for word in ["wrong", "incorrect", "misunderstood", "mistake", "error"]
        ):
            improvement_type = "address_misconception"
            improvement_points.append(
                "Correct misconceptions in the agent's understanding"
            )
            improvement_points.append(
                "Provide clearer guidance about expected knowledge areas"
            )
            if "question" in feedback_lower:
                improvement_points.append(
                    "Improve ability to understand user questions correctly"
                )

        elif any(
            word in feedback_lower
            for word in ["format", "structure", "organize", "readability", "layout"]
        ):
            improvement_type = "improve_format"
            improvement_points.append(
                "Improve the structure and organization of responses"
            )
            improvement_points.append("Enhance readability with better formatting")
            if "bullet" in feedback_lower or "list" in feedback_lower:
                improvement_points.append(
                    "Use appropriate lists and bullet points for clarity"
                )

        else:
            # General improvement when no specific category is detected
            improvement_points.append("General improvement based on user feedback")
            improvement_points.append(
                "Review overall agent behavior and response quality"
            )

        # Record that we've done analysis on this feedback
        # Print the analysis to the console for visibility
        print(f"\n--- Agent Trainer: Feedback Analysis ---")
        print(f'Feedback: "{feedback}"')
        print(f"Improvement type: {improvement_type}")
        print("Specific improvement points:")
        for point in improvement_points:
            print(f"- {point}")
        print("-------------------------------------")

        # Return both the type and points for more detailed processing
        return {"type": improvement_type, "points": improvement_points}

    def _create_improvement_prompt(self, agent_name, current_prompt, notes):
        """Create a prompt for the LLM to generate an improved system prompt."""
        # Extract notes text
        note_texts = [n["note"] for n in notes]
        notes_str = "\n".join(
            [f"- {note}" for note in note_texts[-10:]]
        )  # Use the 10 most recent notes

        # Collect improvement points from suggestions
        improvement_points = []
        for suggestion in self.improvement_suggestions.get(agent_name, []):
            if "improvement_points" in suggestion:
                improvement_points.extend(suggestion["improvement_points"])

        # Deduplicate and limit improvement points
        unique_points = []
        for point in improvement_points:
            if point not in unique_points:
                unique_points.append(point)

        # Create a formatted string of improvement points
        if unique_points:
            improvement_points_str = "\n".join(
                [f"- {point}" for point in unique_points[-10:]]
            )
        else:
            # Default improvements if none were explicitly identified
            improvement_points_str = "- Enhance clarity and specificity\n- Improve structure and organization\n- Add more examples if relevant"

        # Get primary improvement types from suggestions
        improvement_types = [
            s["improvement_type"]
            for s in self.improvement_suggestions.get(agent_name, [])
            if "improvement_type" in s
        ]

        # Count occurrences of each improvement type
        type_counts = {}
        for imp_type in improvement_types:
            type_counts[imp_type] = type_counts.get(imp_type, 0) + 1

        # Identify the most common improvement needs
        focus_areas = ""
        if type_counts:
            # Sort by count, highest first
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            focus_areas = "Focus especially on:\n"
            for imp_type, count in sorted_types[:3]:  # Top 3 improvement types
                focus_areas += (
                    f"- {imp_type.replace('_', ' ').title()} ({count} mentions)\n"
                )

        # Create the improvement prompt
        improvement_prompt = f"""### Instruction:
You are an AI Prompt Engineer. Your task is to improve a system prompt for an AI agent named '{agent_name}' based on user feedback.

### Current System Prompt:
```
{current_prompt}
```

### User Feedback and Training Notes:
{notes_str}

### Specific Improvement Points:
{improvement_points_str}

{focus_areas}

### Guidelines for Improvement:
1. Maintain the agent's core functionality and personality
2. Make the prompt more specific and clearer based on the user feedback
3. Add examples where helpful
4. Address any misconceptions or areas where the agent is struggling
5. Ensure the prompt is in Alpaca format with clear sections (Instruction, Context, Guidelines, Response Format)
6. Organize information logically and make the guidance more actionable
7. Don't make the prompt excessively long - aim for conciseness while being comprehensive

### Task:
Create an improved system prompt for the '{agent_name}' agent that addresses the issues identified in the feedback.
The improved prompt should be formatted in Alpaca format with clearly defined sections and should build upon the strengths of the current prompt while fixing its weaknesses.

Return ONLY the improved system prompt without any additional explanations or comments."""

        # Log the improvement prompt for debugging purposes
        print(f"\n--- Creating improvement prompt for '{agent_name}' ---")
        print(
            f"Using {len(notes)} feedback notes and {len(unique_points)} improvement points"
        )
        print(
            f"Main improvement areas: {', '.join(key for key, _ in type_counts.items()[:3])}"
            if type_counts
            else "No specific improvement areas identified"
        )

        return improvement_prompt

    def _call_llm_for_improvement(self, improvement_prompt):
        """Call the LLM to generate an improved system prompt."""
        # Create the payload
        payload = {
            "model": MODEL_NAME,
            "prompt": improvement_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        }

        # Make the API call
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        # Extract the response
        result = response.json()
        improved_prompt = result.get("response", "").strip()

        return improved_prompt
