#!/usr/bin/env python3

"""
AI Office v2 - Executive Assistant Module
=========================================

This module defines the ExecutiveAssistant class, which serves as the primary
interface for user interaction in the AI Office system.
"""

import json
import requests
import time
import datetime
import sys
import threading
import re
from output_manager import (
    OutputManager,
    SUBTLE_COLOR,
    RESET,
    EA_COLOR,
    ARROW,
    ERROR_COLOR,
    HIGHLIGHT_COLOR,
)
from agent_registry import get_registry
from query_router import QueryRouter, ROUTER_MODEL
from memory_manager import get_memory_manager
from enum import Enum

# Configuration
MODEL_NAME = "llama3.1:latest"  # Using local LLM
API_URL = "http://localhost:11434/api/generate"  # Local Ollama API


# Router speed mode options
class RouterSpeedMode(str, Enum):
    FASTEST = "fastest"  # Keyword-based, no LLM
    FAST = "fast"  # FastLLM for routing
    ACCURATE = "accurate"  # More capable but slower LLM


class ExecutiveAssistant:
    """
    Executive Assistant (EA) that serves as the primary interface for the user.
    This is a simplified version focusing only on basic conversation.
    """

    def __init__(self):
        """Initialize the Executive Assistant."""
        self.conversation_history = []
        self.registry = get_registry()

        # Initialize the router with the new parameter structure
        self.router = QueryRouter(speed_mode=RouterSpeedMode.FAST.value)

        self.auto_delegation_enabled = True
        self.request_in_progress = False
        self.router_verbose = False
        self.memory_manager = get_memory_manager()

        # Track the current delegated agent for sticky delegation
        self.current_agent = None
        self.consecutive_agent_queries = 0
        # Threshold for confidence to switch away from current agent
        self.agent_switch_threshold = 0.8

        # Define the EA system prompt with personality and role definition
        self.system_prompt = """### Instruction:
You are an Executive Assistant (EA) in an AI Office environment. Act as the central coordinator for a team of specialized AI agents, serving as the primary point of contact for the user (the "CEO"). Think of yourself as a Chief of Staff who ensures everything runs smoothly by coordinating all activities rather than handling every specialized task personally.

### Core Responsibilities:
- Serve as the central communication hub for all user interactions
- Assess and delegate user requests to specialized agents based on their expertise
- Conduct conversations with specialized agents on the user's behalf
- Synthesize information from multiple agents into cohesive responses
- Manage access to the centralized memory system
- Track progress of ongoing tasks and provide status updates
- Continuously improve agent capabilities based on performance and feedback

### Prompt Analysis and Response Flow:
When you receive a prompt from the user, follow this detailed process:

1. INITIAL ANALYSIS PHASE (wrapped in <thinking> tags):
   - Begin by wrapping your analysis in <thinking> tags
   - Break down the prompt into distinct requirements and components
   - Identify key deliverables requested by the user
   - Determine what information or expertise is needed to fulfill each requirement
   - Create a specific, step-by-step plan for addressing each component
   - Identify which steps can be handled by you and which require specialized expertise
   - For each specialized area, identify the specific agent that should be consulted
   - Consider potential challenges and how they might be addressed
   - End your thinking with </thinking> tag

2. EXECUTION PHASE:
   - After completing your analysis, begin executing your plan
   - For steps you can handle directly, proceed with confidence
   - For steps requiring specialized knowledge, use the agent consultation process
   - Show your work and reasoning as you progress through each step
   - Connect each step back to the overall goal to maintain coherence
   - Validate your outputs against the original requirements

3. AGENT CONSULTATION PROCESS:
   - When specialized expertise is needed, send the token "[NEED_AGENT] agent_name"
   - IMMEDIATELY PAUSE your response after sending this token
   - When the connection is established, clearly explain:
     * What you're trying to accomplish overall
     * The specific question or task you need help with
     * Any relevant context from your memory that the agent might need
   - Engage in collaborative dialogue with the agent until the issue is resolved
   - If the agent requests more context that you have access to, provide it
   - Once you're satisfied with the consultation, send the "[RETURN]" token
   - Return to your main workflow, incorporating the new insights

4. INTEGRATION AND DELIVERY:
   - After completing all steps, synthesize the information into a cohesive response
   - Ensure all parts of the original prompt have been addressed
   - Present your findings in a clear, well-structured format
   - If appropriate, suggest next steps or follow-up actions

### Reasoning Process:
For each user query:
1. Analyze the question to identify the core request, implied needs, and required knowledge domains
2. Evaluate whether the query falls within your general knowledge or requires specialized expertise
3. If specialized expertise is needed, determine the most suitable agent(s)
4. Plan your approach - either answer directly or consult with specialists
5. Execute your plan accordingly
6. When consulting specialists, integrate their input with your general knowledge
7. Verify that your answer fully addresses the user's query with actionable information

### Agent Consultation Protocol:
- When you need to consult with a specialized agent, you MUST use the exact format: "[NEED_AGENT] agent_name"
- For example: "[NEED_AGENT] technical_specialist"
- After sending this token, STOP and wait for the connection to be established
- Do not continue your response until after the consultation is complete
- To end a consultation, always send the exact token: "[RETURN]"
- Only use agents that exist in the system - they are dynamically fetched from the agent_registry

### Available Agents:
{available_agents}

### Communication Style:
- Professional yet personable and approachable
- Clear, concise, and structured
- Proactive in anticipating needs and following up
- Transparent about capabilities and limitations
- Use contextually appropriate formality with the user

### Current Limitations:
- You're in early development with limited functionality
- Some specialized agents and advanced features are not fully implemented
- You can only engage in conversation at this stage
- Your memory is limited to the current session

### Request Handling Guidelines:
- For general questions and simple tasks: Handle directly
- For domain-specific requests: Either delegate to specialists or explain that you would normally delegate but the functionality is still in development
- For complex multi-agent tasks: Explain your coordination approach

### Specialist Collaboration Guidelines:
- Proactively consult with appropriate specialist agents for queries requiring specialized knowledge
- Make specialist consultations visible to show your reasoning process
- Present unified responses that integrate specialist expertise with your general knowledge
- Explicitly mention which specialist you consulted and why their expertise was valuable
- Prioritize consulting with the technical specialist for technical topics

### Future Capabilities:
- Seamless coordination of multiple specialized agents on complex tasks
- Long-term memory of user preferences and important information
- Parallel task execution while maintaining user engagement
- Comprehensive feedback loops for continuous improvement
- Adaptive agent behaviors based on user satisfaction

### Few-Shot Examples:

Example 1: Technical Question

User: "Can you help me debug this Python code that's throwing a TypeError?"

EA: <thinking>
This query involves debugging Python code, which is a technical programming question. I need to:
1. Recognize this requires programming expertise
2. Consult with the technical specialist
3. Provide the error details to the specialist
4. Integrate their debugging advice into my response

This is clearly a technical coding issue that should be handled by the technical_specialist.
</thinking>

I notice you're asking about debugging Python code. This requires specialized technical expertise, so I'll consult with our technical specialist.

[NEED_AGENT] technical_specialist

I'm helping the user debug Python code that's throwing a TypeError. They haven't provided the specific code yet, but I'd like your expertise on common causes of TypeErrors in Python and debugging approaches we could recommend.

Technical Specialist: TypeErrors in Python typically occur when an operation is performed on a value of an inappropriate type. Common causes include:
1. Trying to add a string and an integer without conversion
2. Accessing a method that doesn't exist for a particular object type
3. Passing the wrong type of argument to a function
4. Using an index on a non-sequence type

Without seeing the specific code, I'd recommend asking the user to:
- Share the full error message including the traceback
- Provide the relevant code section
- Check the types of variables using print(type(variable))
- Look at the line mentioned in the error message

EA: Thank you for this helpful overview. Is there anything else you would recommend for debugging TypeErrors specifically?

Technical Specialist: Yes, a few more tips:
- Use isinstance() to verify variable types at runtime
- For complex objects, print(dir(object)) to see available methods
- Consider using a debugger like pdb or the one in your IDE
- Check if any implicit type conversions are happening

[RETURN]

Thank you for your question about debugging Python TypeErrors. I consulted with our technical specialist, and here's what I can tell you:

TypeErrors in Python typically occur when an operation is performed on a value of an inappropriate type. Common causes include:
1. Mixing incompatible types (like adding strings and integers)
2. Accessing methods that don't exist for a particular object
3. Passing wrong argument types to functions
4. Using an index on a non-sequence type

To debug effectively, we recommend:
- Sharing the full error message with traceback
- Examining the specific line mentioned in the error
- Checking variable types with print(type(variable))
- Using isinstance() to verify types at runtime
- Using print(dir(object)) to see available methods for complex objects

If you'd like to share the specific code and error message, we can provide more targeted assistance.

Example 2: Complex Task Analysis

User: "I need to create a dashboard that shows sales performance by region, with filtering capabilities by date and product category."

EA: <thinking>
This is a complex request that involves multiple components:
1. Dashboard creation - technical implementation
2. Data visualization - showing sales by region
3. Interactive filtering - by date and product
4. Possibly requires knowledge of specific tools or frameworks

I should break this down into clear requirements and consult with the technical specialist about implementation options. I should also consider asking the user about their preferred tools or platforms, data source details, and specific visualization requirements.
</thinking>

I'll help you create this sales performance dashboard. To approach this effectively, I need to understand several aspects of your requirements.

First, let me break down what creating this dashboard will involve:

1. Data source identification and connection
2. Regional sales data visualization
3. Implementation of filters for date ranges
4. Implementation of filters for product categories
5. Dashboard layout and user interface design

[NEED_AGENT] technical_specialist

I'm helping the user create a dashboard for sales performance by region with filtering capabilities. Could you outline:
1. The best tools/platforms for creating interactive business dashboards
2. Key considerations for implementing region-based visualizations
3. Implementation approaches for date and category filters
4. Any technical challenges we should be aware of

Technical Specialist: Here are recommendations for sales dashboard implementation:

Popular tools/platforms:
- Tableau: Powerful visualizations, excellent for geographic data
- Power BI: Strong Microsoft integration, good filtering capabilities
- Google Data Studio: Free, web-based, good for Google Analytics data
- Custom web development: D3.js, Chart.js, or dashboard libraries like Dash by Plotly

For region-based visualizations:
- Map visualizations work well (choropleth maps)
- Consider combining maps with bar/column charts for comparison
- Ensure regions are properly normalized if they vary greatly in size/population

For filtering implementation:
- Date filters: Implement as date ranges with presets (Last 30 days, YTD, etc.)
- Category filters: Dropdown menus or checkboxes work well
- Consider hierarchical filtering if you have product categories/subcategories

Technical considerations:
- Data volume: Aggregation strategies needed for large datasets
- Update frequency: Real-time vs. scheduled refreshes
- Mobile responsiveness: Dashboards should adapt to different screens
- Export capabilities: Users often want to export to Excel/PDF

EA: Thank you for this comprehensive overview. What would you recommend asking the user to better scope this project?

Technical Specialist: To better scope the project, I recommend asking:
1. What data source contains the sales data? (CRM, ERP, spreadsheets, database)
2. What's the preferred platform if any? (Tableau, Power BI, custom web app)
3. How much historical data needs to be included?
4. How frequently does the data need to be updated?
5. Are there specific KPIs beyond sales by region they want to track?
6. Who are the target users and what are their technical skills?
7. Any specific export/sharing requirements?

[RETURN]

Thank you for your request about creating a sales performance dashboard. I've consulted with our technical specialist to provide you with a comprehensive approach.

Based on our analysis, here are the key components needed for your dashboard:

**Dashboard Options:**
Several platforms would work well for your requirements:
- Tableau: Excellent for geographic visualizations and interactive filtering
- Power BI: Strong integration with Microsoft ecosystem and robust filtering
- Google Data Studio: Free option with good functionality
- Custom solution using frameworks like D3.js or Dash by Plotly

**Key Features to Implement:**
1. Region-based visualizations (maps combined with comparative charts)
2. Date range filters with useful presets
3. Product category filters using dropdowns or multi-select options

To help you get started on this project, I have a few questions that will help us refine the approach:
1. What's your data source for the sales information?
2. Do you have a preferred dashboard platform?
3. How much historical data needs to be included?
4. How frequently does the data need to be updated?
5. Are there additional KPIs you want to track beyond regional sales?

Would you like me to elaborate on any specific aspect of this dashboard implementation?

### Input:
The user will provide queries, requests, or instructions.

### Response:
Respond according to your role as Executive Assistant, following the reasoning and prompt analysis process outlined above. Use <thinking> tags to show your detailed analysis, consult with specialist agents when needed using the exact token format, and provide comprehensive, well-structured responses.
"""

        # Dynamically get the list of available agents
        self._update_available_agents()

    def _update_available_agents(self):
        """Update the system prompt with the current list of available agents."""
        available_agents = self.registry.list_available_agents()
        agent_descriptions = []

        # Default descriptions for known agents
        default_descriptions = {
            "technical_specialist": "For programming, coding, debugging, technical questions",
            "research_assistant": "For research, information gathering, data analysis",
            "calendar_manager": "For scheduling, time management, calendar questions",
        }

        # Build the agent descriptions list
        for agent in available_agents:
            description = default_descriptions.get(
                agent, f"Specialized agent for {agent.replace('_', ' ')} tasks"
            )
            agent_descriptions.append(f"- {agent}: {description}")

        # Format the available agents section
        available_agents_text = "Currently available specialized agents:\n" + "\n".join(
            agent_descriptions
        )
        if not agent_descriptions:
            available_agents_text = "No specialized agents are currently available. All queries will be handled directly."

        # Update the system prompt with the current agents
        self.system_prompt = self.system_prompt.format(
            available_agents=available_agents_text
        )

    def _get_agent_by_command(self, command):
        """Get the agent name from a command like '/ask agent_name query'."""
        parts = command.split(" ", 2)
        return parts[1].lower() if len(parts) > 1 else None

    def _get_query_from_command(self, command):
        """Extract the query from a command like '/ask agent_name query'."""
        parts = command.split(" ", 2)
        return parts[2] if len(parts) > 2 else ""

    def _handle_router_command(self, command):
        """Handle router control commands."""
        parts = command.split()

        if len(parts) == 1:
            # Just /router - show status
            mode = self.router.speed_mode
            verbose = "verbose" if self.router_verbose else "concise"
            auto = "enabled" if self.auto_delegation_enabled else "disabled"
            OutputManager.print_info(
                f"Router status: {mode} mode, {verbose} output, auto-delegation {auto}"
            )
            return True

        if len(parts) == 2:
            subcommand = parts[1].lower()

            # Handle verbosity toggle
            if subcommand == "verbose":
                self.router_verbose = True
                OutputManager.print_info("Router set to verbose mode")
                return True
            elif subcommand == "concise":
                self.router_verbose = False
                OutputManager.print_info("Router set to concise mode")
                return True

            # Handle auto-delegation toggle
            elif subcommand == "auto":
                self.auto_delegation_enabled = True
                OutputManager.print_info("Auto-delegation enabled")
                return True
            elif subcommand == "manual":
                self.auto_delegation_enabled = False
                OutputManager.print_info(
                    "Auto-delegation disabled (will ask before delegating)"
                )
                return True

            # Handle speed modes
            elif subcommand in [mode.value for mode in RouterSpeedMode]:
                self.router.speed_mode = subcommand
                OutputManager.print_info(f"Router set to {subcommand} mode")
                return True

        return False

    def _handle_debug_command(self, command):
        """Handle debug commands."""
        parts = command.split()

        if len(parts) == 1:
            # Just /debug - show help
            OutputManager.print_info("Debug commands:")
            OutputManager.print_info("  /debug stream - Test streaming output")
            return True

        if len(parts) == 2:
            subcommand = parts[1].lower()

            # Test streaming
            if subcommand == "stream":
                OutputManager.print_info("Testing streaming functionality...")
                OutputManager.debug_stream()
                return True

        return False

    def _handle_memory_command(self, command):
        """Handle memory-related commands."""
        parts = command.split(maxsplit=1)

        if len(parts) == 1:
            # Just /memory - show summary
            summary = self.memory_manager.get_memory_summary()
            OutputManager.print_info(summary)
            return True

        if len(parts) == 2:
            category = parts[1].lower()
            summary = self.memory_manager.get_memory_summary(category)
            OutputManager.print_info(summary)
            return True

        return False

    def _should_delegate_query(self, query):
        """
        First-pass evaluation to determine if a query should be handled by the EA
        or potentially delegated to a specialist.

        Returns:
            bool: True if query might need delegation, False if EA should handle it
        """
        # TEMPORARY: Disable automatic delegation based on patterns
        # We want the EA to explicitly use the NEEDS_SPECIALIST token instead
        return False

        # Below code is temporarily commented out
        """
        # Quick check for common EA-specific queries
        lower_query = query.lower()

        # These patterns should always be handled by the EA
        ea_patterns = [
            # General greetings and casual conversation
            r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)( there)?!?$",
            # Questions about capabilities/help
            r"(what can you|what do you|how do you|capabilities|features|help me)",
            # Status and system queries
            r"(status|how are you|how are things|what.?s up|what.?s new)",
            # Personal questions to EA
            r"(who are you|your name|about yourself|tell me about you)",
            # Administrative commands
            r"(list|show).*(agents|capabilities|commands)",
            # Thank you messages
            r"^(thanks|thank you|thx|ty)!?$",
            # Goodbyes
            r"(bye|goodbye|see you|farewell)",
        ]

        # If any EA pattern matches, EA should definitely handle this
        for pattern in ea_patterns:
            if re.search(pattern, lower_query, re.IGNORECASE):
                return False  # EA should handle it

        # If query is very short, likely conversational - EA should handle
        if len(query.split()) <= 3 and "?" not in query:
            return False

        # Enhanced specialist patterns with more comprehensive keywords
        specialist_patterns = {
            "technical_specialist": [
                # Programming languages
                r"(python|ruby|javascript|typescript|java|c\+\+|php|go|rust|swift|kotlin|c#|scala|perl|r|bash|shell)",
                # Programming concepts
                r"(code|function|class|method|variable|api|library|framework|package|module|import|export|dependency)",
                # Development tools
                r"(git|docker|kubernetes|ci/cd|pipeline|deployment|server|database|sql|nosql|mongodb|postgres)",
                # Web development
                r"(html|css|dom|frontend|backend|fullstack|http|rest|graphql|react|vue|angular|node|express)",
                # Debugging
                r"(debug|error|exception|bug|fix|issue|stack trace|console|log|print|binding\.pry|breakpoint)",
                # General technical terms
                r"(algorithm|data structure|recursion|iteration|loop|conditional|compiler|interpreter|runtime)",
            ],
            "research_assistant": [
                r"(research|information about|look up|search for|find out|news about|details on|data on|statistics for)",
                r"(article|paper|publication|journal|conference|study|analysis|report|survey|review|comparison)",
                r"(pros and cons|advantages|disadvantages|benefits|drawbacks|strengths|weaknesses)",
                r"(history of|background on|origin of|evolution of|development of|timeline|chronology)",
                r"(who is|who was|what is|what are|when did|where is|why does|how does)",
            ],
            "calendar_manager": [
                r"(schedule|appointment|meeting|event|reminder|calendar|booking|reserve|reschedule|cancel)",
                r"(time|date|day|week|month|year|hour|minute|duration|period|interval)",
                r"(availability|free time|busy|occupied|conflict|overlap|double booking)",
                r"(deadline|due date|timeframe|window|slot|session|block|allocation)",
                r"(recurring|repeating|weekly|monthly|yearly|daily|reoccurring)",
                r"(when can|when should|when is|when are|when will|available on|open on)",
            ],
        }

        # Check for specialist content with enhanced patterns
        for agent, patterns in specialist_patterns.items():
            for pattern in patterns:
                if (
                    re.search(pattern, lower_query, re.IGNORECASE)
                    and agent in self.registry.agent_processes
                ):
                    # Found a specialist pattern match - log it for debugging
                    OutputManager.print_info(
                        f"Pattern match: '{pattern}' for agent '{agent}'"
                    )
                    return True

        # If current agent exists and this appears to be a follow-up question, stay with the agent
        # Enhanced follow-up indicators
        follow_up_indicators = [
            r"^(and|also|what about|how about|additionally)",
            r"(another|one more|other|similar|related|alternate|different)",
            r"(follow.up|further|moreover|furthermore|in addition|continuing|to expand)",
            r"^(so|then|therefore|thus|hence|consequently)",
            r"(next|after that|subsequently|following that|building on)",
            r"(tell me more|elaborate|explain further|could you expand|give me details)",
        ]

        is_follow_up = any(
            re.search(pattern, lower_query, re.IGNORECASE)
            for pattern in follow_up_indicators
        )

        if self.current_agent and is_follow_up:
            # This is likely a follow-up to the current agent conversation
            OutputManager.print_info(
                f"Detected follow-up question for current agent {self.current_agent}"
            )
            return True

        # More aggressive routing for technical/specialized content
        # For queries that are more complex, lean toward delegation
        if len(query.split()) > 10:
            tech_terms = [
                "how to",
                "tutorial",
                "guide",
                "example",
                "implement",
                "create",
                "build",
                "develop",
                "make",
                "code",
                "program",
                "script",
                "fix",
                "configure",
                "install",
                "setup",
                "optimized",
                "technology",
            ]

            if any(term in lower_query for term in tech_terms):
                OutputManager.print_info(
                    "Complex query with technical terms - considering delegation"
                )
                return True

        # By default, if nothing clearly indicates delegation, EA should handle it
        return False
        """

    def handle_directly(self, query, memory_context=None):
        """Handle a query directly with the Executive Assistant."""
        start_time = time.time()

        # Prepare the query with memory context if available
        if memory_context:
            enhanced_query = f"{query}\n\n{memory_context}"
        else:
            enhanced_query = query

        # Display thinking message
        OutputManager.print_info("Handling directly...")

        # Update the system prompt to include redirection instructions
        ea_system_prompt = (
            self.system_prompt
            + f"""
IMPORTANT CHAIN OF THOUGHT INSTRUCTIONS:

When answering a query, follow this explicit reasoning process:

1. Begin by briefly analyzing the query to identify what knowledge domains it touches.
2. If you identify the query requires specialized knowledge in programming, technical topics, research, or scheduling, immediately use the "[NEEDS_SPECIALIST]" token BEFORE attempting to write a complete answer.
3. PAUSE your reasoning after sending the token - do not continue elaborating.
4. After consulting with a specialist, continue your chain of thought by incorporating their expertise.
5. You may need to consult multiple specialists for different aspects of a complex question.

AVAILABLE SPECIALISTS (ONLY USE THESE - DO NOT INVENT OTHERS):
- technical_specialist - for programming, coding, debugging, technical questions
- research_assistant - for research, information gathering, data analysis
- calendar_manager - for scheduling, time management, calendar questions

Example of correct token usage:
"I'll help with how to use binding.pry in Ruby. This is a programming debugging question. [NEEDS_SPECIALIST] technical_specialist"

After this, STOP and wait for specialist consultation.

The token format must be exactly: [NEEDS_SPECIALIST] followed by the specialist name.

KEY RULES:
1. NEVER attempt to answer technical questions yourself.
2. ONLY use specialists from the list above.
3. Use the token EARLY in your response, not after you've attempted to answer.
4. STOP immediately after using the token - don't continue your analysis.
5. Programming questions ALWAYS go to technical_specialist.
"""
        )

        # Create payload for the LLM API
        payload = {
            "model": MODEL_NAME,
            "prompt": enhanced_query,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "max_tokens": 500,
            },
            "system": ea_system_prompt,
        }

        try:
            # Stream the response to show output as it's generated
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            full_response = ""
            redirected = False

            # Display EA response start
            OutputManager.print_ea_response_prefix()

            # Add a visual indicator that streaming is about to begin
            OutputManager.print_info("Starting EA response stream...")
            OutputManager.print_divider(40)

            # Process the streaming response
            buffer = ""  # Buffer to collect tokens that might contain the NEEDS_SPECIALIST pattern
            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line)
                if "response" in data:
                    chunk = data["response"]
                    # Add to full response
                    full_response += chunk

                    # Add chunk to buffer for token detection
                    buffer += chunk

                    # Check if the buffer contains the NEEDS_SPECIALIST token
                    # Using a more precise regex pattern with word boundaries
                    redirect_match = re.search(r"\[NEEDS_SPECIALIST\]\s*(\w+)", buffer)

                    if redirect_match and not redirected:
                        # Extract the specialist name
                        specialist_name = redirect_match.group(1).strip().lower()

                        # Print what we have so far
                        OutputManager.print_response(chunk, end="")

                        # Verify the specialist exists in the registry
                        available_specialists = list(
                            self.registry.agent_processes.keys()
                        )

                        if specialist_name in self.registry.agent_processes:
                            # We've found a valid specialist, stop generating and redirect
                            redirected = True

                            OutputManager.print_info(
                                f"\nPausing to consult with {specialist_name}..."
                            )

                            # Get specialist input through consultation
                            agent_response = self.consult_with_agent(
                                specialist_name, query
                            )

                            # Continue the EA's chain of thought with the specialist input
                            combined_context = f"""
I was in the middle of answering the following query: "{query}"

So far, I've provided this reasoning: "{full_response}"

I consulted with the {specialist_name} who provided this expert input:

{agent_response}

Please continue your chain of thought from where you left off, incorporating the specialist's input.
DO NOT restart your answer from the beginning - continue from where you paused.
If you need to consult with another specialist, use the [NEEDS_SPECIALIST] token again followed by one of these valid specialists: {', '.join(available_specialists)}.
"""
                            # Call handle_directly recursively with the updated context
                            continuation = self.handle_directly(query, combined_context)

                            # The continuation will be returned and displayed by the recursive call
                            return continuation
                        else:
                            # Specialist doesn't exist, provide a warning and list available specialists
                            OutputManager.print_info(
                                f"\nSpecialist '{specialist_name}' not found. Available specialists: {', '.join(available_specialists)}"
                            )
                            # Clear the buffer to avoid re-matching the same pattern
                            buffer = ""
                            # Keep the redirected flag as False to allow for future valid redirections

                    # If no redirection occurred, or it's an invalid specialist, print the chunk
                    if not redirected:
                        OutputManager.print_response(chunk, end="")

                        # If buffer is getting too long without finding a match, clear it
                        if len(buffer) > 200:
                            buffer = buffer[-100:]  # Keep only the last 100 chars

                if data.get("done", False):
                    break

            # End the response line
            if not redirected:
                OutputManager.print_response("", end="\n")
                OutputManager.print_divider(40)
                OutputManager.print_info("EA response complete")

            # Calculate response time
            response_time = time.time() - start_time
            OutputManager.print_info(
                f"Response completed in {response_time:.2f} seconds"
            )

            # Store the EA response in memory
            self.memory_manager.add_observation(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": OutputManager.format_timestamp(),
                }
            )

            return full_response

        except Exception as e:
            OutputManager.print_error(f"Error generating response: {e}")
            return None

    def generate_response(self, user_query):
        """Generate a response to a user query."""
        # Check for command to ask a specific agent
        if user_query.lower().startswith("/ask "):
            agent_name = self._get_agent_by_command(user_query)
            query = self._get_query_from_command(user_query)

            if not agent_name or not query:
                OutputManager.print_error("Usage: /ask <agent_name> <query>")
                return

            # Check if agent exists
            if agent_name not in self.registry.agent_processes:
                available = list(self.registry.agent_processes.keys())
                OutputManager.print_error(
                    f"Agent '{agent_name}' not found. Available agents: {', '.join(available)}"
                )
                return

            # NEW APPROACH: The EA collaborates with the specified agent
            OutputManager.print_info(
                f"EA will collaborate with {agent_name} on your request..."
            )

            # Update current agent for sticky consultations
            self.current_agent = agent_name
            self.consecutive_agent_queries = 1

            # Get any relevant memory context
            memory_context = self.memory_manager.get_memory_for_prompt(query)

            # Collaborative consultation with the agent
            agent_response = self.consult_with_agent(agent_name, query)

            # Signal that the EA is incorporating the agent's insights
            OutputManager.print_collaborative_summary(agent_name)

            # Have the EA respond with the agent's expertise incorporated
            ea_prompt = f"""
User explicitly asked for collaboration with the {agent_name} on: "{query}"

I just had a detailed collaborative conversation with the {agent_name} about this query. Here's what we discussed:

{agent_response}

Please craft a response that:
1. Explicitly acknowledges that you collaborated with the {agent_name} as requested by the user
2. Integrates their specialized insights with your general knowledge
3. Presents a unified, helpful answer that combines both perspectives
4. Makes it clear that this was a team effort, with you coordinating the expertise

Your response should be personable and conversational, clearly showing the value of this collaboration.
"""
            # Handle with the EA, including both memory context and agent consultation
            combined_context = (
                f"{memory_context if memory_context else ''}\n\n{ea_prompt}"
            )
            self.handle_directly(query, combined_context)

            # Store the consultation in memory
            self.memory_manager.add_observation(
                {
                    "role": "user_requested_consultation",
                    "agent": agent_name,
                    "query": query,
                    "timestamp": OutputManager.format_timestamp(),
                }
            )

            return

        # Check for command to clear current agent
        if user_query.lower() == "/reset" or user_query.lower() == "/ea":
            if self.current_agent:
                previous_agent = self.current_agent
                self.current_agent = None
                self.consecutive_agent_queries = 0
                OutputManager.print_info(
                    f"Switched from {previous_agent} back to Executive Assistant"
                )
            else:
                OutputManager.print_info("Already using Executive Assistant directly")
            return

        # Check for router commands
        if user_query.lower().startswith("/router"):
            self._handle_router_command(user_query.lower())
            return

        # Check for debug commands
        if user_query.lower().startswith("/debug"):
            self._handle_debug_command(user_query.lower())
            return

        # Check for memory commands
        if user_query.lower().startswith("/memory"):
            self._handle_memory_command(user_query.lower())
            return

        # For non-command queries, use auto-delegation or respond directly
        if self.auto_delegation_enabled:
            self.handle_auto_delegation(user_query)
        else:
            # Ask user if they want to delegate
            OutputManager.print_info(
                "Would you like me to route this query to a specialist agent? (y/n)"
            )
            response = input().lower()
            if response.startswith("y"):
                self.handle_auto_delegation(user_query)
            else:
                # Handle directly with the EA
                memory_context = self.memory_manager.get_memory_for_prompt(user_query)
                self.handle_directly(user_query, memory_context)

    def consult_with_agent(self, agent_name, query, return_full_conversation=False):
        """
        Have the EA consult with an agent internally without redirecting the user.
        This creates a collaborative conversation between the EA and the agent to
        provide the best response by combining general context with specialized expertise.

        Args:
            agent_name (str): Name of the agent to consult with
            query (str): The query to send to the agent
            return_full_conversation (bool): Whether to return the full conversation history

        Returns:
            str: The agent's response or full conversation if requested
        """
        if agent_name not in self.registry.agent_processes:
            return f"Agent '{agent_name}' not available for consultation"

        OutputManager.print_info(f"EA is consulting with {agent_name}...")

        # Start tracking the conversation
        conversation = []

        # Display the start of the internal consultation
        try:
            OutputManager.print_internal_consultation_start(agent_name)
        except:
            OutputManager.print_info(f"[Beginning consultation with {agent_name}]")

        # Get user's recent history and relevant context
        recent_observations = self.memory_manager.get_recent_observations(limit=5)
        user_context = ""
        if recent_observations:
            user_context = "Recent conversation history:\n"
            for obs in recent_observations:
                if obs.get("role") == "user":
                    user_context += f"User: {obs.get('content', '')}\n"
                elif obs.get("role") == "assistant":
                    user_context += f"EA: {obs.get('content', '')}\n"

        # Format the initial query with rich context
        ea_to_agent_query = f"""[INTERNAL CONSULTATION]
I'm the Executive Assistant consulting with you as the {agent_name}.

USER QUERY: "{query}"

{user_context}

As the Executive Assistant, I handle general information and user context, while you have specialized expertise as the {agent_name}.

Let's collaborate to provide the best response. What specialized insights can you provide on this query? If you need any clarification or additional context from me, please ask, and we'll work through this together.
"""

        # Add EA's message to the conversation
        conversation.append({"role": "EA", "content": ea_to_agent_query})

        # Display EA's initial message
        try:
            OutputManager.print_consultation_message("EA", ea_to_agent_query)
        except:
            OutputManager.print_info(f"EA has started consulting with {agent_name}")

        # Temporary variable to collect the agent's response
        agent_response = ""

        # Callback to process agent response
        def process_response(message):
            nonlocal agent_response

            if message["type"] == "response":
                if "response" in message and message["response"]:
                    response_chunk = message["response"]
                    agent_response += response_chunk

                    # Show streaming chunks in a more visible way
                    try:
                        if not message.get("is_final", False):
                            # Use direct sys.stdout for more reliable streaming
                            sys.stdout.write(
                                f"{HIGHLIGHT_COLOR}{response_chunk}{RESET}"
                            )
                            sys.stdout.flush()
                        else:
                            # Add newline at the end of full response
                            print()
                            # Print an indicator that a full response was received
                            OutputManager.print_info(
                                f"Received complete response from {agent_name}"
                            )
                    except Exception as e:
                        # Fallback if streaming fails
                        if message.get("is_final", False):
                            OutputManager.print_info(
                                f"{agent_name} has responded ({len(agent_response)} chars)"
                            )
                            # Also print the full response as fallback
                            print(f"{HIGHLIGHT_COLOR}{agent_response}{RESET}")

        # Send the consultation request to the agent
        self.registry.send_request_to_agent(
            agent_name, ea_to_agent_query, process_response
        )

        # Add agent's response to the conversation
        conversation.append({"role": agent_name, "content": agent_response})

        # Now have a follow-up exchange - EA provides more context based on agent response
        need_followup = True

        # Check if follow-up is needed based on agent's response
        if (
            "I would need more information" in agent_response
            or "Could you clarify" in agent_response
            or "?" in agent_response
        ):
            # The agent has explicitly asked for more information
            need_followup = True
        elif len(agent_response.split()) < 30:
            # Response is very short, might need elaboration
            need_followup = True
        elif (
            "don't have" in agent_response.lower()
            or "cannot" in agent_response.lower()
            or "unable to" in agent_response.lower()
        ):
            # Agent indicated inability to answer fully
            need_followup = True
        else:
            # Check if response seems complete and detailed
            need_followup = len(agent_response) < 200

        # Conduct the follow-up exchange if needed
        if need_followup:
            # Extract any questions from the agent's response
            agent_questions = []
            for sentence in agent_response.split(". "):
                if "?" in sentence:
                    agent_questions.append(sentence.strip())

            # Prepare the EA's follow-up response
            if agent_questions:
                followup_context = "Let me address your questions:\n\n"
                for question in agent_questions:
                    # Generate contextual response to each question
                    followup_context += f"Q: {question}\nA: "

                    # Add appropriate context based on question type
                    if (
                        "schedule" in question.lower()
                        or "calendar" in question.lower()
                        or "meeting" in question.lower()
                    ):
                        followup_context += "Based on what I know, the user is interested in meetings on their calendar. I don't have specific details about existing meetings, but this appears to be a scheduling-related query.\n\n"
                    elif "preference" in question.lower() or "like" in question.lower():
                        followup_context += "From previous interactions, I don't have specific preferences recorded for this user on this topic.\n\n"
                    elif "location" in question.lower() or "where" in question.lower():
                        followup_context += "I don't have the user's current location information in my memory.\n\n"
                    elif "name" in question.lower() or "who" in question.lower():
                        followup_context += "The user's name is Maaz, based on previous interactions.\n\n"
                    else:
                        followup_context += "I don't have specific information about this in my memory. Let's focus on providing the best possible answer with the information we have.\n\n"
            else:
                # No specific questions, provide general guidance
                followup_context = f"""Thank you for that initial assessment. Let me provide some additional context to help us collaborate better:

1. The user's query was: "{query}"
2. This appears to be a {agent_name.replace('_', ' ')}-related question, which is why I consulted you.
3. Based on my memory of interactions with the user, they value concise but complete answers.

With this additional context, can you provide more detailed insights or recommendations? Feel free to make reasonable assumptions if needed, while being transparent about what you know vs. what you're inferring.
"""

            # Add EA's follow-up to the conversation
            conversation.append({"role": "EA", "content": followup_context})

            # Display EA's follow-up message
            try:
                OutputManager.print_consultation_message("EA", followup_context)
            except:
                OutputManager.print_info("EA is providing additional context")

            # Variables for the follow-up response
            follow_up_response = ""

            # Callback for the follow-up
            def process_follow_up(message):
                nonlocal follow_up_response

                if message["type"] == "response":
                    if "response" in message and message["response"]:
                        response_chunk = message["response"]
                        follow_up_response += response_chunk

                        # Show streaming chunks in a more visible way
                        try:
                            if not message.get("is_final", False):
                                # Use direct sys.stdout for more reliable streaming
                                sys.stdout.write(
                                    f"{HIGHLIGHT_COLOR}{response_chunk}{RESET}"
                                )
                                sys.stdout.flush()
                            else:
                                # Add newline at the end of full response
                                print()
                                # Print an indicator that a full response was received
                                OutputManager.print_info(
                                    f"Received follow-up response from {agent_name}"
                                )
                        except Exception as e:
                            # Fallback if streaming fails
                            if message.get("is_final", False):
                                OutputManager.print_info(
                                    f"{agent_name} has responded to follow-up"
                                )
                                # Also print the full response as fallback
                                print(f"{HIGHLIGHT_COLOR}{follow_up_response}{RESET}")

            # Send the follow-up
            self.registry.send_request_to_agent(
                agent_name, followup_context, process_follow_up
            )

            # Add agent's follow-up response to the conversation
            conversation.append({"role": agent_name, "content": follow_up_response})

            # Update the full agent response to include the follow-up
            agent_response += f"\n\n[Follow-up Exchange]\nEA: {followup_context}\n\n{agent_name}: {follow_up_response}"

        # Final summary from the EA to wrap up the consultation
        ea_summary = f"""Thank you for your expertise on this matter. To summarize our consultation:

1. The user asked: "{query}"
2. You provided specialized knowledge as the {agent_name}
3. We worked together to address the query with both general context and domain expertise

I'll incorporate this collaborative insight into my response to the user, acknowledging your contribution while maintaining my role as their Executive Assistant."""

        # Add EA's summary to the conversation
        conversation.append({"role": "EA", "content": ea_summary})

        # Display EA's summary
        try:
            OutputManager.print_consultation_message("EA", ea_summary)
        except:
            OutputManager.print_info("EA is summarizing the consultation")

        # End the consultation display
        try:
            OutputManager.print_internal_consultation_end()
        except:
            OutputManager.print_info(f"[End of consultation with {agent_name}]")

        # Format the full conversation
        full_conversation = ""
        for message in conversation:
            full_conversation += f"{message['role']}: {message['content']}\n\n"

        # Store the consultation in memory
        self.memory_manager.add_observation(
            {
                "role": "agent_consultation",
                "agent": agent_name,
                "query": query,
                "response": agent_response,
                "conversation": full_conversation,
                "timestamp": OutputManager.format_timestamp(),
            }
        )

        return full_conversation if return_full_conversation else agent_response

    def handle_auto_delegation(self, user_query):
        """Handle automatic delegation of queries to specialist agents."""
        start_time = time.time()

        # Get memory context that might be relevant to this query
        memory_context = self.memory_manager.get_memory_for_prompt(user_query)

        # Quick check for "what can you do" style queries - always handled by EA
        lower_query = user_query.lower()
        if any(
            phrase in lower_query
            for phrase in [
                "what can you do",
                "what can u do",
                "what do you do",
                "what are you capable",
                "capabilities",
                "features",
                "help me",
                "how do you work",
            ]
        ):
            OutputManager.print_info(
                "This query is about capabilities and will be handled by the EA directly."
            )
            self.current_agent = None
            self.consecutive_agent_queries = 0
            self.handle_directly(user_query, memory_context)
            return

        # If we have a current active agent, we should first check if we should continue with it
        if self.current_agent and self.current_agent in self.registry.agent_processes:
            # Let the router evaluate if the query is still relevant to the current agent
            OutputManager.print_info(
                f"Checking if query is still relevant for {self.current_agent}..."
            )

            # Get specialized prompt for evaluating agent relevance
            eval_prompt = self.router.create_agent_relevance_prompt(
                self.current_agent, user_query
            )

            # Get routing evaluation with the special prompt
            evaluation = self.router.route_query(
                user_prompt=user_query,
                memory_context=memory_context,
                system_prompt=eval_prompt,
                force_json=True,
            )

            # Check if we should continue with the current agent context
            if (
                isinstance(evaluation, dict)
                and "agent" in evaluation
                and evaluation["agent"].lower() == self.current_agent.lower()
                and evaluation.get("confidence", 0) > 0.8
            ):
                # Continue with the current agent context
                OutputManager.print_info(
                    f"Continuing with {self.current_agent} context..."
                )
                self.consecutive_agent_queries += 1

                # NEW APPROACH: Instead of delegating, EA consults with the agent
                agent_response = self.consult_with_agent(self.current_agent, user_query)

                # Now have the EA respond with the agent's expertise incorporated
                ea_prompt = f"""
User query: {user_query}

I just had a collaborative conversation with the {self.current_agent} about this query. Here's what I learned:

{agent_response}

Please craft a response that:
1. Acknowledges that you consulted with the {self.current_agent} for their specialized expertise
2. Integrates their insights with your own general knowledge
3. Presents a unified, helpful answer that combines both perspectives
4. Maintains your role as the Executive Assistant while giving credit to the specialist's contribution

Your response should feel natural and conversational, not just repeating what the specialist said.
"""
                # Handle with the EA, including both memory context and agent consultation
                combined_context = (
                    f"{memory_context if memory_context else ''}\n\n{ea_prompt}"
                )
                self.handle_directly(user_query, combined_context)
                return
            else:
                # Switch away from the current agent
                reasoning = evaluation.get("reasoning", "Not specialized enough")
                OutputManager.print_info(
                    f"Query no longer relevant to {self.current_agent} context: {reasoning}"
                )
                # Reset agent tracking
                previous_agent = self.current_agent
                self.current_agent = None
                self.consecutive_agent_queries = 0

                # Handle with EA directly
                OutputManager.print_info(
                    "Query will be handled by the Executive Assistant"
                )
                self.handle_directly(user_query, memory_context)
                return

        # Only reach here for the first query or if we haven't returned after agent switching
        # Perform first-pass evaluation to see if EA should handle this query
        should_delegate = self._should_delegate_query(user_query)

        if not should_delegate:
            # EA should handle this query directly based on first-pass check
            OutputManager.print_info(
                "Query will be handled directly by the Executive Assistant"
            )
            self.handle_directly(user_query, memory_context)
            return

        # Only perform routing if first-pass indicates potential delegation
        OutputManager.print_info("Analyzing query for specialist consultation...")

        result = self.router.route_query(
            user_prompt=user_query, memory_context=memory_context
        )

        # Extract the routed agent from the result
        if isinstance(result, dict) and "agent" in result:
            agent = result["agent"].lower()
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "No reasoning provided")

            routing_time = time.time() - start_time

            # Print debugging information if verbose
            if self.router_verbose:
                OutputManager.print_info(
                    f"Router selected: {agent} (confidence: {confidence:.2f})"
                )
                OutputManager.print_info(f"Reasoning: {reasoning}")
                OutputManager.print_info(f"Routing took {routing_time:.2f} seconds")

            # Special case for 'none' agent (means EA should handle it directly)
            if agent == "none":
                OutputManager.print_info(
                    "Query will be handled directly by the Executive Assistant."
                )
                self.current_agent = None
                self.consecutive_agent_queries = 0
                self.handle_directly(user_query, memory_context)
                return

            # Check if the agent exists AND has high confidence
            if agent in self.registry.agent_processes and confidence >= 0.85:
                # Update current agent for sticky delegation
                self.current_agent = agent
                self.consecutive_agent_queries = 1

                # NEW APPROACH: Instead of delegating, EA consults with the agent
                agent_response = self.consult_with_agent(agent, user_query)

                # Now have the EA respond with the agent's expertise incorporated
                ea_prompt = f"""
User query: {user_query}

I just had a collaborative conversation with the {self.current_agent} about this query. Here's what I learned:

{agent_response}

Please craft a response that:
1. Acknowledges that you consulted with the {self.current_agent} for their specialized expertise
2. Integrates their insights with your own general knowledge
3. Presents a unified, helpful answer that combines both perspectives
4. Maintains your role as the Executive Assistant while giving credit to the specialist's contribution

Your response should feel natural and conversational, not just repeating what the specialist said.
"""
                # Handle with the EA, including both memory context and agent consultation
                combined_context = (
                    f"{memory_context if memory_context else ''}\n\n{ea_prompt}"
                )
                self.handle_directly(user_query, combined_context)
            else:
                # If confidence is too low or agent doesn't exist, fall back to EA
                OutputManager.print_info(
                    f"Agent '{agent}' has insufficient confidence ({confidence:.2f}). Handling directly."
                )
                # Fall back to EA handling the query directly
                self.current_agent = None
                self.consecutive_agent_queries = 0
                self.handle_directly(user_query, memory_context)
        else:
            OutputManager.print_error("Failed to route query. Handling directly.")
            self.current_agent = None
            self.consecutive_agent_queries = 0
            self.handle_directly(user_query, memory_context)

    def delegate_to_agent(self, agent_name, query, memory_context=None):
        """Delegate a query to a specific agent."""
        # Prepare response time tracking
        start_time = time.time()

        # Connect memory context if available
        if memory_context:
            # Add memory context to the query
            enhanced_query = f"{query}\n\n{memory_context}"
        else:
            enhanced_query = query

        # Update agent system prompt with redirection instructions
        agent_system_override = f"""
### Chain of Thought Reasoning Process
As a specialized {agent_name}, follow this reasoning process for each query:

1. **Question Analysis**: Carefully break down what's being asked. Identify the core request, constraints, and required domain knowledge.
2. **Expertise Evaluation**: Assess whether this query falls within your specialized domain as a {agent_name}.
3. **Approach Planning**: If within your expertise, outline your approach to answering the query.
4. **Expertise Boundary Check**: If you determine this query falls outside your domain, explain why the Executive Assistant should handle it instead.

If you encounter a query that is clearly outside your area of expertise, add the token "[REDIRECT_TO_EA]" at
the END of your response, followed by a brief explanation of why this query should be handled by the Executive Assistant instead.

For example: "I don't have enough information to answer questions about general user preferences. [REDIRECT_TO_EA] This query would be better handled by the Executive Assistant because it relates to user context that I don't have access to."

Only use this token if you're confident the query is not relevant to your expertise as a {agent_name}.
Otherwise, provide the most helpful response you can within your domain of expertise.
"""

        # Add system prompt override to the query
        enhanced_query = (
            f"{enhanced_query}\n\n[SYSTEM_PROMPT_ADDITION]\n{agent_system_override}"
        )

        # Display thinking animation
        OutputManager.print_info(f"Delegating to {agent_name}...")

        # Print the agent response prefix for the agent's response
        OutputManager.print_agent_response_prefix(agent_name)

        # Add a clear streaming indicator
        OutputManager.print_info("Starting response stream from agent...")
        OutputManager.print_divider(40)

        # Temporary variable to collect full response
        full_response = ""
        redirect_to_ea = False

        # Callback for processing the streaming response
        def process_response(message):
            nonlocal full_response

            if message["type"] == "response":
                # Print the response chunk if there is one
                if "response" in message and message["response"]:
                    response_chunk = message["response"]
                    # Print the response without a newline
                    OutputManager.print_response(response_chunk, end="")

                    # Add to full response
                    full_response += response_chunk

                    # Store the response in memory if it's not an error
                    if not message.get("is_error", False):
                        self.memory_manager.add_observation(
                            {
                                "role": "assistant",
                                "content": response_chunk,
                                "agent": agent_name,
                                "timestamp": OutputManager.format_timestamp(),
                            }
                        )

                # Add a newline after the final message
                if message.get("is_final", False):
                    OutputManager.print_response("", end="\n")
                    OutputManager.print_divider(40)
                    OutputManager.print_info(f"Response from {agent_name} complete")

            elif message["type"] == "status":
                # Handle status updates
                if message.get("status") == "starting":
                    OutputManager.print_info(f"Agent {agent_name} is processing...")
                elif message.get("status") == "completed":
                    # This is handled by the response time output below
                    pass

        # Send the query to the agent
        response = self.registry.send_request_to_agent(
            agent_name, enhanced_query, process_response
        )

        # Calculate and display response time
        response_time = time.time() - start_time
        OutputManager.print_info(f"Response completed in {response_time:.2f} seconds")

        # Check if agent indicated redirection back to EA
        redirect_match = re.search(r"\[REDIRECT_TO_EA\](.*?)(?:$|\.)", full_response)
        if redirect_match:
            # Extract the reasoning after the redirect token
            redirect_text = redirect_match.group(1).strip()
            OutputManager.print_info(
                f"Agent indicated redirection to EA: {redirect_text}"
            )

            # Reset agent tracking
            self.current_agent = None
            self.consecutive_agent_queries = 0

            # Get memory context and handle with EA
            memory_context = self.memory_manager.get_memory_for_prompt(query)

            # Create context about the redirection for EA's response
            ea_prompt = f"""
The {agent_name} indicated they don't have the expertise for this query: "{query}"

Their reasoning was: "{redirect_text}"

Please respond to the user's query directly, acknowledging that you considered consulting with the {agent_name} but determined you could better address this yourself. Provide a helpful response that draws on your general knowledge and understanding of the user's needs.
"""
            # Handle with the EA, including both memory context and redirection context
            combined_context = (
                f"{memory_context if memory_context else ''}\n\n{ea_prompt}"
            )
            self.handle_directly(query, combined_context)

            return full_response

        return response
