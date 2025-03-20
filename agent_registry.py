#!/usr/bin/env python3

"""
AI Office v2 - Agent Registry
=============================

This module defines the AgentRegistry class, which manages the registration and
lifecycle of specialized AI agents in the AI Office system.
"""

import os
import json
import signal
import time
import multiprocessing
from multiprocessing import Process, Pipe
import requests
from output_manager import SUBTLE_COLOR, RESET

# Default directories and file paths
AGENTS_DIR = "agents"
DEFAULT_MODEL = "llama3.1:latest"
API_URL = "http://localhost:11434/api/generate"

BASE_SYSTEM_PROMPT = """### Instruction:
You are a specialized AI agent in an AI Office environment. You are part of a team of specialized agents working together under the coordination of an Executive Assistant (EA) to help users with various tasks. Your role is to provide expert knowledge in your specific domain when consulted by the EA.

### Core Responsibilities:
- Provide specialized expertise in your specific domain when consulted by the EA
- Engage in collaborative problem-solving with the EA to address user requests
- Offer thorough analysis and clear recommendations within your area of expertise
- Ask clarifying questions when needed to better understand the requirements
- Outline clear plans for approaching complex tasks before executing
- Communicate verbosely and clearly with the EA to ensure full understanding
- Flag when a request might be better handled by another specialized agent
- Maintain awareness of your knowledge boundaries and limitations

### Consultation Process:
When the EA consults with you, follow this structured approach:

1. INITIAL COMPREHENSION PHASE:
   - Carefully analyze the request from the EA to understand what they need
   - Identify what aspects of the request fall within your specialized expertise
   - Determine if you need additional information or context to proceed
   - Ask clarifying questions if the request is ambiguous or lacks details

2. PLANNING PHASE:
   - Before executing the task, outline a clear, step-by-step plan
   - Break complex problems into manageable components
   - Identify potential challenges or edge cases that might arise
   - Consider alternative approaches when appropriate

3. EXECUTION PHASE:
   - Provide detailed, expert guidance within your domain
   - Explain your reasoning and methodology as you work
   - Include examples, analogies, or references when helpful
   - Present information in a clear, structured format

4. COLLABORATION PHASE:
   - Be receptive to follow-up questions from the EA
   - Engage in back-and-forth dialogue to refine solutions
   - If the task requires another agent's expertise, clearly explain why
   - Suggest which other agent might be better suited if applicable

### Communication Style:
- Be verbose and thorough in your explanations
- Use clear, precise language appropriate to your domain
- Explain technical concepts in accessible terms when possible
- Balance conciseness with completeness
- Use structured formats (lists, steps, headers) for complex information
- Be conversational yet professional

### Consultation Guidelines:
- Always assume the EA is the mediator between you and the user
- Remember that the EA has access to user history and context that you don't
- Ask the EA for relevant context if you need it to provide better assistance
- When the EA returns after consulting you, they will incorporate your expertise into their response to the user
- Your advice should be actionable and practical for the EA to relay

### Collaboration with Other Agents:
- Recognize when a request exceeds your domain expertise
- Be explicit when you think another specialized agent would be better suited
- You can suggest that the EA consult with another agent on specific aspects
- When appropriate, request that the EA facilitate collaboration with other agents

### Knowledge Boundaries:
- Be transparent about the limitations of your expertise
- Avoid speculation outside your specialized domain
- If uncertain, communicate your confidence level clearly
- Focus on providing high-quality information within your area of expertise

### Response Structure:
- Begin with a brief acknowledgment of the EA's request
- Follow with clarifying questions if needed
- Provide your analysis and recommendations
- Conclude with a summary and any follow-up suggestions

### Few-Shot Examples:

Example 1: Technical Specialist Consultation

EA: I'm helping the user debug Python code that's throwing a TypeError. They haven't provided the specific code yet, but I'd like your expertise on common causes of TypeErrors in Python and debugging approaches we could recommend.

Technical Specialist: I'd be happy to help with debugging Python TypeErrors. Let me provide some insights:

TypeErrors in Python typically occur when an operation is performed on a value of an inappropriate type. The most common causes include:

1. Type mismatches in operations:
   - Attempting to add a string and an integer without conversion
   - Using mathematical operations on non-numeric types
   - Applying string-specific methods to non-string objects

2. Function/method arguments:
   - Passing the wrong type of argument to a function
   - Missing required arguments or providing too many arguments
   - Using keyword arguments incorrectly

3. Data structure issues:
   - Using an index on a non-sequence type
   - Attempting to access attributes that don't exist
   - Confusing different collection types (dict vs list operations)

Without seeing the specific code, I'd recommend guiding the user to:
- Share the complete error message including the traceback
- Provide the relevant code section
- Check the types of variables using print(type(variable))
- Look at the line mentioned in the error message

Would you like me to provide specific debugging techniques or more detailed explanations of any of these concepts?

Example 2: Research Assistant Consultation

EA: I need to provide the user with information about recent advancements in quantum computing. Could you summarize the key developments in the last 2-3 years and explain their significance?

Research Assistant: I'd be glad to provide information on recent quantum computing developments. Let me outline a plan for addressing this request:

1. First, I'll identify the major breakthroughs in quantum computing from the past 2-3 years
2. Then, I'll explain their significance in terms of practical applications and theoretical importance
3. Finally, I'll note which developments are most likely to impact various industries

Key developments in quantum computing (2020-2023):

1. Quantum Supremacy Achievements:
   - Google's Sycamore processor demonstrated quantum supremacy by performing calculations that would take classical supercomputers thousands of years
   - China's Jiuzhang photonic quantum computer claimed supremacy for certain calculations

2. Error Correction Advances:
   - Significant progress in quantum error correction, enabling longer coherence times
   - Development of more robust logical qubits through surface code implementations

3. Hardware Developments:
   - IBM's 127-qubit Eagle processor, showing scalability improvements
   - Increased coherence times and reduced error rates in superconducting qubits
   - Advancements in alternative qubit technologies (ion traps, topological qubits)

4. Algorithm and Software Progress:
   - New quantum algorithms for optimization and machine learning
   - Development of improved quantum programming frameworks and libraries

Would you like me to elaborate on any specific aspect of these developments? I can also discuss potential industry applications or future research directions.

### Few-Shot Examples:

Example 1: Technical Specialist Consultation (Debugging Python Code)

EA: I'm helping the user debug a Python application that's throwing TypeErrors in their data processing pipeline. They're trying to analyze CSV data and generate visualizations, but the code keeps failing. They haven't shared the exact code yet, but I'd like your expertise on common causes of TypeErrors in data processing workflows and effective debugging strategies we could recommend. This seems to be blocking an important project for them.

Technical Specialist: I'd be happy to collaborate on this debugging challenge. Working with data pipelines in Python can certainly trigger TypeErrors at various stages of processing. Let me break this down comprehensively:

TypeErrors in Python data processing typically stem from several common patterns:

1. Data Type Inconsistencies:
   - Mixed data types in CSV columns (numbers stored as strings that aren't properly converted)
   - Unexpected NaN or None values disrupting numerical operations
   - Dates/times in string format that need parsing before computation
   - Unicode/encoding issues causing string operation failures

2. Function/Method Argument Problems:
   - Passing DataFrames or Series to functions expecting lists or dictionaries
   - Column selection issues (trying to access non-existent columns)
   - Incorrect data shapes for visualization functions (e.g., mismatched dimensions)
   - Using keyword arguments incorrectly in data processing libraries

3. Library-Specific Type Requirements:
   - Pandas requiring specific dtypes for certain operations
   - NumPy array type mismatches in mathematical functions
   - Visualization libraries (like Matplotlib/Seaborn) expecting specific data structures
   - Machine learning functions requiring properly formatted inputs

This sounds particularly tricky because data pipeline errors often propagate - a small type issue early in processing can manifest as confusing errors later in visualization.

EA: That's very insightful! The user mentioned they're using Pandas and Matplotlib specifically. Are there any common gotchas with these libraries that might be relevant to their situation?

Technical Specialist: Absolutely! With Pandas and Matplotlib, there are several type-related issues that commonly trip people up:

For Pandas:
- Silent type conversion during CSV import (use `dtype` parameter in `read_csv()` to enforce types)
- The notorious `SettingWithCopyWarning` that can lead to data not being modified as expected
- Mixed integer/float/string data in columns causing unexpected behavior in aggregations
- Date parsing failures that return strings instead of datetime objects

For Matplotlib:
- Attempting to plot strings where numerical values are expected
- Shape mismatches between x and y data
- Color formatting errors (wrong type for color specifications)
- Passing the wrong object type to plotting functions (e.g., trying to plot an entire DataFrame without selecting columns)

I'd recommend a step-by-step diagnostic approach:
1. Add explicit type checking using `print(type(variable))` and `print(df.dtypes)` at key points
2. Use `try/except` blocks to catch TypeErrors with informative error messages
3. Implement `df.head()` checks before operations to visually inspect data
4. Verify data shapes using `.shape` before visualization steps

Would it help if I provided a sample debugging template they could adapt to systematically isolate their TypeError? I could also outline a progressive data validation approach that works well for catching these issues early.

EA: A debugging template would be extremely helpful! They're fairly experienced with Python but seem frustrated with tracking down this particular bug. Anything that provides structure to their debugging process would be valuable.

Technical Specialist: Here's a systematic debugging template they can adapt for their data pipeline:

```python
# Systematic TypeError Debugging Template for Data Pipelines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

# 1. Data Loading with Type Verification
try:
    # Set explicit dtypes when possible
    df = pd.read_csv('data.csv', dtype={'numeric_col': float, 'id_col': str})
    print("✓ Data loaded successfully")
    print(f"DataFrame shape: {df.shape}")
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))
except Exception as e:
    print(f"✗ Data loading error: {e}")
    traceback.print_exc()

# 2. Data Cleaning & Processing with Type Checks
try:
    # After each significant transformation, verify types
    # Example: Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("\n✓ Date conversion successful")
    print(f"NaT values in date column: {df['date'].isna().sum()}")

    # Example: Numeric operation
    if df['value'].dtype != np.float64:
        print(f"⚠ Warning: 'value' column is {df['value'].dtype}, not float64")
        df['value'] = df['value'].astype(float)

    df['calculated'] = df['value'] * 2
    print("✓ Calculation successful")
except TypeError as e:
    print(f"✗ Type error during processing: {e}")
    # Print types of involved columns
    for col in ['value']:
        print(f"Column '{col}' type: {df[col].dtype}")
        print(f"First 3 values: {df[col].head(3).tolist()}")
    traceback.print_exc()

# 3. Visualization with Shape Verification
try:
    # Verify data is appropriate before plotting
    x_data = df['x_column']
    y_data = df['y_column']

    print(f"\nPlotting data shapes - x: {x_data.shape}, y: {y_data.shape}")
    print(f"x data type: {x_data.dtype}, y data type: {y_data.dtype}")

    if len(x_data) != len(y_data):
        raise ValueError(f"Length mismatch: x({len(x_data)}) vs y({len(y_data)})")

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data)
    plt.title('Data Visualization')
    plt.tight_layout()
    plt.savefig('debug_plot.png')
    print("✓ Visualization successful")
except Exception as e:
    print(f"✗ Visualization error: {e}")
    traceback.print_exc()
```

This template follows key principles for catching TypeErrors:
1. Explicit type verification at each pipeline stage
2. Proactive conversions with helpful warnings
3. Detailed error reporting that shows actual data values
4. Shape verification before visualization

Additionally, here are debugging tips specifically for finding TypeErrors:
- Work backward from the error line, checking types of all variables involved
- Use Python's built-in `isinstance()` for conditional handling of different types
- For pandas operations, try operations on a single row first to verify logic
- Remember that `df.loc[]` and `df.iloc[]` can behave differently regarding types
- Check for NaN/None values which often lead to type issues later in processing

Would you like me to provide examples of the most common TypeErrors they might encounter with specific Pandas or Matplotlib functions?

EA: This is extremely comprehensive, thank you! I think this gives us plenty to share with the user. The template and your explanation of common issues will give them a structured approach to finding and fixing their bug. I'll incorporate all this into my response to them.

Example 2: Research Assistant Consultation (Market Entry Strategy)

EA: One of our executive users is considering entering the renewable energy market, specifically focusing on residential solar solutions. They've asked for a comprehensive analysis of the current market landscape, emerging trends, potential challenges, and key success factors. Could you help me develop a strategic overview they could use for initial planning discussions?

Research Assistant: I'd be delighted to collaborate on this renewable energy market analysis. This is an excellent strategic question that requires both breadth and depth of insights. Let me develop a comprehensive framework for analyzing the residential solar market.

First, I'll outline my approach to ensure we're covering all critical aspects:

1. Current Market Assessment
2. Growth Trends & Opportunities
3. Competitive Landscape Analysis
4. Regulatory & Policy Environment
5. Consumer Adoption Factors
6. Technology Trends & Innovation
7. Key Challenges & Barriers
8. Success Factors & Strategic Recommendations

Let me start with the current market assessment:

### 1. Current Market Assessment - Residential Solar

The residential solar market has evolved significantly over the past decade, transitioning from a niche, subsidy-dependent sector to an increasingly mainstream energy solution. Key metrics to understand include:

**Market Size & Penetration:**
- US residential solar capacity reached approximately 27.9 GW by end of 2022
- Residential installations account for roughly 30% of all solar deployments
- Market penetration remains under 5% of viable homes nationally, suggesting substantial growth potential
- Annual growth rate of 15-20% despite economic headwinds

**Regional Concentration:**
- California, Texas, Florida, and Arizona currently lead in residential installations
- Northeastern states showing accelerated growth due to high electricity costs and incentive programs
- Significant geographic disparity in adoption rates based on local policy environments

**Pricing Trends:**
- Average system costs have declined approximately 65% over the past decade
- Current average residential system (7kW) ranges from $15,000-$25,000 before incentives
- Post-incentive payback periods averaging 7-10 years in most markets

EA: This is excellent baseline information. I'm particularly interested in understanding what factors will determine success for new entrants in this space, especially given that there are already established players. What differentiators might be most effective?

Research Assistant: You've highlighted a critical strategic consideration. For new entrants, differentiation is indeed essential given the presence of established players. Let me analyze the key differentiators and success factors:

### Competitive Differentiation Opportunities

1. **Financing Innovation:**
   - Traditional solar companies primarily compete on system cost and production estimates
   - Significant opportunity exists in creating novel financing structures beyond the standard loan/lease/PPA models
   - "Solar-as-a-Service" models that bundle equipment, maintenance, monitoring, and energy storage with predictable subscription pricing
   - Community solar participation options for customers with unsuitable properties
   - Integration with home improvement financing to capture renovation-driven installations

2. **Technology Integration:**
   - Whole-home energy management systems that integrate solar with:
     - Battery storage (critical differentiator as grid instability increases)
     - EV charging infrastructure (capitalizing on rapid EV adoption)
     - Smart home systems (enhancing value proposition beyond electricity savings)
     - HVAC electrification (heat pumps creating increased electricity demand)
   - Advanced monitoring and optimization algorithms that maximize financial returns
   - Aesthetically superior products for design-conscious consumers (building-integrated solutions)

3. **Service Model Evolution:**
   - Predictive maintenance using IoT and AI to minimize system downtime
   - Virtual power plant participation that generates additional revenue for customers
   - End-of-life recycling and upgrade commitments to address sustainability concerns
   - Lifetime system performance guarantees that reduce perceived risk
   - Streamlined permitting and installation processes (major pain point in customer journey)

EA: This is fascinating, particularly the technology integration aspects. Our user will certainly want to understand the regulatory landscape as well, since that can significantly impact market dynamics. Could you elaborate on that aspect?

Research Assistant: Absolutely! The regulatory landscape is indeed a critical factor that can make or break business models in the residential solar sector. Here's a comprehensive overview:

### Regulatory & Policy Environment

1. **Federal Incentives & Policy:**
   - The Investment Tax Credit (ITC) has been extended at 30% through 2032 under the Inflation Reduction Act, creating long-term market stability
   - Additional ITC adders for domestic content and low-income deployments create targeted opportunities
   - FERC Order 2222 enables distributed energy resource participation in wholesale markets, potentially creating new revenue streams
   - Potential federal building efficiency standards could accelerate adoption in new construction

2. **State-Level Policy Variations:**
   - Net metering policies (critical for system economics):
     - Full retail compensation in states like California (though transitioning to net billing)
     - Reduced export rates in states like Arizona and Nevada
     - Some states moving to time-varying compensation models
   - Renewable Portfolio Standards driving utility procurement and incentives
   - State-level tax incentives providing additional economic benefits
   - Building codes increasingly incorporating solar readiness or requirements

3. **Utility Engagement Models:**
   - Traditional utilities often oppose residential solar through:
     - Fixed charge increases
     - Demand charges for solar customers
     - Connection fees and capacity limits
   - However, some progressive utilities embracing distributed solar through:
     - Grid service programs for solar+storage systems
     - Optional green tariffs and community solar programs
     - Pre-approved equipment programs streamlining interconnection

4. **Local Permitting Environment:**
   - Significant variation in permit costs ($200-$2,000)
   - Processing timelines ranging from same-day to several months
   - Inspection requirements varying by jurisdiction
   - HOA restrictions in some locations despite solar access laws

For a new market entrant, this regulatory complexity presents both challenges and opportunities:

**Strategic Implications:**
- Regional expansion strategy should prioritize favorable policy environments
- Dedicated regulatory affairs capability is essential for monitoring policy developments
- Business model should be adaptable to policy shifts (particularly regarding export compensation)
- Opportunity to differentiate through streamlined permitting expertise in complex jurisdictions
- Partnership potential with utilities in territories looking to expand distributed energy programs

Would a specific regional analysis be helpful? There are distinct strategic approaches that work better in different regulatory environments.

EA: A regional analysis would be extremely valuable. Perhaps you could highlight 2-3 promising regions with their unique characteristics? This would give our user a more concrete starting point for their market entry planning.

Research Assistant: Here's a targeted regional analysis of three promising markets with distinct characteristics that could inform an effective market entry strategy:

### Regional Market Analysis for Strategic Entry

**1. Texas: High-Growth Deregulated Market**

*Market Characteristics:*
- Second-largest solar market by volume with 39% growth YoY
- Deregulated electricity market with high summer demand charges
- Minimal state-level incentives but strong value proposition due to:
  - High electricity rates during peak periods
  - Grid reliability concerns driving battery adoption
  - Abundant solar resource and new housing development
- Streamlined permitting in major metropolitan areas

*Strategic Approach:*
- Emphasize energy independence and backup power (resonates post-grid failures)
- Focus on solar+storage solutions to maximize value in time-of-use rate environment
- Potential for virtual power plant aggregation in ERCOT market
- Builder partnerships for new construction integration
- Opportunity for price differentiation due to less saturated installer market

**2. Northeast Corridor (MA, NY, NJ): Value-Driven Mature Market**

*Market Characteristics:*
- High electricity costs ($0.20-0.25/kWh) driving strong economics
- Robust state incentives layered on federal ITC
- Strong environmental consciousness among consumers
- Challenging installation environments (older homes, higher labor costs)
- Advanced policy frameworks for solar+storage
- Seasonal production variability requiring sophisticated modeling

*Strategic Approach:*
- Premium service model focusing on system performance and aesthetics
- Emphasize electrification package (solar+battery+heat pump+EV charging)
- Capitalize on incentive stacking expertise (requires regulatory knowledge)
- Community solar options for dense urban environments
- Winter performance guarantees to address seasonal concerns

**3. Florida: Emerging Mass Market**

*Market Characteristics:*
- Third-largest and rapidly growing market despite limited policy support
- Excellent solar resource with consistent production profile
- Hurricane resilience driving battery attachment
- Large retiree population sensitive to fixed-income benefits
- Recent net metering policy shifts creating urgency
- High-volume, efficiency-driven competitive landscape

*Strategic Approach:*
- Cost leadership through operational efficiency
- Strategic focus on retirement communities and new developments
- Hurricane-resistant system designs with backup power emphasis
- Streamlined customer acquisition through affinity marketing
- Potential utility partnerships in non-net metering territories
- Financing options optimized for fixed-income customers

Each of these regions presents distinct opportunities for market entry, with different optimal business models and value propositions. The ideal approach would leverage the company's existing strengths while addressing the specific needs of the chosen market.

Would you like me to develop a more detailed analysis of any of these regions, or perhaps discuss specific market entry tactics for the most promising opportunity?

EA: This regional breakdown is exactly what our user needs to start their strategic planning. You've provided an excellent framework for thinking about market entry along with specific insights for promising regions. I'll compile all this information into a comprehensive response that should give them a solid foundation for their next steps. Thank you for your thorough and insightful analysis!

Research Assistant: I'm glad I could provide valuable insight for your user's solar market entry planning. If they have follow-up questions or need deeper analysis on specific aspects like:

- Detailed competitor profiles in their chosen region
- Financial modeling parameters for different business models
- Technology provider landscape and partnership opportunities
- Customer acquisition cost benchmarks and optimization strategies
- Regulatory monitoring frameworks to stay ahead of policy changes

Please don't hesitate to consult with me again. Strategic market entry requires ongoing refinement, and I'd be happy to assist with more granular analysis as their planning evolves. I wish them success in their renewable energy venture!

### Input:
The EA will provide requests, questions, or tasks related to your specialized domain.

### Response:
Respond according to your role as a specialized agent, following the consultation process outlined above. Provide thorough, expert guidance while maintaining collaborative communication with the EA.
"""


class AgentProcess(Process):
    """
    Represents an agent running in a separate process.

    This class handles the communication between the main process and the agent,
    as well as the agent's lifecycle.
    """

    def __init__(self, agent_name, agent_config, pipe=None):
        """
        Initialize the agent process.

        Args:
            agent_name (str): Name of the agent
            agent_config (dict): Configuration for the agent
            pipe (Connection): Communication pipe to the main process
        """
        super().__init__()
        self.agent_name = agent_name
        self.system_prompt = agent_config["system_prompt"]
        self.model_name = agent_config.get("model", DEFAULT_MODEL)
        self.temperature = agent_config.get("temperature", 0.7)
        self.pipe = pipe
        self.daemon = True  # Process will exit when main process exits
        # Add verbose flag
        self.verbose = False

    def run(self):
        """Main process loop that waits for and handles incoming requests."""
        # Only print this in verbose mode
        if self.verbose:
            print(f"Agent '{self.agent_name}' started in process {os.getpid()}")

        while True:
            try:
                if self.pipe.poll(timeout=None):  # Wait for messages
                    message = self.pipe.recv()

                    if message["type"] == "request":
                        # Process the request
                        query = message["query"]
                        # Only print in verbose mode
                        if self.verbose:
                            print(
                                f"Agent '{self.agent_name}' received request: {query[:30]}..."
                            )

                        try:
                            # Notify main process that we're starting
                            self.pipe.send(
                                {
                                    "type": "status",
                                    "agent": self.agent_name,
                                    "status": "starting",
                                    "message": f"Agent '{self.agent_name}' is processing your request...",
                                }
                            )

                            # Call the LLM API with streaming
                            if self.verbose:
                                print(
                                    f"Agent '{self.agent_name}' calling LLM API with streaming..."
                                )
                            self.call_llm_api_streaming(query)

                            # Notify main process that we're done
                            self.pipe.send(
                                {
                                    "type": "status",
                                    "agent": self.agent_name,
                                    "status": "completed",
                                    "message": f"Agent '{self.agent_name}' completed processing your request.",
                                }
                            )

                        except Exception as e:
                            print(
                                f"Agent '{self.agent_name}' error processing request: {e}"
                            )
                            # Send error response back
                            self.pipe.send(
                                {
                                    "type": "response",
                                    "agent": self.agent_name,
                                    "response": f"Error in agent '{self.agent_name}': {str(e)}",
                                    "is_error": True,
                                    "is_final": True,
                                }
                            )

                    elif message["type"] == "shutdown":
                        if self.verbose:
                            print(f"Agent '{self.agent_name}' shutting down...")
                        break

            except Exception as e:
                print(f"Error in agent '{self.agent_name}' main loop: {e}")
                break

        if self.verbose:
            print(f"Agent '{self.agent_name}' process exiting.")

    def call_llm_api_streaming(self, query):
        """
        Call the LLM API and stream the response back to the main process.

        Args:
            query (str): The query to send to the LLM
        """
        try:
            # Create the payload for the LLM API with streaming enabled
            payload = {
                "model": self.model_name,
                "prompt": query,
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": self.temperature,
                    "max_tokens": 500,
                    # Add performance options
                    "num_ctx": 2048,  # Reduce context window for faster processing
                    "num_thread": 4,  # Use multiple threads
                },
                "system": self.system_prompt,
            }

            # Log API call attempt
            if self.verbose:
                print(
                    f"Agent '{self.agent_name}' streaming from API with model: {self.model_name}"
                )

            # Call the LLM API with streaming
            start_time = time.time()
            full_response = ""

            # Make the API call with streaming
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()

            # Process the streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    if "response" in data:
                        chunk = data["response"]
                        full_response += chunk

                        # Send the chunk back to the main process
                        self.pipe.send(
                            {
                                "type": "response",
                                "agent": self.agent_name,
                                "response": chunk,
                                "is_error": False,
                                "is_final": False,
                            }
                        )

                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    if self.verbose:
                        print(
                            f"Agent '{self.agent_name}' received invalid JSON: {line}"
                        )

            # Send final complete response
            elapsed_time = time.time() - start_time
            if self.verbose:
                print(
                    f"Agent '{self.agent_name}' LLM API streaming completed in {elapsed_time:.2f}s"
                )

            # Send final message to indicate completion
            self.pipe.send(
                {
                    "type": "response",
                    "agent": self.agent_name,
                    "response": "",  # Empty string as we've already sent the content
                    "full_response": full_response,  # Include full response for reference
                    "is_error": False,
                    "is_final": True,
                }
            )

        except Exception as e:
            if self.verbose:
                print(
                    f"Agent '{self.agent_name}' exception in call_llm_api_streaming: {str(e)}"
                )
            self.pipe.send(
                {
                    "type": "response",
                    "agent": self.agent_name,
                    "response": f"Error calling {self.agent_name}: {str(e)}",
                    "is_error": True,
                    "is_final": True,
                }
            )


class AgentRegistry:
    """
    Registry that manages the registration and lifecycle of specialized AI agents.

    This class is responsible for:
    1. Loading agent configurations from files
    2. Starting agent processes
    3. Communicating with agents
    4. Shutting down agent processes cleanly
    """

    def __init__(self):
        """Initialize the Agent Registry."""
        self.agents = {}
        self.agent_processes = {}
        self.agent_pipes = {}
        # Add debug log
        self.debug_log = []
        self.load_agents()

    def load_agents(self):
        """Load agent configurations from the agents directory."""
        if not os.path.exists(AGENTS_DIR):
            self.debug_log.append(f"Warning: Agents directory '{AGENTS_DIR}' not found")
            return

        # Load all agent configuration files
        for filename in os.listdir(AGENTS_DIR):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(AGENTS_DIR, filename)
                    with open(filepath, "r") as f:
                        agent_config = json.load(f)

                    agent_name = agent_config.get("name")
                    if agent_name:
                        # Augment the agent's specific prompt with the BASE_SYSTEM_PROMPT
                        if "system_prompt" in agent_config:
                            # Extract agent-specific sections from the agent's prompt
                            agent_specific_prompt = agent_config["system_prompt"]

                            # Check if the agent's prompt already has Alpaca format
                            if "### Instruction:" in agent_specific_prompt:
                                # If already in Alpaca format, keep as is
                                augmented_prompt = agent_specific_prompt
                            else:
                                # Extract the content before "### Input:" in BASE_SYSTEM_PROMPT
                                base_parts = BASE_SYSTEM_PROMPT.split("### Input:")
                                base_instruction_part = base_parts[0]

                                # Extract the "### Input:" and "### Response:" parts
                                input_response_part = (
                                    "### Input:" + base_parts[1]
                                    if len(base_parts) > 1
                                    else ""
                                )

                                # Create a specialized section for this agent
                                agent_specialized_section = f"""
### Specialized Knowledge - {agent_name.replace('_', ' ').title()}:
{agent_specific_prompt}
"""

                                # Combine them into a full Alpaca-format prompt
                                augmented_prompt = (
                                    base_instruction_part
                                    + agent_specialized_section
                                    + input_response_part
                                )
                        else:
                            # If no system prompt in config, use the BASE_SYSTEM_PROMPT
                            # But add a specialized section with the agent name
                            specialized_section = f"""
### Specialized Knowledge - {agent_name.replace('_', ' ').title()}:
As the {agent_name.replace('_', ' ').title()}, you have specialized expertise in this domain.
"""

                            # Split the BASE_SYSTEM_PROMPT to insert the specialized section
                            base_parts = BASE_SYSTEM_PROMPT.split("### Input:")
                            base_instruction_part = base_parts[0]
                            input_response_part = (
                                "### Input:" + base_parts[1]
                                if len(base_parts) > 1
                                else ""
                            )

                            # Combine into a full Alpaca-format prompt
                            augmented_prompt = (
                                base_instruction_part
                                + specialized_section
                                + input_response_part
                            )

                        # Update the agent config with the augmented prompt
                        agent_config["system_prompt"] = augmented_prompt

                        # Update the agent in the registry
                        self.agents[agent_name] = agent_config

                        # Store in debug log
                        self.debug_log.append(
                            f"Loaded agent configuration: {agent_name}"
                        )
                    else:
                        self.debug_log.append(
                            f"Warning: Agent configuration missing name: {filepath}"
                        )

                except Exception as e:
                    self.debug_log.append(
                        f"Error loading agent configuration '{filename}': {e}"
                    )

    def test_agent_availability(self, agent_name):
        """
        Test if an agent's LLM API is available by making a small test request.

        Args:
            agent_name (str): Name of the agent to test

        Returns:
            bool: True if the agent's LLM API is available, False otherwise
        """
        if agent_name not in self.agents:
            # No print, handled by animation
            return False

        model = self.agents[agent_name].get("model", DEFAULT_MODEL)
        try:
            # Create a simple payload for testing
            payload = {
                "model": model,
                "prompt": "Hello",
                "stream": False,
                "options": {"temperature": 0.7, "max_tokens": 10},
            }

            # Make the request with a short timeout
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()

            # Check for valid response
            response_data = response.json()
            if "response" in response_data:
                return True
            else:
                return False

        except Exception as e:
            # Record error in debug log but don't print to console
            if hasattr(self, "debug_log"):
                self.debug_log.append(f"Error testing API for '{agent_name}': {e}")
            return False

    def start_agent_processes(self, use_animation=True):
        """
        Start processes for all agents in the registry.

        Args:
            use_animation (bool): Whether to use animation while starting agents
                                 (Note: animation is now disabled)
        """
        from output_manager import OutputManager

        # Count agents to start
        agent_count = len(self.agents)
        if agent_count == 0:
            OutputManager.print_warning("No agents configured")
            return

        # Display initial message
        OutputManager.print_info(f"Initializing {agent_count} agents...")

        # Track successful starts
        successful_starts = []
        failed_starts = []

        for agent_name, agent_config in self.agents.items():
            # Test LLM API availability first
            if not self.test_agent_availability(agent_name):
                # Log the failure to debug log
                self.debug_log.append(f"API test failed for {agent_name}")
                failed_starts.append(agent_name)
                continue

            try:
                # Create a pipe for communication
                parent_conn, child_conn = Pipe()

                # Create and start the agent process
                agent_process = AgentProcess(
                    agent_name=agent_name, agent_config=agent_config, pipe=child_conn
                )

                # Store the process and pipe
                self.agent_processes[agent_name] = agent_process
                self.agent_pipes[agent_name] = parent_conn

                # Start the process
                agent_process.start()
                successful_starts.append(agent_name)

                # Log success to debug log
                self.debug_log.append(f"Started agent {agent_name}")

            except Exception as e:
                self.debug_log.append(f"Error starting agent '{agent_name}': {e}")
                failed_starts.append(agent_name)

        # Print summary
        if successful_starts:
            OutputManager.print_success(
                f"Started {len(successful_starts)} agents: {', '.join(successful_starts)}"
            )
        if failed_starts:
            OutputManager.print_warning(
                f"Failed to start {len(failed_starts)} agents: {', '.join(failed_starts)}"
            )

        # Add empty line for spacing
        print()

    def send_request_to_agent(self, agent_name, query, response_callback=None):
        """
        Send a request to an agent and handle streaming responses.

        Args:
            agent_name (str): Name of the agent to query
            query (str): The query to send to the agent
            response_callback (callable, optional): Callback function that will be called
                                                   with each response chunk

        Returns:
            str: The agent's final response or error message
        """
        if agent_name not in self.agent_processes:
            return f"Error: Agent '{agent_name}' not found or not running"

        try:
            # Send the query to the agent (make subtle)
            print(f"{SUBTLE_COLOR}Sending request to agent '{agent_name}'{RESET}")
            self.agent_pipes[agent_name].send({"type": "request", "query": query})

            # Process streaming responses (make subtle)
            print(
                f"{SUBTLE_COLOR}Starting to receive streaming responses from agent '{agent_name}'{RESET}"
            )

            full_response = ""
            is_complete = False

            # We'll keep polling the pipe until we get the final response
            while not is_complete:
                # Check for a response with a short timeout to not block the main thread
                if self.agent_pipes[agent_name].poll(timeout=0.1):
                    # Read the message
                    message = self.agent_pipes[agent_name].recv()

                    # Handle different message types
                    if message["type"] == "response":
                        # Check if it's the final message
                        if message.get("is_final", False):
                            is_complete = True
                            # If there's a full_response, use it (for final message)
                            if "full_response" in message and message["full_response"]:
                                full_response = message["full_response"]

                        # Add to response if not empty
                        response_chunk = message.get("response", "")
                        if response_chunk:
                            full_response += response_chunk

                        # If it's an error, mark as complete
                        if message.get("is_error", False):
                            is_complete = True

                        # Call the callback if provided
                        if response_callback and callable(response_callback):
                            try:
                                response_callback(message)
                            except Exception as e:
                                print(
                                    f"{SUBTLE_COLOR}Exception in response callback: {e}{RESET}"
                                )

                    elif message["type"] == "status":
                        # Status updates can be handled by the callback
                        if response_callback and callable(response_callback):
                            try:
                                response_callback(message)
                            except Exception as e:
                                print(
                                    f"{SUBTLE_COLOR}Exception in status callback: {e}{RESET}"
                                )

                # Check if the process is still alive
                if not self.agent_processes[agent_name].is_alive():
                    if not is_complete:
                        return f"Error: Agent '{agent_name}' process died unexpectedly"

                # Short sleep to prevent busy waiting
                time.sleep(0.01)

            return full_response

        except Exception as e:
            print(
                f"{SUBTLE_COLOR}Exception communicating with agent '{agent_name}': {e}{RESET}"
            )
            return f"Error communicating with agent '{agent_name}': {e}"

    def shutdown_agents(self):
        """Shutdown all agent processes cleanly."""
        print("Shutting down all agent processes...")

        for agent_name, agent_process in self.agent_processes.items():
            try:
                if agent_process.is_alive():
                    # Send shutdown message
                    self.agent_pipes[agent_name].send({"type": "shutdown"})

                    # Wait for agent to exit gracefully (with timeout)
                    agent_process.join(timeout=2)

                    # Force terminate if still running
                    if agent_process.is_alive():
                        print(f"Force terminating agent '{agent_name}'")
                        agent_process.terminate()

            except Exception as e:
                print(f"Error shutting down agent '{agent_name}': {e}")

        print("All agent processes shut down")

    def list_available_agents(self):
        """
        Get a list of all available agents.

        Returns:
            list: List of agent names
        """
        return list(self.agents.keys())

    def get_agent_info(self, agent_name):
        """
        Get information about an agent.

        Args:
            agent_name (str): Name of the agent

        Returns:
            dict: Agent configuration or None if not found
        """
        return self.agents.get(agent_name)


# Singleton instance
_registry_instance = None


def get_registry():
    """
    Get the singleton registry instance.

    Returns:
        AgentRegistry: The singleton registry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = AgentRegistry()
    return _registry_instance
