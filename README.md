# Llama 3.1 AI Office

A Python application that creates an AI office environment using Llama 3.1 through Ollama. The system features an Executive Assistant that can intelligently delegate tasks to specialized AI agents and maintains a persistent memory of important facts.

## New Feature: Executive Assistant as Primary Contact

The AI Office now features the Executive Assistant (EA) as your primary point of contact:

- **Direct Interaction**: The EA handles general questions, information requests, and basic tasks directly
- **Smart Delegation**: Only delegates to specialized agents when a task requires specific expertise
- **Memory Integration**: The EA can access and use your memory system for contextual responses
- **Continuity of Experience**: Provides a more natural office-like experience with a consistent main contact

This enhancement creates a more natural interaction flow, similar to how a real executive assistant would handle your requests - answering what they can directly and only bringing in specialists when their expertise is needed.

## New Feature: Conversation Session Management

The AI Office now includes intelligent conversation session management:

- **Continuous Conversations**: Maintain ongoing conversations with specialized agents without having to repeat context
- **Smart Topic Detection**: The system automatically detects when you've changed topics and routes to the appropriate agent
- **Seamless Handoffs**: Agents can hand control back to the EA when they've completed their task
- **Explicit Control**: Use commands to manually control which agent you're talking to
- **Session Timeout**: Inactive sessions automatically return to the EA after a period of inactivity

This creates a much more natural conversation flow, allowing you to have multi-turn interactions with specialized agents without the EA interrupting each time you respond to an agent's question.

## New Feature: Asynchronous Memory Manager

The AI Office now includes a dedicated asynchronous memory management system:

- **Non-blocking Memory Operations**: Memory extraction and storage happen in a separate process, keeping the main conversation responsive
- **Parallel Processing**: Memory operations run concurrently with conversation processing
- **Synchronized Access**: File locking ensures data consistency when multiple operations occur simultaneously
- **Fault Tolerance**: Memory operations are isolated, so failures don't affect the main conversation flow
- **Resource Efficiency**: Memory-intensive operations don't block or slow down the user experience

This enhancement makes interactions smoother by moving the potentially slow memory extraction and storage operations to the background while maintaining the benefits of the memory system for contextual conversations.

## New Feature: Multi-Process Architecture

The AI Office also offers a multi-process architecture where each agent runs in its own isolated process:

- **Process Isolation**: Each agent operates in its own process, maintaining independent context and memory
- **Improved Parallelism**: Multiple agents can work simultaneously
- **Better Resource Management**: Memory is allocated per agent, preventing one agent from consuming all resources
- **Enhanced Fault Tolerance**: If one agent crashes, it doesn't affect the entire system
- **Specialized Memory**: Each agent maintains its own local memory in addition to sharing facts with the central system

By default, the system uses the multi-process architecture. To run in single-process mode, use the environment variable:

```bash
AI_OFFICE_MULTIPROCESS=0 python main.py
```

## Prerequisites

1. Install [Ollama](https://ollama.ai/download) on your system
2. Pull the Llama 3.1 model using the Ollama CLI:
   ```
   ollama pull llama3.1
   ```
3. (Optional) For faster agent selection, you can also pull the TinyLlama model:
   ```
   ollama pull tinyllama
   ```

## Setup

1. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Features

- **Executive Assistant (EA)**: Your main point of contact that handles general requests directly (ENHANCED!)
- **Smart Delegation**: The EA only delegates to specialists when required (NEW!)
- **Multi-Process Architecture**: Each agent runs in its own isolated process
- **Inter-Process Communication**: Secure message passing between agents and the EA
- **Intelligent Task Delegation**: Uses AI to analyze tasks and determine the most appropriate specialized agent
- **Fast Agent Selection**: Uses keyword-based classification to determine when delegation is needed
- **Persistent Agent Registry**: Saves your custom agents to a file for use across sessions
- **Memory System**: Extracts and stores key facts from conversations to provide context in future interactions
- **Built-in Specialized Agents**:
  - Calendar Manager - For scheduling and time management
  - Research Assistant - For finding information and answering questions
  - Creative Writer - For generating creative content
  - Technical Expert - For programming and technology help
  - Personal Assistant - For life advice and personal tasks

## Usage

Run the application:

```bash
python main.py
```

This will start an interactive prompt where you can speak with your Executive Assistant. The EA will handle most requests directly, only delegating to specialized agents when their expertise is required for your task.

### Commands

The following commands are available for managing your AI agents and memory:

- `/list_agents` - Show all available agents
- `/add_agent <name> "<system prompt>"` - Create a new specialized agent
- `/delete_agent <name>` - Remove an agent
- `/view_agent <name>` - View an agent's system prompt
- `/edit_agent <name> "<new system prompt>"` - Update an agent's system prompt
- `/memory` - Show all facts stored in memory
- `/memory <query>` - Show facts matching a specific query
- `/clear_memory` - Delete all stored memories

Conversation session commands:

- `/who` - Shows which agent you're currently talking to
- `/switch_to <agent>` - Explicitly switch to talking with a specific agent
- `/ea` - Switch back to talking with the Executive Assistant

### Examples

Adding a custom agent:

```
> /add_agent fitness_coach "You are a fitness and nutrition expert who provides workout plans, nutritional advice, and motivation for health goals."
```

Interacting with your AI office:

```
> Can you help me plan a healthy meal for the week?
```

Checking what the system remembers about you:

```
> /memory diet
```

## Memory System

The AI Office includes a memory system that:

1. **Automatically extracts key facts** from conversations
2. **Stores facts persistently** in a JSON file
3. **Provides relevant context** to agents when answering related questions
4. **Allows querying** stored facts with the `/memory` command

Facts are automatically extracted after each conversation. The system focuses on personal preferences, important dates, specific requirements, and other details that would be useful to remember in future interactions.

When you ask a question, relevant facts from past conversations are retrieved and provided as context to the agent handling your request, allowing for more personalized and contextually aware responses.

## How It Works

### Asynchronous Memory Management (New)

1. A dedicated Memory Manager Process runs separately from the main conversation flow
2. When a conversation occurs:
   - The conversation is immediately responded to for optimal user experience
   - In parallel, the conversation is sent to the Memory Manager for fact extraction
   - The Memory Manager processes the extraction asynchronously
   - Facts are stored in a shared memory file with proper locking mechanisms
3. When memory is needed:
   - Relevant facts are quickly retrieved using a synchronized read operation
   - The system maintains an in-memory cache for immediate access
   - If an update is in progress, reads wait only if absolutely necessary
4. Memory operations use a task queue and result queue for communication
5. File locking ensures data consistency even with concurrent operations

### Conversation Session Management (New)

1. When you interact with a specialized agent, the system begins a "conversation session" with that agent
2. Subsequent messages are automatically routed to the active agent until:
   - You explicitly switch agents using commands like `/ea` or `/switch_to <agent>`
   - The active agent completes its task and hands control back to the EA
   - The conversation topic changes significantly, as detected by content analysis
   - The session times out due to inactivity
3. This allows for natural multi-turn conversations with specialists without interruption
4. The system provides clear indicators showing which agent you're currently conversing with
5. You can always check who you're talking to with the `/who` command

### Executive Assistant as Primary Contact

1. When you enter a query, the Executive Assistant evaluates it first
2. For general questions, greetings, simple requests, or brief questions, the EA handles them directly
3. The EA checks if your query contains specific keywords that indicate specialized knowledge is needed
4. Only when specialized expertise is clearly required does the EA delegate to a specialized agent
5. Both the EA and specialized agents can access your memory to provide context-aware responses
6. This creates a natural, office-like interaction flow with a consistent main point of contact

### Multi-Process Architecture

1. **Executive Assistant (EA)** acts as the orchestrator for all agents
2. Each agent runs in its own isolated process with its own memory and context
3. The EA uses pipes for inter-process communication with the agents
4. When the EA receives a user request, it:
   - Determines if it should handle the request or delegate to a specialist
   - Retrieves relevant context from central memory
   - Either responds directly or sends a message to the selected agent's process
   - Waits for the agent to process the request and return a response
   - Extracts facts from the response and updates the central memory
5. Each agent process:
   - Waits for requests from the EA
   - Processes requests using its specialized system prompt
   - Maintains its own local memory of conversations
   - Returns responses and extracted facts to the EA
   - Continues running in the background, ready to handle future requests

### Agent Selection and Task Delegation

1. The Executive Assistant analyzes your request using:
   - First, tries to handle the request directly if it's general or simple
   - Uses keyword matching to identify requests that clearly need a specialist
   - Delegates only when specialized knowledge would clearly benefit you
2. It retrieves relevant facts from memory to provide context
3. The appropriate agent (EA or specialist) processes your request
4. Key facts from the conversation are extracted and added to memory
5. All agents are powered by Llama 3.1 running locally through Ollama
6. Agent configurations and memories are saved to JSON files for persistence

## Customization

You can modify:

- Any agent's system prompt to specialize them for different tasks
- The temperature and max_tokens parameters to adjust response style
- Add as many specialized agents as needed for your workflow
- Switch between single-process and multi-process modes

## Future Development Ideas

- Asynchronous task processing with notification system
- RAG capabilities for providing agents with specific knowledge
- Multi-step workflows involving multiple agents
- Integration with external APIs for real-world actions
- Usage tracking and cost estimation
- Parallel processing of tasks across multiple agents simultaneously

## Troubleshooting

- If you get an error about the model not being found, make sure you've run `ollama pull llama3.1`
  - Note: The script uses the model name `llama3.1:latest`, which is how Ollama registers the model
- If you can't connect to Ollama, ensure the Ollama service is running on your system
- If you encounter issues with the multi-process architecture, try running in single-process mode:
  ```bash
  AI_OFFICE_MULTIPROCESS=0 python main.py
  ```
