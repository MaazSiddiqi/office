# AI Office v2

This version implements a multi-agent system with the following features:

- Multiple specialized AI agents (Research Assistant, Technical Specialist, Calendar Manager)
- Agent process management with multiprocessing
- Query routing based on keyword matching
- Streaming responses from LLM
- Colored terminal output with status indicators
- Memory management system
- Executive Assistant for high-level coordination

## Components

- `main.py`: Main entry point
- `agent_config.py`: Shared agent configuration
- `agent_process.py`: Process management for individual agents
- `agent_registry.py`: Registry for managing all agents
- `executive_assistant.py`: High-level coordination
- `logger.py`: Centralized logging
- `memory_manager.py`: Long-term memory management
- `query_router.py`: Query routing to appropriate agents
- `conversation_memory.py`: Conversation history management

## Dependencies

- Python 3.8+
- colorama
- requests
- ollama (local LLM)

## Usage

1. Ensure Ollama is running: `ollama serve`
2. Run the system: `python main.py`
3. Type queries and the system will route them to appropriate agents
4. Type 'exit' to quit

## Configuration

Agent configurations are stored in JSON files in the `agents` directory.
