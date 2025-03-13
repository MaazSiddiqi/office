# AI Office v2

## Overview

AI Office is a multi-agent AI system that simulates a professional office environment where specialized AI agents work collaboratively on tasks under the management of an Executive Assistant (EA). You, as the "CEO," interact exclusively with the EA, who orchestrates all agent activities on your behalf.

## Installation

### Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) with llama3.1 model installed

### Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure Ollama is running with the llama3.1 model:
   ```bash
   ollama run llama3.1
   ```

## Usage

With the virtual environment activated, run:

```bash
python main.py
```

### Commands

- `/task [description]` - Create a new task
- `/status` - View all ongoing tasks
- `/agents` - List available specialized agents
- `/feedback [agent] [comments]` - Provide feedback on an agent's performance
- `/help` - View available commands
- `exit`, `quit`, `/exit` - End the session

## Key Features

- **Executive Assistant (EA) as Single Point of Contact**: All user interactions flow through the EA, streamlining communication
- **Behind-the-Scenes Agent Collaboration**: Specialized agents work on tasks without direct user interaction
- **Parallel Task Processing**: Multiple tasks proceed simultaneously while you continue conversing with the EA
- **Centralized Memory Management**: EA controls access to organizational knowledge
- **Asynchronous Feedback Loop**: Continuous agent improvement through automated training and feedback

## System Architecture

### Core Components

#### Executive Assistant (EA)

- Primary interface between user and all AI agents
- Manages conversation context and delegation of tasks
- Conducts conversations with specialized agents on the user's behalf
- Synthesizes information from multiple sources into cohesive responses
- Controls access to centralized memory

#### Specialized Agents

- Domain-specific AI agents running as parallel processes
- Communicate only with the EA or Task Manager, not directly with the user
- Focus exclusively on their area of expertise
- Request additional information through the EA when needed

#### Task Manager

- Asynchronous process tracking all ongoing tasks
- Maintains task status, history, and dependencies
- Enables parallel execution of multiple tasks
- Provides progress updates to the EA

#### Memory System

- Centralized knowledge store managed by the EA
- Ensures consistent information access across the system
- Persists important conversation context and facts
- Implements appropriate information access controls

#### Feedback & Training System

- Continuous improvement loop for agent performance
- Propagates user feedback to relevant agents
- Automatically analyzes task success/failure
- Refines system prompts based on performance data

## Development Status

This project is currently in active development as a proof-of-concept for a future web application. The terminal interface serves as a prototype to validate key concepts like parallel agent execution, memory synchronization, task delegation, and feedback loops.

## Future Directions

- **Web Interface**: Development of a responsive web application
- **Agent Marketplace**: Community-contributed specialized agents
- **External Tool Integration**: Connections to productivity tools and data sources
- **Enhanced Visualization**: Visual representations of agent activities and task progress

## Technical Implementation

The system is built in Python using:

- Multiprocessing for parallel agent execution
- Local LLM inference capabilities
- Asynchronous task management
- Persistent memory storage

## Contributing

This project is currently in the early stages of development. Contribution guidelines will be established as the project matures.

## License

[License information will be provided here]
