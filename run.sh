#!/bin/bash

# Exit on error
set -e

echo "Setting up AI Office..."

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f ".setup_complete" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
  touch .setup_complete
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version &>/dev/null; then
  echo "Error: Ollama is not running. Please start Ollama first with: ollama serve"
  exit 1
fi

# Run the AI Office
echo "Starting AI Office..."
python main.py
