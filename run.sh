#!/bin/bash

# Exit on error
set -e

echo "Setting up AI Office..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version &>/dev/null; then
  echo "Error: Ollama is not running. Please start Ollama first with: ollama serve"
  exit 1
fi

# Run the AI Office
echo "Starting AI Office..."
python main.py
