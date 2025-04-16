#!/bin/bash

set -e

# Install pyenv if not installed
if ! command -v pyenv &> /dev/null; then
    echo "pyenv not found. Installing pyenv..."
    curl https://pyenv.run | bash
fi 

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Mediapipe Python build supports Python 3.9-3.12
# Install Python 3.12 if not installed
if ! pyenv versions | grep -q "3.12"; then
    echo "Installing Python 3.12..."
    pyenv install 3.12
fi

# Set local Python version
pyenv local 3.12

# Create virtual environment if not created
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt