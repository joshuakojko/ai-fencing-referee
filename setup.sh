#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script must be run with bash"
    exit 1
fi

set -e

# Detect OS
OS=$(uname -s)

# Install pyenv if not installed
if [[ "$OS" == "Darwin" || "$OS" == "Linux" ]]; then
    if ! command -v pyenv &> /dev/null; then
        echo "pyenv not found. Installing pyenv..."
        curl https://pyenv.run | bash
    fi 
    # Initialize pyenv
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* ]]; then
    if ! command -v pyenv &> /dev/null; then
        echo "pyenv not found. Installing pyenv..."
        curl -L https://github.com/pyenv-win/pyenv-win/archive/master.zip -o pyenv-win.zip
        unzip pyenv-win.zip
        rm pyenv-win.zip
        mv pyenv-win-master ~/.pyenv

        # Initialize pyenv-win
        export PYENV_ROOT="$HOME/.pyenv/pyenv-win"
        export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi 
else 
    echo "Unsupported OS: $OS"
    exit 1
fi

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
if [[ "$OS" == "Darwin" || "$OS" == "Linux" ]]; then
    source .venv/bin/activate
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* ]]; then
    source .venv/Scripts/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download ffmpeg if not installed
if [[ "$OS" == "Darwin" || "$OS" == "Linux" ]]; then
    if ! command -v ffmpeg &> /dev/null; then
        echo "ffmpeg not found. Installing ffmpeg..."
        if [[ "$OS" == "Darwin" ]]; then
            brew install ffmpeg
        else
            sudo apt update
            sudo apt install ffmpeg
        fi
    else
        echo "ffmpeg is already installed"
    fi
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* ]]; then
    if ! command -v ffmpeg &> /dev/null; then
        echo "ffmpeg not found. Installing ffmpeg..."
        if ! command -v 7z &> /dev/null; then
            echo "7z not installed"
            exit 1
        fi
        # Download and extract ffmpeg for Windows
        curl -L https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z -o ffmpeg.7z
        7z x ffmpeg.7z
        mv ffmpeg-*-git-* ffmpeg-git
        mkdir -p "$HOME/bin"
        mv ffmpeg-git/bin/* "$HOME/bin/"
        rm -rf ffmpeg-git ffmpeg.7z
        export PATH="$HOME/bin:$PATH"
        if [[ "$OS" == "MINGW"* ]]; then
            echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
            [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
        elif [[ "$OS" == "CYGWIN"* ]]; then
            echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bash_profile"
            [ -f "$HOME/.bash_profile" ] && source "$HOME/.bash_profile"
        fi
    fi
fi