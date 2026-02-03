#!/bin/bash

set -e  # Exit on error

echo "🔧 Checking system permissions..."
if groups $USER | grep &>/dev/null "\bdialout\b"; then
    echo "✅ User is already in the dialout group."
else
    echo "Adding $USER to dialout group for serial port access..."
    sudo usermod -aG dialout "$USER"
    echo "⚠️  Permissions updated. NOTE: You may need to log out and back in."
    export GRP_CHECK=1
fi

echo "🔧 Installing system dependencies..."
sudo apt update
# Removed python3.10-dev; uv will handle the 3.11 headers internally
sudo apt install -y python3-tk python3-pyqt5 portaudio19-dev curl

#=== Install uv (Universal Virtualenv) ===
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure Python 3.11 is available via uv
echo "🐍 Ensuring Python 3.11 is installed..."
uv python install 3.11

echo "🔧 Setting up Python 3.11 virtual environment..."
# Create venv specifically using 3.11
uv venv .venv --python 3.11 --clear
source .venv/bin/activate

# Fix: link 'python' to 'python3' inside venv
# ln -sf python3 .venv/bin/python

echo "📦 Installing Python dependencies..."
# Using 'uv pip' is 10-100x faster than standard pip
uv pip install --upgrade pip
uv pip install -e .

echo "---"
echo "✅ Installation complete with Python 3.11."
if [ "$GRP_CHECK" == "1" ]; then
    echo "🚀 IMPORTANT: Please log out and back in to finalize serial permissions."
fi
echo "Activate your environment with: source .venv/bin/activate"