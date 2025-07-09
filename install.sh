#!/bin/bash

set -e  # Exit on error

echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y python3-tk python3-pyqt5
sudo apt install -y python3.10-dev portaudio19-dev

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing Python dependencies..."
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

echo "✅ Installation complete. Activate your environment with: source .venv/bin/activate"
