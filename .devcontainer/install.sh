#!/bin/bash

set -e  # Exit on error

echo "🔧 Checking system permissions..."

# Check if user is in 'dialout' group for serial access
if groups $USER | grep &>/dev/null "\bdialout\b"; then
    echo "✅ User is already in the dialout group."
else
    echo "Adding $USER to dialout group for serial port access..."
    sudo usermod -aG dialout "$USER"
    echo "⚠️  Permissions updated. NOTE: You may need to log out and back in for this to take effect."
    # Optional: try to apply to current sub-shell, though login is safer
    export GRP_CHECK=1
fi

echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y git-lfs xxd curl
git lfs install

# Python and build dependencies
sudo apt install -y python3 python3-venv python3-tk python3-pyqt5
sudo apt install -y python3.11-dev portaudio19-dev

echo "📦 Syncing Python dependencies..."
UV_PYTHON=/usr/bin/python3.11 uv sync

echo "---"
echo "✅ Installation complete."
if [ "$GRP_CHECK" == "1" ]; then
    echo "🚀 IMPORTANT: Please log out and back in (or restart) to finalize serial port permissions."
fi
echo "Activate your environment with: source .venv/bin/activate"
source .venv/bin/activate
