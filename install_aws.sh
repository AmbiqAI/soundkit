#!/bin/bash

set -e  # Exit on error

echo "🔧 Installing system dependencies..."

# Update yum packages
sudo yum update -y

# Install Amazon Linux equivalents of Ubuntu packages
sudo yum install -y python3 python3-pip python3-devel
sudo python3 -m pip install --upgrade pip

# Amazon Linux uses different package names for audio libs
sudo yum install -y portaudio-devel alsa-lib-devel pulseaudio-libs-devel

# (Optional) if you need Tkinter / GUI libs:
sudo yum install -y tkinter xorg-x11-server-Xorg xorg-x11-xauth xorg-x11-apps

# Ensure curl is installed
if ! command -v curl >/dev/null 2>&1; then
    echo "Installing curl..."
    sudo yum install -y curl
fi

#=== Install uv (Universal Virtualenv) ===

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (Universal Virtualenv)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # ensure $HOME/.local/bin is on PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

# verify uv
uv --version

# enable Bash completions
if command -v uv >/dev/null 2>&1; then
    echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
    source ~/.bashrc
fi

echo "📦 Installing Python dependencies..."
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

echo "✅ Installation complete."
echo "Run: source .venv/bin/activate"
