#!/bin/bash

set -e  # Exit on error

echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y python3-tk python3-pyqt5
sudo apt install -y python3.10-dev portaudio19-dev

#=== Install uv (Universal Virtualenv) ===
# install uv to ~/.local/bin and update your shell PATH
curl -LsSf https://astral.sh/uv/install.sh | sh

# make sure ~/.local/bin is on PATH for this session
export PATH="$HOME/.local/bin:$PATH"

# verify
uv --version

# enable Bash completions (use zsh/fish/etc. if that's your shell)
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing Python dependencies..."
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118



echo "✅ Installation complete. Activate your environment with: source .venv/bin/activate"
