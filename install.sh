#!/bin/bash

set -e  # Exit on error

echo "🔧 Installing system dependencies..."
sudo apt update

# Python and build dependencies
sudo apt install -y python3 python3-venv python3-tk python3-pyqt5
sudo apt install -y python3.10-dev portaudio19-dev

# Ensure curl is installed
if ! command -v curl >/dev/null 2>&1; then
	echo "Installing curl..."
	sudo apt install -y curl
fi

#=== Install uv (Universal Virtualenv) ===

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
	echo "Installing uv (Universal Virtualenv)..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	# make sure ~/.local/bin is on PATH for this session
	export PATH="$HOME/.local/bin:$PATH"
fi

# verify
uv --version

# enable Bash completions (use zsh/fish/etc. if that's your shell)
if command -v uv >/dev/null 2>&1; then
	echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
	source ~/.bashrc
fi

echo "🔧 Setting up Python virtual environment..."
# Detect Python interpreter (prefer python, fallback to python3)
if command -v python >/dev/null 2>&1; then
	PYTHON_CMD=python
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_CMD=python3
else
	echo "❌ Python interpreter not found. Please install Python 3.10+ (e.g., 'sudo apt install python3 python3-venv') and re-run."
	exit 127
fi

# Check Python version (requires >= 3.10)
PY_MAJOR="$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
	PY_VER_STR="$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
	echo "❌ Found Python $PY_VER_STR; requires >= 3.10. Please upgrade Python and re-run."
	exit 1
fi

$PYTHON_CMD -m venv .venv
source .venv/bin/activate

echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

echo "📦 Installing Python dependencies..."
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

echo "✅ Installation complete. Activate your environment with: source .venv/bin/activate"