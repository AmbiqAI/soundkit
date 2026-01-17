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

# enable Bash completions
if command -v uv >/dev/null 2>&1; then
    # Use -i to avoid double-adding if script is run twice
    if ! grep -q "uv generate-shell-completion" ~/.bashrc; then
        echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
    fi
fi

echo "🔧 Setting up Python virtual environment..."
# Detect Python interpreter (prefer python, fallback to python3)
if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
else
    echo "❌ Python interpreter not found. Please install Python 3.10+ and re-run."
    exit 127
fi

# Check Python version (requires >= 3.10)
PY_MAJOR="$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    PY_VER_STR="$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "❌ Found Python $PY_VER_STR; requires >= 3.10."
    exit 1
fi

$PYTHON_CMD -m venv .venv
source .venv/bin/activate
# Ensure 'python' points to the venv's python even if the system doesn't have it
ln -sf python3 .venv/bin/python
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

echo "📦 Installing Python dependencies..."
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

echo "---"
echo "✅ Installation complete."
if [ "$GRP_CHECK" == "1" ]; then
    echo "🚀 IMPORTANT: Please log out and back in (or restart) to finalize serial port permissions."
fi
echo "Activate your environment with: source .venv/bin/activate"