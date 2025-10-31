#!/usr/bin/sh
set -e

VENV_DIR=".venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv and install requirements if needed
. "$VENV_DIR/bin/activate"

if [ ! -f "$VENV_DIR/.requirements_installed" ]; then
    echo "Installing Python dependencies..."
    pip install -q -r requirements.txt
    touch "$VENV_DIR/.requirements_installed"
fi

# Build and run the Zig program
echo "Building and running mxfp4Loader..."
zig build run -- test_models/tiny_gpt_oss/model.safetensors NEW.safetensors

# Compare the output
echo ""
echo "Comparing tensors..."
python3 compare_tensors.py test_models/tiny_gpt_oss/test_output_model.safetensors NEW.safetensors

deactivate
