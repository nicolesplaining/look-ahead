#!/usr/bin/env bash

set -euo pipefail

VENV_DIR="${1:-.venv}"
REQ_FILE="${2:-requirements.txt}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed or not in PATH."
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: requirements file '$REQ_FILE' not found."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in '$VENV_DIR'..."
  python3 -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at '$VENV_DIR'."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from '$REQ_FILE'..."
pip install -r "$REQ_FILE"

echo "Done."
echo "To activate later, run: source \"$VENV_DIR/bin/activate\""
