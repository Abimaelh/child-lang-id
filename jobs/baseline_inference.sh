#!/bin/bash
set -e

PYTHON_EXEC=$(which python)

echo "Using Python from: $PYTHON_EXEC"
echo "Running inference with arguments: $@"

$PYTHON_EXEC baseline_inference.py "$@"