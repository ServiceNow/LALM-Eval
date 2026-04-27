#!/bin/bash

# Ensure setup.sh has been run
if [ ! -d "./eva" ]; then
    echo "Error: EVA not found. Please run 'bash setup.sh' first."
    exit 1
fi

# Parse command line arguments
DEBUG=false
ARGS=()

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      ARGS+=("$1")
      # Unknown option
      shift
      ;;
  esac
done

if [ "$DEBUG" = true ]; then
  echo "Starting in debug mode - waiting for debugger to attach on port 5678..."
  python -m debugpy --listen 0.0.0.0:5678 --wait-for-client evaluate.py "$@"
else
  python evaluate.py "${ARGS[@]}"
fi