#!/bin/bash

# EVA mode
if [[ "$1" == "--eva" ]]; then
    if [ ! -f ".eva_path" ]; then
        echo "Error: EVA not found. Please run 'bash setup.sh' first."
        exit 1
    fi
    EVA_PATH=$(cat .eva_path)
    if [ ! -f "$EVA_PATH/.env" ]; then
        echo "Error: $EVA_PATH/.env not found. Please create it from $EVA_PATH/.env.example and fill in your API keys."
        exit 1
    fi
    shift
    if [[ "$1" == "--text" ]]; then
        shift
        cd "$EVA_PATH" && uv run python scripts/run_text_only.py "$@"
    else
        cd "$EVA_PATH" && uv run eva "$@"
    fi
    exit 0
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