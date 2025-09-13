#!/bin/bash
set -e

if [ -f ".env" ]; then
  echo "Loading environment variables from .env file"
  export $(cat .env | grep -v '^#' | xargs)
else
  echo "No .env file found, using default environment variables"
fi

if [ -d "/models" ] && [ -n "$(ls -A /models 2>/dev/null)" ]; then
  echo "Using local model from /models directory"
  export MODEL_PATH="/models"
fi

echo "Using MODEL_PATH: $MODEL_PATH"
echo "Using USE_THINK_MODE: $USE_THINK_MODE"
echo "Using PORT: $PORT"
echo "Using PYTHONPATH: $PYTHONPATH"

export PYTHONPATH="$PYTHONPATH:/app:/app/audio-flamingo:/app/audio-flamingo/llava/model:/app/audio-flamingo/llava/model/quantization"

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
