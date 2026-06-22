#!/bin/bash

# Install AU-Harness dependencies
pip install -r requirements.txt

# Find EVA if already cloned, otherwise clone it
EVA_PATH=$(find ~ -maxdepth 5 -name ".git" -type d 2>/dev/null \
    | while read gitdir; do
        remote=$(git -C "$(dirname $gitdir)" remote get-url origin 2>/dev/null)
        if [[ "$remote" == *"ServiceNow/eva"* ]]; then
            echo "$(dirname $gitdir)"
            break
        fi
    done)

if [ -n "$EVA_PATH" ]; then
    echo "Found EVA at $EVA_PATH, pulling latest..."
    git -C "$EVA_PATH" pull
else
    echo "EVA not found, cloning..."
    git clone -b ${EVA_BRANCH:-latest} --depth 1 --no-tags --single-branch https://github.com/ServiceNow/eva.git ../eva
    EVA_PATH="../eva"
    echo ""
    echo "NOTE: Please create $EVA_PATH/.env from $EVA_PATH/.env.example and fill in your API keys before running EVA."
    echo "  cp $EVA_PATH/.env.example $EVA_PATH/.env"
fi

# Install EVA dependencies
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing via pip..."
    pip install uv
fi
cd "$EVA_PATH" && uv sync --all-extras && cd -

# Save EVA path for use by evaluate.sh
echo "$EVA_PATH" > .eva_path
echo "EVA ready at $EVA_PATH"
