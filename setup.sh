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
    git clone https://github.com/ServiceNow/eva.git ./eva
    EVA_PATH="./eva"
fi

echo "EVA ready at $EVA_PATH"
