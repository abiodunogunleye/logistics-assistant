#!/usr/bin/env bash
set -euo pipefail

files=$(git diff --cached --name-only | grep -E "(^|/)(\.env($|\.)|\.streamlit/secrets\.toml$)" | grep -v "\.env\.example$" || true)

if [ -n "$files" ]; then
    echo "Error: the following file(s) must not be committed:"
    echo "$files"
    exit 1
fi
