#!/bin/bash

# Run pre-commit on all files
pre-commit run --all-files

# Run make style as suggested by Clémentine
make style

# Check if there are changes that need to be staged and committed
if ! git diff --quiet; then
    echo "Fixing inconsistencies and committing..."
    git add .
    git commit -m "fix checks"
    git push origin main
else
    echo "No changes detected."
fi
