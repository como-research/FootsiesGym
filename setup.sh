#!/bin/bash

# Script to unpack footsies_linux_server_021725.zip and rename to footsies_binaries

set -e  # Exit on any error

# Check for python3 or python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null
then
    echo "'python' or 'python3' command not found. Please install it to continue."
    exit 1
fi

# Check if the zip file exists
if [ ! -f "binaries/footsies_linux_server_021725.zip" ]; then
    echo "Error: binaries/footsies_linux_server_021725.zip not found!"
    exit 1
fi

# Create binaries directory if it doesn't exist
mkdir -p binaries

# Change to binaries directory
cd binaries

# Determine which python command to use
PYTHON_CMD=python3
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD=python
fi

# Extract the zip file using Python
echo "Extracting footsies_linux_server_021725.zip..."
$PYTHON_CMD -m zipfile -e footsies_linux_server_021725.zip .

# Find the extracted directory (assuming there's only one top-level directory)
EXTRACTED_DIR=$(find . -maxdepth 1 -type d ! -name '.' | head -n 1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "Error: No directory found after extraction!"
    exit 1
fi

# Remove the leading './' from the directory name
EXTRACTED_DIR=${EXTRACTED_DIR#./}

# Rename the extracted directory to footsies_binaries
if [ "$EXTRACTED_DIR" != "footsies_binaries" ]; then
    echo "Renaming $EXTRACTED_DIR to footsies_binaries..."
    mv "$EXTRACTED_DIR" footsies_binaries
else
    echo "Directory already named footsies_binaries"
fi

echo "Setup complete! Contents are now in binaries/footsies_binaries/"
