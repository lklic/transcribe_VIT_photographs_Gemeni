#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$( dirname "$SCRIPT_DIR" )"

echo "Viewer script directory: $SCRIPT_DIR"
echo "Parent (project) directory: $PARENT_DIR"

# Step 1: Generate the index file
echo "Running index generator..."
python3 "$SCRIPT_DIR/generate_index.py"
if [ $? -ne 0 ]; then
    echo "Error generating index file. Aborting."
    exit 1
fi
echo "Index generation complete."

# Step 2: Navigate to the parent directory
cd "$PARENT_DIR" || exit 1 # Exit if cd fails
echo "Changed directory to: $(pwd)"

# Step 3: Start the web server from the parent directory
echo "Starting Python HTTP server on port 8000..."
echo "Access the viewer at http://<your-ip>:8000/digiteca-viewer/"
python3 -m http.server 8000

# Note: The script will stay running here until the server is stopped (e.g., with Ctrl+C)
