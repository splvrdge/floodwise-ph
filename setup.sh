#!/bin/bash
# Setup script for Streamlit Cloud

# Exit on error
set -e

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p .streamlit

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:/home/appuser"

# Start health check in background
nohup python health.py > health.log 2>&1 &

# Make sure the script is executable
chmod +x setup.sh

# Keep the container alive
while true; do sleep 1000; done
