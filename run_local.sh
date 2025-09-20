#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting FloodWise PH - Local Development Setup"
echo "=========================================="

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$(printf '%s\n' "3.9" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.9" ]]; then
    echo "❌ Python 3.9 or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "ℹ️  Please edit the .env file and add your OpenAI API key"
fi

# Check if Dataset directory exists
if [ ! -d "Dataset" ]; then
    echo "📂 Creating Dataset directory..."
    mkdir -p Dataset
    echo "ℹ️  Please add your dataset file (flood-control-projects-table_2025-09-20.csv) to the Dataset/ directory"
fi

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create config.toml if it doesn't exist
if [ ! -f ".streamlit/config.toml" ]; then
    echo "⚙️  Creating Streamlit config..."
    cat > .streamlit/config.toml <<EOL
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#4f8bf9"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
EOL
fi

echo "✅ Setup complete!"
echo ""
echo "To start the application, run:"
echo "source venv/bin/activate"
echo "streamlit run app.py"
echo ""
echo "The application will be available at: http://localhost:8501"

# Ask if user wants to start the app now
read -p "Would you like to start the application now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting FloodWise PH..."
    streamlit run app.py
fi
