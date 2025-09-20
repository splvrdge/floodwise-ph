#!/bin/bash

# Quick start script for running the Philippines Flood Control Chatbot locally

echo "🌊 FloodWise PH - Local Setup"
echo "=========================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "floodwise-ph"; then
    echo "✅ Environment 'floodwise-ph' already exists"
else
    echo "📦 Creating conda environment..."
    conda env create -f environment.yml
    if [ $? -eq 0 ]; then
        echo "✅ Environment created successfully"
    else
        echo "❌ Failed to create environment"
        exit 1
    fi
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up environment file..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your OpenAI API key"
    echo "   Your API key is already configured in the .env file"
else
    echo "✅ Environment file already exists"
fi

echo ""
echo "🚀 To run the application:"
echo "   1. conda activate floodwise-ph"
echo "   2. streamlit run app.py"
echo ""
echo "📖 The app will open in your browser at http://localhost:8501"
echo "📁 Upload the CSV file: Dataset/flood-control-projects-table_2025-09-20.csv"
