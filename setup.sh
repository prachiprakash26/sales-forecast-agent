#!/usr/bin/env bash

# Exit on error
set -e

echo "=== Setting up Sales Forecasting Agent Environment ==="

# Check python installation
if ! command -v python &> /dev/null; then
    echo "Python is not installed or not in PATH."
    exit 1
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create directories if they don't exist
echo "Creating data and source directories..."
mkdir -p data notebooks src/data src/forecasting src/tools src/agent src/utils

# Run mock data generator
echo "Generating mock sales dataset..."
python -m src.data.data_generator

echo "Setup complete! Run 'jupyter notebook' to start the Sales Forecasting Agent notebook."
