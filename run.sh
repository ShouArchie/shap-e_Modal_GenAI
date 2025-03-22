#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Clone Shap-E repository
echo "Cloning Shap-E repository..."
git clone https://github.com/openai/shap-e.git

# Install Shap-E
echo "Installing Shap-E..."
pip install -e ./shap-e

# Create required directories
echo "Creating cache and output directories..."
mkdir -p shap_e_model_cache outputs

# Set executable permissions
chmod +x run.sh

# Run the app
echo "Starting the Flask API server..."
python main.py 