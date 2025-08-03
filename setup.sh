#!/bin/bash

echo "Setting up Hailo BlueIris Detection Environment..."

# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libopencv-dev python3-opencv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create output directory
mkdir -p output

# Copy example config if config doesn't exist
if [ ! -f config.yaml ]; then
    cp config.yaml.example config.yaml
    echo "Created config.yaml from example. Please edit with your settings."
fi

echo "Setup complete!"
echo "Next steps:"
echo "1. Edit config.yaml with your BlueIris settings"
echo "2. Install Hailo SDK following Hailo's documentation"
echo "3. Run: source venv/bin/activate && python main.py"