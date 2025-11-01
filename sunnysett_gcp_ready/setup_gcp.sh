#!/bin/bash
# SunnySett GCP Setup Script

echo "Setting up SunnySett on GCP VM..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.10+
sudo apt-get install -y python3.10 python3.10-pip python3.10-venv

# Create virtual environment
python3.10 -m venv sunnysett_env
source sunnysett_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    cat > .env << EOF
# ============================================
# HUGGING FACE API CONFIGURATION
# ============================================
HUGGINGFACE_API_KEY=your_huggingface_key_here

# ============================================
# KAGGLE (Dataset Resources)
# ============================================
KAGGLE_USERNAME=your_kaggle_username_here
KAGGLE_KEY=your_kaggle_key_here
EOF
    echo "Please edit .env file with your actual API keys"
fi

echo "Setup complete!"
echo "To activate the environment: source sunnysett_env/bin/activate"
echo "To run a model: python sunnysett_models/Marketing/bert_base_uncased.py"
