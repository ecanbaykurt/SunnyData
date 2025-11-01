#!/usr/bin/env python3
"""
SunnySett Model: openai/whisper-base for Education
================================================

This script demonstrates how to use openai/whisper-base for automatic-speech-recognition
in education applications.

Description: Whisper model for speech recognition and transcription

Dependencies:
- transformers
- torch
- numpy
- pandas (optional)

Usage:
    python openai_whisper_base.py
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

def install_dependencies():
    """Install required packages if not already installed."""
    required_packages = [
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ]
    
    # Add specific dependencies based on task
    if "automatic-speech-recognition" == "feature-extraction":
        required_packages.append("sentence-transformers>=2.2.0")
        required_packages.append("scikit-learn>=1.0.0")
    elif "automatic-speech-recognition" == "automatic-speech-recognition":
        required_packages.append("librosa>=0.9.0")
        required_packages.append("soundfile>=0.10.0")
    elif "automatic-speech-recognition" == "text-to-image":
        required_packages.append("diffusers>=0.10.0")
        required_packages.append("pillow>=8.0.0")
    
    for package in required_packages:
        try:
            __import__(package.split(">=")[0].replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_model():
    """Load the openai/whisper-base model and tokenizer."""
    try:
        from transformers import pipeline
        
        model_name = "openai/whisper-base"
        print(f"Loading {model_name}...")
        
        # Create pipeline for automatic-speech-recognition
        classifier = pipeline("automatic-speech-recognition", model=model_name)
        
        return classifier, model_name
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def run_inference(classifier, model_name):
    """Run sample inference on education text."""
    if classifier is None:
        print("Model not loaded. Cannot run inference.")
        return
    
    print(f"\n=== {model_name} Education Analysis ===")
    
    # Sample education texts
    sample_texts = [
        "Sample text for education analysis and testing.",
        "This is another example of education content.",
        "Testing the model with various education scenarios.",
        "Example text to demonstrate the model's capabilities.",
        "Sample data for education processing and analysis."
    ]
    
    print("Education Text Analysis:")
    print("-" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        try:
            result = classifier(text)
            print(f"{i}. Text: '{text}'")
            print(f"   Result: {result}")
            print()
        except Exception as e:
            print(f"Error processing text {i}: {e}")

def main():
    """Main function to run the openai/whisper-base analysis."""
    print("SunnySett Education Model: openai/whisper-base")
    print("=" * 60)
    
    # Install dependencies
    print("Checking dependencies...")
    install_dependencies()
    
    # Load model
    classifier, model_name = load_model()
    
    if classifier:
        print(f"✅ Model '{model_name}' loaded successfully!")
        
        # Run inference
        run_inference(classifier, model_name)
        
        print("\n=== Usage Tips ===")
        print(f"1. This model is designed for automatic-speech-recognition")
        print(f"2. Description: Whisper model for speech recognition and transcription")
        print("3. Use for education applications and analysis")
        print("4. Consider fine-tuning on your specific domain data")
        print("5. Check the model card for more detailed usage instructions")
        
    else:
        print("❌ Failed to load model. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
