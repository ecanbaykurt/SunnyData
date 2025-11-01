#!/usr/bin/env python3
"""
SunnySett Model: microsoft/DialoGPT-medium for Cybersecurity
==========================================================

This script demonstrates how to use microsoft/DialoGPT-medium for text-generation
in cybersecurity applications.

Description: DialoGPT for security chatbot and incident response

Dependencies:
- transformers
- torch
- numpy
- pandas (optional)

Usage:
    python microsoft_DialoGPT_medium.py
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
    if "text-generation" == "feature-extraction":
        required_packages.append("sentence-transformers>=2.2.0")
        required_packages.append("scikit-learn>=1.0.0")
    elif "text-generation" == "automatic-speech-recognition":
        required_packages.append("librosa>=0.9.0")
        required_packages.append("soundfile>=0.10.0")
    elif "text-generation" == "text-to-image":
        required_packages.append("diffusers>=0.10.0")
        required_packages.append("pillow>=8.0.0")
    
    for package in required_packages:
        try:
            __import__(package.split(">=")[0].replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_model():
    """Load the microsoft/DialoGPT-medium model and tokenizer."""
    try:
        from transformers import pipeline
        
        model_name = "microsoft/DialoGPT-medium"
        print(f"Loading {model_name}...")
        
        # Create pipeline for text-generation
        classifier = pipeline("text-generation", model=model_name)
        
        return classifier, model_name
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def run_inference(classifier, model_name):
    """Run sample inference on cybersecurity text."""
    if classifier is None:
        print("Model not loaded. Cannot run inference.")
        return
    
    print(f"\n=== {model_name} Cybersecurity Analysis ===")
    
    # Sample cybersecurity texts
    sample_texts = [
        "Sample text for cybersecurity analysis and testing.",
        "This is another example of cybersecurity content.",
        "Testing the model with various cybersecurity scenarios.",
        "Example text to demonstrate the model's capabilities.",
        "Sample data for cybersecurity processing and analysis."
    ]
    
    print("Cybersecurity Text Analysis:")
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
    """Main function to run the microsoft/DialoGPT-medium analysis."""
    print("SunnySett Cybersecurity Model: microsoft/DialoGPT-medium")
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
        print(f"1. This model is designed for text-generation")
        print(f"2. Description: DialoGPT for security chatbot and incident response")
        print("3. Use for cybersecurity applications and analysis")
        print("4. Consider fine-tuning on your specific domain data")
        print("5. Check the model card for more detailed usage instructions")
        
    else:
        print("❌ Failed to load model. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
