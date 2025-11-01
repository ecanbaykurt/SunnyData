#!/usr/bin/env python3
"""
SunnySett Model: sentence-transformers/all-MiniLM-L6-v2 for Engineering
=====================================================================

This script demonstrates how to use sentence-transformers/all-MiniLM-L6-v2 for feature-extraction
in engineering applications.

Description: Sentence transformer for semantic search and similarity

Dependencies:
- transformers
- torch
- numpy
- pandas (optional)

Usage:
    python sentence_transformers_all_MiniLM_L6_v2.py
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
    if "feature-extraction" == "feature-extraction":
        required_packages.append("sentence-transformers>=2.2.0")
        required_packages.append("scikit-learn>=1.0.0")
    elif "feature-extraction" == "automatic-speech-recognition":
        required_packages.append("librosa>=0.9.0")
        required_packages.append("soundfile>=0.10.0")
    elif "feature-extraction" == "text-to-image":
        required_packages.append("diffusers>=0.10.0")
        required_packages.append("pillow>=8.0.0")
    
    for package in required_packages:
        try:
            __import__(package.split(">=")[0].replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_model():
    """Load the sentence-transformers/all-MiniLM-L6-v2 model and tokenizer."""
    try:
        from transformers import pipeline
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading {model_name}...")
        
        # Create pipeline for feature-extraction
        classifier = pipeline("feature-extraction", model=model_name)
        
        return classifier, model_name
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def run_inference(classifier, model_name):
    """Run sample inference on engineering text."""
    if classifier is None:
        print("Model not loaded. Cannot run inference.")
        return
    
    print(f"\n=== {model_name} Engineering Analysis ===")
    
    # Sample engineering texts
    sample_texts = [
        "Sample text for engineering analysis and testing.",
        "This is another example of engineering content.",
        "Testing the model with various engineering scenarios.",
        "Example text to demonstrate the model's capabilities.",
        "Sample data for engineering processing and analysis."
    ]
    
    print("Engineering Text Analysis:")
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
    """Main function to run the sentence-transformers/all-MiniLM-L6-v2 analysis."""
    print("SunnySett Engineering Model: sentence-transformers/all-MiniLM-L6-v2")
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
        print(f"1. This model is designed for feature-extraction")
        print(f"2. Description: Sentence transformer for semantic search and similarity")
        print("3. Use for engineering applications and analysis")
        print("4. Consider fine-tuning on your specific domain data")
        print("5. Check the model card for more detailed usage instructions")
        
    else:
        print("❌ Failed to load model. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
