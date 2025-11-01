#!/usr/bin/env python3
"""
SunnySett Model: mrm8488/distilroberta-finetuned-financial-news-sentiment for Finance
===================================================================================

This script demonstrates how to use mrm8488/distilroberta-finetuned-financial-news-sentiment for sentiment-analysis
in finance applications.

Description: DistilRoBERTa model fine-tuned for financial news sentiment

Dependencies:
- transformers
- torch
- numpy
- pandas (optional)

Usage:
    python mrm8488_distilroberta_finetuned_financial_news_sentiment.py
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
    if "sentiment-analysis" == "feature-extraction":
        required_packages.append("sentence-transformers>=2.2.0")
        required_packages.append("scikit-learn>=1.0.0")
    elif "sentiment-analysis" == "automatic-speech-recognition":
        required_packages.append("librosa>=0.9.0")
        required_packages.append("soundfile>=0.10.0")
    elif "sentiment-analysis" == "text-to-image":
        required_packages.append("diffusers>=0.10.0")
        required_packages.append("pillow>=8.0.0")
    
    for package in required_packages:
        try:
            __import__(package.split(">=")[0].replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_model():
    """Load the mrm8488/distilroberta-finetuned-financial-news-sentiment model and tokenizer."""
    try:
        from transformers import pipeline
        
        model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment"
        print(f"Loading {model_name}...")
        
        # Create pipeline for sentiment-analysis
        classifier = pipeline("sentiment-analysis", model=model_name)
        
        return classifier, model_name
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def run_inference(classifier, model_name):
    """Run sample inference on finance text."""
    if classifier is None:
        print("Model not loaded. Cannot run inference.")
        return
    
    print(f"\n=== {model_name} Finance Analysis ===")
    
    # Sample finance texts
    sample_texts = [
        "Sample text for finance analysis and testing.",
        "This is another example of finance content.",
        "Testing the model with various finance scenarios.",
        "Example text to demonstrate the model's capabilities.",
        "Sample data for finance processing and analysis."
    ]
    
    print("Finance Text Analysis:")
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
    """Main function to run the mrm8488/distilroberta-finetuned-financial-news-sentiment analysis."""
    print("SunnySett Finance Model: mrm8488/distilroberta-finetuned-financial-news-sentiment")
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
        print(f"1. This model is designed for sentiment-analysis")
        print(f"2. Description: DistilRoBERTa model fine-tuned for financial news sentiment")
        print("3. Use for finance applications and analysis")
        print("4. Consider fine-tuning on your specific domain data")
        print("5. Check the model card for more detailed usage instructions")
        
    else:
        print("❌ Failed to load model. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
