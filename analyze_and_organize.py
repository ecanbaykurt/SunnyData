#!/usr/bin/env python3
"""
SunnySett Codebase Analysis and Organization Script
=================================================

This script analyzes all model APIs, identifies missing Python files,
removes duplicates, and organizes the codebase for GCP shipping.

Usage:
    python analyze_and_organize.py
"""

import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict

def load_all_model_apis():
    """Load all model APIs from both catalogs."""
    all_models = []
    
    # Load from prompt 2 catalog
    try:
        with open('sunnysett_model_catalog_prompt_2.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_models.append({
                    'category': row['category'],
                    'model_id': row['model_id'],
                    'model_name': row['model_name'],
                    'pipeline_tag': row['pipeline_tag'],
                    'modality': row['modality'],
                    'description': row['description']
                })
    except FileNotFoundError:
        print("Warning: sunnysett_model_catalog_prompt_2.csv not found")
    
    # Load from prompt 3 catalog
    try:
        with open('sunnysett_model_catalog_prompt_3.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_models.append({
                    'category': row['category'],
                    'model_id': row['model_id'],
                    'model_name': row['model_name'],
                    'pipeline_tag': row['pipeline_tag'],
                    'modality': row['modality'],
                    'description': row['description']
                })
    except FileNotFoundError:
        print("Warning: sunnysett_model_catalog_prompt_3.csv not found")
    
    return all_models

def get_current_python_files():
    """Get all current Python files in sunnysett_models."""
    python_files = []
    models_dir = Path("sunnysett_models")
    
    if models_dir.exists():
        for py_file in models_dir.rglob("*.py"):
            python_files.append({
                'name': py_file.name,
                'path': str(py_file),
                'category': py_file.parent.name,
                'model_id': py_file.stem
            })
    
    return python_files

def create_model_id_mapping():
    """Create mapping from model_id to expected filename."""
    mapping = {}
    all_models = load_all_model_apis()
    
    for model in all_models:
        model_id = model['model_id']
        # Convert model_id to filename format
        filename = model_id.replace("/", "_").replace("-", "_") + ".py"
        mapping[model_id] = {
            'filename': filename,
            'category': model['category'],
            'model_name': model['model_name'],
            'pipeline_tag': model['pipeline_tag'],
            'modality': model['modality'],
            'description': model['description']
        }
    
    return mapping

def analyze_missing_files():
    """Analyze what files are missing."""
    model_mapping = create_model_id_mapping()
    current_files = get_current_python_files()
    
    # Create set of current filenames for quick lookup
    current_filenames = {f['name'] for f in current_files}
    
    missing_files = []
    duplicate_files = []
    unique_files = []
    
    # Check for missing files
    for model_id, info in model_mapping.items():
        expected_filename = info['filename']
        if expected_filename not in current_filenames:
            missing_files.append({
                'model_id': model_id,
                'filename': expected_filename,
                'category': info['category'],
                'model_name': info['model_name']
            })
        else:
            unique_files.append({
                'model_id': model_id,
                'filename': expected_filename,
                'category': info['category']
            })
    
    # Check for duplicates
    filename_counts = defaultdict(list)
    for file_info in current_files:
        filename_counts[file_info['name']].append(file_info)
    
    for filename, files in filename_counts.items():
        if len(files) > 1:
            duplicate_files.append({
                'filename': filename,
                'count': len(files),
                'locations': [f['path'] for f in files]
            })
    
    return missing_files, duplicate_files, unique_files

def create_missing_python_files(missing_files):
    """Create missing Python files."""
    print(f"\nCreating {len(missing_files)} missing Python files...")
    
    for missing in missing_files:
        category = missing['category']
        model_id = missing['model_id']
        filename = missing['filename']
        model_name = missing['model_name']
        
        # Create category directory if it doesn't exist
        category_dir = Path(f"sunnysett_models/{category}")
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Python file
        file_path = category_dir / filename
        
        # Generate basic Python script content
        script_content = f'''#!/usr/bin/env python3
"""
SunnySett Model: {model_id} for {category.replace("_", " ").title()}
{'=' * (len(model_id) + len(category) + 20)}

This script demonstrates how to use {model_id} for {category.replace("_", " ").lower()} applications.

Description: {model_name}

Dependencies:
- transformers
- torch
- numpy
- pandas (optional)

Usage:
    python {filename}
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
    
    for package in required_packages:
        try:
            __import__(package.split(">=")[0].replace("-", "_"))
        except ImportError:
            print(f"Installing {{package}}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_model():
    """Load the {model_id} model."""
    try:
        from transformers import pipeline
        
        model_name = "{model_id}"
        print(f"Loading {{model_name}}...")
        
        # Create pipeline for {category.replace("_", " ").lower()} tasks
        classifier = pipeline("text-classification", model=model_name)
        
        return classifier, model_name
        
    except Exception as e:
        print(f"Error loading model: {{e}}")
        return None, None

def run_inference(classifier, model_name):
    """Run sample inference on {category.replace("_", " ").lower()} data."""
    if classifier is None:
        print("Model not loaded. Cannot run inference.")
        return
    
    print(f"\\n=== {{model_name}} {category.replace("_", " ").title()} Analysis ===")
    
    # Sample {category.replace("_", " ").lower()} data
    sample_texts = [
        "Sample text for {category.replace("_", " ").lower()} analysis and testing.",
        "This is another example of {category.replace("_", " ").lower()} content.",
        "Testing the model with various {category.replace("_", " ").lower()} scenarios.",
        "Example data to demonstrate the model's capabilities.",
        "Sample data for {category.replace("_", " ").lower()} processing and analysis."
    ]
    
    print("{category.replace("_", " ").title()} Data Analysis:")
    print("-" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        try:
            result = classifier(text)
            print(f"{{i}}. Text: '{{text}}'")
            print(f"   Result: {{result}}")
            print()
        except Exception as e:
            print(f"Error processing text {{i}}: {{e}}")

def main():
    """Main function to run the {model_id} analysis."""
    print("SunnySett {category.replace("_", " ").title()} Model: {model_id}")
    print("=" * 60)
    
    # Install dependencies
    print("Checking dependencies...")
    install_dependencies()
    
    # Load model
    classifier, model_name = load_model()
    
    if classifier:
        print(f"✅ Model '{{model_name}}' loaded successfully!")
        
        # Run inference
        run_inference(classifier, model_name)
        
        print("\\n=== Usage Tips ===")
        print(f"1. This model is designed for {category.replace("_", " ").lower()} applications")
        print(f"2. Description: {model_name}")
        print("3. Use for {category.replace("_", " ").lower()} analysis and processing")
        print("4. Consider fine-tuning on your specific domain data")
        print("5. Check the model card for more detailed usage instructions")
        
    else:
        print("❌ Failed to load model. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
'''
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"Created: {file_path}")

def remove_duplicates(duplicate_files):
    """Remove duplicate files, keeping only one copy."""
    print(f"\nRemoving {len(duplicate_files)} duplicate files...")
    
    for duplicate in duplicate_files:
        filename = duplicate['filename']
        locations = duplicate['locations']
        
        # Keep the first occurrence, remove the rest
        keep_file = locations[0]
        remove_files = locations[1:]
        
        print(f"Keeping: {keep_file}")
        for file_to_remove in remove_files:
            try:
                os.remove(file_to_remove)
                print(f"Removed: {file_to_remove}")
            except OSError as e:
                print(f"Error removing {file_to_remove}: {e}")

def organize_codebase():
    """Organize the codebase for GCP shipping."""
    print("\nOrganizing codebase for GCP shipping...")
    
    # Create clean GCP directory structure
    gcp_dir = Path("sunnysett_gcp_ready")
    if gcp_dir.exists():
        shutil.rmtree(gcp_dir)
    
    gcp_dir.mkdir()
    
    # Copy essential files
    essential_files = [
        "README.md",
        "sunnysett_models_metadata.csv",
        "sunnysett_models_metadata.json",
        "sunnysett_models_summary.md",
        "sunnysett_model_catalog_prompt_2.csv",
        "sunnysett_model_catalog_prompt_3.csv",
        ".env",
        ".gitignore"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            shutil.copy2(file, gcp_dir)
            print(f"Copied: {file}")
    
    # Copy sunnysett_models directory
    if os.path.exists("sunnysett_models"):
        shutil.copytree("sunnysett_models", gcp_dir / "sunnysett_models")
        print("Copied: sunnysett_models/")
    
    # Create requirements.txt
    requirements_content = """# SunnySett Requirements
# Core ML libraries
transformers>=4.21.0
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0

# Additional dependencies
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
librosa>=0.9.0
soundfile>=0.10.0
diffusers>=0.10.0
pillow>=8.0.0
opencv-python>=4.6.0
torchvision>=0.13.0

# Time series and forecasting
nixtla>=0.1.0
prophet>=1.1.0
autogluon.tabular>=0.7.0

# Environment management
python-dotenv>=0.19.0

# Optional: Jupyter for interactive use
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""
    
    with open(gcp_dir / "requirements.txt", 'w') as f:
        f.write(requirements_content)
    
    print("Created: requirements.txt")
    
    # Create GCP setup script
    setup_script = gcp_dir / "setup_gcp.sh"
    setup_content = """#!/bin/bash
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

echo "✅ Setup complete!"
echo "To activate the environment: source sunnysett_env/bin/activate"
echo "To run a model: python sunnysett_models/Marketing/bert_base_uncased.py"
"""
    
    with open(setup_script, 'w') as f:
        f.write(setup_content)
    
    # Make setup script executable
    os.chmod(setup_script, 0o755)
    print("Created: setup_gcp.sh")
    
    return gcp_dir

def main():
    """Main function to analyze and organize the codebase."""
    print("SunnySett Codebase Analysis and Organization")
    print("=" * 50)
    
    # Load all model APIs
    all_models = load_all_model_apis()
    print(f"Total model APIs found: {len(all_models)}")
    
    # Analyze current Python files
    current_files = get_current_python_files()
    print(f"Current Python files: {len(current_files)}")
    
    # Analyze missing and duplicate files
    missing_files, duplicate_files, unique_files = analyze_missing_files()
    
    print(f"\nAnalysis Results:")
    print(f"- Missing files: {len(missing_files)}")
    print(f"- Duplicate files: {len(duplicate_files)}")
    print(f"- Unique files: {len(unique_files)}")
    
    # Create missing files
    if missing_files:
        create_missing_python_files(missing_files)
    
    # Remove duplicates
    if duplicate_files:
        remove_duplicates(duplicate_files)
    
    # Organize codebase
    gcp_dir = organize_codebase()
    
    print(f"\n✅ Codebase analysis and organization complete!")
    print(f"📁 GCP-ready package created in: {gcp_dir}")
    print(f"📊 Final stats:")
    print(f"   - Total model APIs: {len(all_models)}")
    print(f"   - Python files created: {len(missing_files)}")
    print(f"   - Duplicates removed: {len(duplicate_files)}")
    print(f"   - Ready for GCP deployment!")

if __name__ == "__main__":
    main()
