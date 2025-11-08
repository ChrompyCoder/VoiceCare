#!/bin/bash

# Setup script for Surfboard + XGBoost Parkinson's Detection
# This script helps set up the environment and verify the installation

echo "=================================================="
echo "Surfboard + XGBoost Setup"
echo "Parkinson's Disease Detection"
echo "=================================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo -e "\n[2/5] Creating virtual environment..."
    python3 -m venv venv_surfboard
    source venv_surfboard/bin/activate
    echo "✓ Virtual environment created and activated"
else
    echo -e "\n[2/5] Skipping virtual environment creation"
fi

# Install requirements
echo -e "\n[3/5] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Packages installed"

# Verify installations
echo -e "\n[4/5] Verifying installations..."

python3 << 'EOF'
import sys

packages = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'Scikit-learn'),
    ('xgboost', 'XGBoost'),
    ('librosa', 'Librosa'),
    ('soundfile', 'SoundFile'),
    ('surfboard', 'Surfboard'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn')
]

all_installed = True
for module, name in packages:
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ❌ {name} - NOT INSTALLED")
        all_installed = False

if all_installed:
    print("\n✓ All packages verified!")
else:
    print("\n❌ Some packages are missing. Please install them manually.")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "Installation verification failed!"
    exit 1
fi

# Check data directory
echo -e "\n[5/5] Checking data directory..."
if [ -d "../data" ]; then
    echo "✓ Data directory found: ../data"
    
    # Count audio files
    gitdata_count=$(find ../data/gitdata -name "*.wav" 2>/dev/null | wc -l)
    figdata_count=$(find ../data/figdata -name "*.wav" 2>/dev/null | wc -l)
    
    echo "  - gitdata: $gitdata_count .wav files"
    echo "  - figdata: $figdata_count .wav files"
    echo "  - Total: $((gitdata_count + figdata_count)) .wav files"
else
    echo "⚠️  Data directory not found: ../data"
    echo "   Please ensure the data directory exists and contains audio files"
fi

# Create output directories
echo -e "\nCreating output directories..."
mkdir -p results models
echo "✓ Created: results/"
echo "✓ Created: models/"

echo -e "\n=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo -e "\nNext steps:"
echo "  1. Verify your data is in ../data/"
echo "  2. Run training: python train_xgboost.py"
echo "  3. Run inference: python inference.py --audio <file> --model <model> --scaler <scaler>"
echo -e "\nFor more information, see README.md"
echo "=================================================="
