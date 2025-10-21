#!/bin/bash

# VariantProject Installation Script
# Detects platform and installs dependencies appropriately

echo "======================================"
echo "VariantProject v2.0 Installation"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect Python
if command_exists python3; then
    PYTHON=python3
elif command_exists python; then
    PYTHON=python
else
    echo "Error: Python not found. Please install Python 3.7 or higher."
    exit 1
fi

echo "Using Python: $PYTHON"
$PYTHON --version

# Check Python version
$PYTHON -c "import sys; exit(0 if sys.version_info >= (3,7) else 1)" || {
    echo "Error: Python 3.7 or higher is required"
    exit 1
}

# Detect package manager
if command_exists conda; then
    echo "Conda detected. Using conda for installation..."
    USE_CONDA=true
else
    echo "Conda not found. Will use pip..."
    USE_CONDA=false
fi

# Function to install with pip
install_with_pip() {
    echo "Installing base dependencies with pip..."
    $PYTHON -m pip install --upgrade pip
    $PYTHON -m pip install pandas numpy matplotlib seaborn scikit-learn
    
    echo "Attempting to install RDKit..."
    if $PYTHON -m pip install rdkit-pypi; then
        echo "✓ RDKit installed successfully"
    else
        echo "⚠ RDKit installation failed. You may need to use conda or the simplified demo."
    fi
    
    echo "Installing optional dependencies..."
    $PYTHON -m pip install py3Dmol selfies tqdm ipython jupyter
}

# Function to install with conda
install_with_conda() {
    echo "Installing with conda..."
    
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "variantproject"; then
        echo "Creating conda environment 'variantproject'..."
        conda create -n variantproject python=3.8 -y
    fi
    
    echo "Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate variantproject
    
    echo "Installing RDKit from conda-forge..."
    conda install -c conda-forge rdkit -y
    
    echo "Installing other dependencies..."
    pip install pandas numpy matplotlib seaborn scikit-learn
    pip install py3Dmol selfies tqdm ipython jupyter
    
    echo ""
    echo "✓ Installation complete!"
    echo "To use VariantProject, activate the environment with:"
    echo "  conda activate variantproject"
}

# Main installation
if [ "$USE_CONDA" = true ]; then
    install_with_conda
else
    install_with_pip
fi

# Test installation
echo ""
echo "Testing installation..."
$PYTHON -c "
try:
    import pandas
    print('✓ Pandas installed')
except:
    print('✗ Pandas not found')

try:
    import numpy
    print('✓ NumPy installed')
except:
    print('✗ NumPy not found')

try:
    import sklearn
    print('✓ Scikit-learn installed')
except:
    print('✗ Scikit-learn not found')

try:
    import matplotlib
    print('✓ Matplotlib installed')
except:
    print('✗ Matplotlib not found')

try:
    from rdkit import Chem
    print('✓ RDKit installed and working')
except:
    print('✗ RDKit not available - use molecular_explorer_demo.py instead')

try:
    import selfies
    print('✓ SELFIES installed')
except:
    print('✗ SELFIES not found (optional)')

try:
    import py3Dmol
    print('✓ py3Dmol installed')
except:
    print('✗ py3Dmol not found (optional)')
"

echo ""
echo "======================================"
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. For full version: python VariantProject_v2.py"
echo "2. For demo without RDKit: python molecular_explorer_demo.py"
echo "3. For Jupyter: jupyter notebook VariantProject_Complete_Notebook.ipynb"
echo "======================================"
