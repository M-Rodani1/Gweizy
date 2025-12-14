#!/bin/bash
# Setup script to ensure Python 3.11 and install packages

# Force Python 3.11
export PYENV_VERSION=3.11.9

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install packages using only pre-built wheels
pip install --only-binary :all: pandas numpy scikit-learn || pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.2

# Install rest of requirements
pip install -r requirements.txt
