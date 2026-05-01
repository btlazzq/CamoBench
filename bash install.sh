#!/bin/bash

# ===== 环境名 =====
ENV_NAME=agent
PYTHON_VERSION=3.10

echo ">>> Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo ">>> Activating environment"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ">>> Upgrading pip"
pip install --upgrade pip

echo ">>> Installing dependencies"
pip install -r requirements.txt

echo ">>> Done! Environment ready: $ENV_NAME"