#!/bin/bash
# Setup script for Comparing_ML_FrameWorks
set -e

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "\nEnvironment setup complete. Activate with: source venv/bin/activate"
