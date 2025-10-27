#!/bin/bash

# Ensure gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "Installing gdown via pip..."
    pip install gdown
fi

# Download the file using gdown
echo "Starting download of FedCola dataset..."
gdown --id 1MhOE4q2P_D3Y5muyz-fhN6GnVSTgbK16 -O fedcola_data.zip