#!/bin/bash

# Download all required models from Google Drive

echo "Downloading GPT-2 model..."
mkdir -p models/gpt2

# Download GPT-2 model (File ID: 18MEDHiReBKk1nXuJrvSYNNCID-kJ5wiG)
curl -L "https://drive.google.com/uc?export=download&id=18MEDHiReBKk1nXuJrvSYNNCID-kJ5wiG" -o models/gpt2/gpt2_model.zip

echo "Extracting GPT-2 model..."
unzip models/gpt2/gpt2_model.zip -d models/gpt2/
rm models/gpt2/gpt2_model.zip

echo "GPT-2 model downloaded and extracted to models/gpt2/"

echo "Downloading additional model..."
mkdir -p models/

# Download additional model (File ID: 15kO6Yn8Spo90hYBVLTB3a0CauordZSav)
curl -L "https://drive.google.com/uc?export=download&id=15kO6Yn8Spo90hYBVLTB3a0CauordZSav" -o models/additional_model.zip

echo "Extracting additional model..."
unzip models/additional_model.zip -d models/
rm models/additional_model.zip

echo "Additional model downloaded and extracted to models/"

echo "All models downloaded successfully!"
