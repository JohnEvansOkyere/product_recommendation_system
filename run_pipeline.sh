#!/bin/bash

# PyTorch Recommendation System Pipeline Runner
# This script runs the complete recommendation system pipeline

echo "=========================================="
echo "PyTorch Recommendation System Pipeline"
echo "=========================================="

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate recsys

# Check if environment is activated
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'recsys'"
    echo "Please make sure the environment exists: conda create -n recsys python=3.10"
    exit 1
fi

echo "✅ Conda environment activated: recsys"

# Install dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt

# Run the complete pipeline
echo "🚀 Starting the recommendation system pipeline..."
echo "This will:"
echo "1. Download and extract data"
echo "2. Preprocess and clean data"
echo "3. Engineer features"
echo "4. Train the PyTorch model"
echo "5. Evaluate the model"
echo ""

python main.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Pipeline completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run Streamlit app: streamlit run app.py"
    echo "2. Or run individual stages using the pipeline"
    echo ""
else
    echo ""
    echo "❌ Pipeline failed. Check the logs for details."
    exit 1
fi
