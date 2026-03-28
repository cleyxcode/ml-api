#!/bin/bash
# Jalankan Backend Siram Pintar

cd "$(dirname "$0")"

# Aktifkan virtual environment
source venv/bin/activate

# Generate dataset jika belum ada
if [ ! -f "dataset/dataset_train.csv" ]; then
    echo "Dataset belum ada, generating..."
    python dataset/generate_dataset.py
fi

# Training model jika belum ada
if [ ! -f "model/knn_model.pkl" ]; then
    echo "Model belum ada, training KNN..."
    python train_model.py
fi

# Jalankan API server
echo "Menjalankan FastAPI server di http://0.0.0.0:8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
