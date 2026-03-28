"""
Training model K-Nearest Neighbor (KNN)
Input fitur: soil_moisture, temperature, air_humidity
Output: label kondisi tanaman (Kering, Lembab, Basah)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATASET_DIR, "dataset_train.csv")
TEST_CSV  = os.path.join(DATASET_DIR, "dataset_test.csv")

FEATURES = ["soil_moisture", "temperature", "air_humidity"]
TARGET   = "label"

# ── 1. Load dataset ───────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print("Dataset belum ada. Menjalankan generate_dataset.py ...")
        from dataset.generate_dataset import generate_dataset, save_dataset
        df = generate_dataset()
        save_dataset(df, DATASET_DIR)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"Data training : {len(train_df)}")
    print(f"Data testing  : {len(test_df)}")
    return train_df, test_df


# ── 2. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(train_df, test_df):
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df[TARGET].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


# ── 3. Cari nilai K optimal ───────────────────────────────────────────────────
def find_best_k(X_train, y_train, k_range=range(1, 21)):
    print("\nMencari nilai K optimal (cross-validation 5-fold)...")
    scores = {}
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        scores[k] = cv_scores.mean()
        print(f"  K={k:2d}  akurasi rata-rata={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    best_k = max(scores, key=scores.get)
    print(f"\nK optimal: {best_k} (akurasi={scores[best_k]:.4f})")
    return best_k, scores


# ── 4. Training ───────────────────────────────────────────────────────────────
def train(X_train, y_train, best_k):
    print(f"\nTraining KNN dengan K={best_k}...")
    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
        weights="uniform",
    )
    knn.fit(X_train, y_train)
    return knn


# ── 5. Evaluasi ───────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"HASIL EVALUASI MODEL KNN")
    print(f"{'='*50}")
    print(f"Akurasi : {acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Basah", "Kering", "Lembab"]))
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["Kering", "Lembab", "Basah"])
    print(cm)

    return acc, y_pred


# ── 6. Simpan model & scaler ──────────────────────────────────────────────────
def save_model(model, scaler, best_k, accuracy, k_scores):
    model_path  = os.path.join(MODEL_DIR, "knn_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path   = os.path.join(MODEL_DIR, "model_info.json")

    with open(model_path,  "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "algorithm"  : "K-Nearest Neighbor",
        "best_k"     : best_k,
        "features"   : FEATURES,
        "labels"     : ["Kering", "Lembab", "Basah"],
        "accuracy"   : round(accuracy * 100, 2),
        "metric"     : "euclidean",
        "k_scores"   : {str(k): round(v, 4) for k, v in k_scores.items()},
        "label_desc" : {
            "Kering": "Kelembaban tanah < 40% — perlu disiram",
            "Lembab": "Kelembaban tanah 40-70% — kondisi optimal",
            "Basah" : "Kelembaban tanah > 70% — tidak perlu disiram"
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel    disimpan : {model_path}")
    print(f"Scaler   disimpan : {scaler_path}")
    print(f"Metadata disimpan : {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test, scaler = preprocess(train_df, test_df)
    best_k, k_scores = find_best_k(X_train, y_train)
    model = train(X_train, y_train, best_k)
    accuracy, _ = evaluate(model, X_test, y_test)
    save_model(model, scaler, best_k, accuracy, k_scores)
    print("\nTraining selesai!")
