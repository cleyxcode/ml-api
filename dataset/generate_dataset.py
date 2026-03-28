"""
Generate synthetic dataset untuk training KNN
Parameter: soil_moisture (%), temperature (°C), humidity (%)
Label: Kering (<40%), Lembab (40-70%), Basah (>70%)
Total: 4032 data (14 hari x 24 jam x 12 pembacaan/jam)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

TOTAL_DAYS = 14
INTERVAL_MINUTES = 5
RECORDS_PER_DAY = (24 * 60) // INTERVAL_MINUTES  # 288
TOTAL_RECORDS = TOTAL_DAYS * RECORDS_PER_DAY       # 4032

def get_time_factor(hour):
    """Suhu lebih tinggi siang hari, lebih rendah malam."""
    if 6 <= hour < 10:
        return 0.6   # pagi
    elif 10 <= hour < 15:
        return 1.0   # siang (puncak)
    elif 15 <= hour < 18:
        return 0.8   # sore
    else:
        return 0.3   # malam

def generate_record(timestamp, soil_moisture_base, is_after_watering, weather):
    hour = timestamp.hour

    # --- Suhu udara ---
    base_temp = 28.0
    time_factor = get_time_factor(hour)
    if weather == "hujan":
        temp = base_temp * 0.85 + np.random.normal(0, 0.5)
    else:
        temp = base_temp + (time_factor * 8) + np.random.normal(0, 1.0)
    temp = round(np.clip(temp, 18.0, 40.0), 1)

    # --- Kelembaban udara ---
    if weather == "hujan":
        air_humidity = round(np.random.uniform(80, 98), 1)
    elif weather == "berawan":
        air_humidity = round(np.random.uniform(65, 85), 1)
    else:  # cerah
        if 10 <= hour < 15:
            air_humidity = round(np.random.uniform(40, 65), 1)
        else:
            air_humidity = round(np.random.uniform(55, 80), 1)

    # --- Kelembaban tanah ---
    # Evaporasi berdasarkan suhu dan waktu
    evaporation = 0.0
    if 10 <= hour < 15:
        evaporation = np.random.uniform(0.3, 0.8)
    elif 6 <= hour < 10 or 15 <= hour < 18:
        evaporation = np.random.uniform(0.1, 0.4)
    else:
        evaporation = np.random.uniform(0.0, 0.1)

    if weather == "hujan":
        soil_increment = np.random.uniform(2.0, 5.0)
    else:
        soil_increment = 0.0

    soil_moisture = soil_moisture_base - evaporation + soil_increment + np.random.normal(0, 0.5)
    soil_moisture = round(np.clip(soil_moisture, 0.0, 100.0), 1)

    # --- Label ---
    if soil_moisture < 40:
        label = "Kering"
    elif soil_moisture <= 70:
        label = "Lembab"
    else:
        label = "Basah"

    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "soil_moisture": soil_moisture,
        "temperature": temp,
        "air_humidity": air_humidity,
        "label": label
    }, soil_moisture


def generate_dataset():
    data = []
    start_date = datetime(2025, 1, 1, 0, 0, 0)

    # Pola cuaca per hari (campuran cerah, berawan, hujan)
    weather_pattern = []
    for day in range(TOTAL_DAYS):
        roll = random.random()
        if roll < 0.5:
            weather_pattern.append("cerah")
        elif roll < 0.8:
            weather_pattern.append("berawan")
        else:
            weather_pattern.append("hujan")

    soil_moisture = random.uniform(45, 65)  # mulai kondisi lembab
    watering_cooldown = 0

    for i in range(TOTAL_RECORDS):
        timestamp = start_date + timedelta(minutes=i * INTERVAL_MINUTES)
        day_index = i // RECORDS_PER_DAY
        weather = weather_pattern[day_index]

        record, soil_moisture = generate_record(timestamp, soil_moisture, watering_cooldown > 0, weather)

        # Simulasi penyiraman otomatis saat kering
        if record["label"] == "Kering" and watering_cooldown == 0:
            soil_moisture = min(soil_moisture + random.uniform(25, 35), 85)
            watering_cooldown = 12  # cooldown 12 interval (1 jam)

        if watering_cooldown > 0:
            watering_cooldown -= 1

        data.append(record)

    df = pd.DataFrame(data)
    return df


def save_dataset(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Simpan full dataset
    full_path = os.path.join(output_dir, "dataset_full.csv")
    df.to_csv(full_path, index=False)
    print(f"Dataset lengkap disimpan: {full_path} ({len(df)} data)")

    # Distribusi label
    print("\nDistribusi label:")
    print(df["label"].value_counts())
    print(f"\nPersentase:")
    print(df["label"].value_counts(normalize=True).map(lambda x: f"{x*100:.1f}%"))

    # Split train/test (80/20) stratified
    from sklearn.model_selection import train_test_split
    X = df[["soil_moisture", "temperature", "air_humidity"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df  = pd.concat([X_test,  y_test],  axis=1).reset_index(drop=True)

    train_path = os.path.join(output_dir, "dataset_train.csv")
    test_path  = os.path.join(output_dir, "dataset_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nData training : {len(train_df)} data -> {train_path}")
    print(f"Data testing  : {len(test_df)} data  -> {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    train_df, test_df = save_dataset(df, output_dir)

    print("\nContoh 5 data pertama:")
    print(df.head())
    print("\nStatistik dataset:")
    print(df[["soil_moisture", "temperature", "air_humidity"]].describe())
    print("\nDataset berhasil dibuat!")
