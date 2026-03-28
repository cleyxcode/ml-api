"""
FastAPI Backend - Sistem Penyiraman Tanaman Berbasis IoT + KNN
Endpoint:
  POST /sensor          - terima data sensor dari ESP32, klasifikasi KNN
  GET  /status          - status terakhir sistem
  GET  /history         - riwayat data sensor
  POST /control         - kontrol manual pompa (ON/OFF)
  GET  /model-info      - informasi model KNN
  GET  /schedules       - daftar jadwal penyiraman
  POST /schedules       - buat jadwal baru
  PUT  /schedules/{id}  - update jadwal
  DELETE /schedules/{id}- hapus jadwal
"""

import os
import json
import uuid
import asyncio
import joblib
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "model", "model_info.json")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API",
    description="Sistem Penyiraman Tanaman IoT dengan Klasifikasi KNN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ─────────────────────────────────────────────────────────────────────
knn_model   = None
scaler      = None
model_meta  = {}
history     = deque(maxlen=500)   # simpan 500 data terakhir di memory

system_state = {
    "pump_status"   : False,       # True = ON, False = OFF
    "mode"          : "auto",      # "auto" | "manual" | "schedule"
    "last_label"    : None,
    "last_updated"  : None,
}

# Jadwal penyiraman — disimpan di memory (list of dict)
schedules: List[dict] = []


# ── Load model saat startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global knn_model, scaler, model_meta

    if not os.path.exists(MODEL_PATH):
        print("Model belum ada! Jalankan train_model.py terlebih dahulu.")
        return

    knn_model = joblib.load(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            model_meta = json.load(f)
    else:
        model_meta = {"best_k": "?", "accuracy": "?"}

    print(f"Model KNN dimuat. K={model_meta.get('best_k')}, Akurasi={model_meta.get('accuracy')}%")

    # Jalankan background task cek jadwal setiap menit
    asyncio.create_task(_schedule_checker())


# ── Schema ────────────────────────────────────────────────────────────────────
class SensorData(BaseModel):
    soil_moisture : float = Field(..., ge=0, le=100, description="Kelembaban tanah (%)")
    temperature   : float = Field(..., ge=0, le=60,  description="Suhu udara (°C)")
    air_humidity  : float = Field(..., ge=0, le=100, description="Kelembaban udara (%)")


class ControlCommand(BaseModel):
    action : str = Field(..., description="'on' atau 'off'")
    mode   : Optional[str] = Field("manual", description="'manual' | 'auto' | 'schedule'")


class ScheduleCreate(BaseModel):
    name             : str   = Field(..., description="Nama jadwal, mis: Pagi, Sore")
    time             : str   = Field(..., description="Waktu HH:MM, mis: 06:00")
    duration_minutes : int   = Field(default=5, ge=1, le=60, description="Durasi siram (menit)")
    days             : List[str] = Field(
        default=["senin","selasa","rabu","kamis","jumat","sabtu","minggu"],
        description="Hari aktif: senin, selasa, ..., minggu  atau 'setiap hari'"
    )
    enabled          : bool  = Field(default=True)


class ScheduleUpdate(BaseModel):
    name             : Optional[str]       = None
    time             : Optional[str]       = None
    duration_minutes : Optional[int]       = None
    days             : Optional[List[str]] = None
    enabled          : Optional[bool]      = None


# ── Helper KNN ────────────────────────────────────────────────────────────────
def classify(soil_moisture: float, temperature: float, air_humidity: float) -> dict:
    if knn_model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat. Jalankan train_model.py.")

    features = np.array([[soil_moisture, temperature, air_humidity]])
    features_scaled = scaler.transform(features)

    label       = knn_model.predict(features_scaled)[0]
    proba       = knn_model.predict_proba(features_scaled)[0]
    classes     = knn_model.classes_
    confidence  = round(float(max(proba)) * 100, 2)

    proba_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}

    needs_watering = label == "Kering"

    return {
        "label"          : label,
        "confidence"     : confidence,
        "probabilities"  : proba_dict,
        "needs_watering" : needs_watering,
        "description"    : model_meta.get("label_desc", {}).get(label, ""),
    }


# ── Endpoint: Health Check ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status"     : "online",
        "message"    : "Siram Pintar API berjalan",
        "model_ready": knn_model is not None,
    }


# ── Endpoint: Terima data sensor dari ESP32 ───────────────────────────────────
@app.post("/sensor")
def receive_sensor(data: SensorData):
    result = classify(data.soil_moisture, data.temperature, data.air_humidity)

    timestamp = datetime.now().isoformat()

    record = {
        "timestamp"    : timestamp,
        "soil_moisture": data.soil_moisture,
        "temperature"  : data.temperature,
        "air_humidity" : data.air_humidity,
        **result,
    }
    history.append(record)

    # Update system state
    system_state["last_label"]   = result["label"]
    system_state["last_updated"] = timestamp

    # Kontrol pompa otomatis (mode auto)
    pump_action = None
    if system_state["mode"] == "auto":
        if result["needs_watering"] and not system_state["pump_status"]:
            system_state["pump_status"] = True
            pump_action = "on"
        elif not result["needs_watering"] and system_state["pump_status"]:
            system_state["pump_status"] = False
            pump_action = "off"

    return {
        "received"   : True,
        "timestamp"  : timestamp,
        "sensor"     : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
        "classification": result,
        "pump_status"   : system_state["pump_status"],
        "pump_action"   : pump_action,
        "mode"          : system_state["mode"],
    }


# ── Endpoint: Status sistem terkini ───────────────────────────────────────────
@app.get("/status")
def get_status():
    latest = history[-1] if history else None
    return {
        "pump_status" : system_state["pump_status"],
        "mode"        : system_state["mode"],
        "last_label"  : system_state["last_label"],
        "last_updated": system_state["last_updated"],
        "latest_data" : latest,
    }


# ── Endpoint: Riwayat data ────────────────────────────────────────────────────
@app.get("/history")
def get_history(limit: int = 50):
    limit = min(limit, 500)
    data  = list(history)[-limit:]
    return {
        "total"  : len(data),
        "records": data,
    }


# ── Endpoint: Kontrol manual pompa ───────────────────────────────────────────
@app.post("/control")
def control_pump(cmd: ControlCommand):
    action = cmd.action.lower()
    if action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'")

    mode = cmd.mode.lower() if cmd.mode else "manual"
    if mode not in ("auto", "manual", "schedule"):
        raise HTTPException(status_code=400, detail="Mode harus 'auto', 'manual', atau 'schedule'")

    system_state["pump_status"] = (action == "on")
    system_state["mode"]        = mode

    return {
        "success"    : True,
        "pump_status": system_state["pump_status"],
        "mode"       : system_state["mode"],
        "timestamp"  : datetime.now().isoformat(),
    }


# ── Endpoint: Klasifikasi manual (tanpa sensor) ───────────────────────────────
@app.post("/predict")
def predict(data: SensorData):
    result = classify(data.soil_moisture, data.temperature, data.air_humidity)
    return {
        "input" : data.dict(),
        "result": result,
    }


# ── Endpoint: Info model ──────────────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


# ── Jadwal penyiraman ─────────────────────────────────────────────────────────
HARI_MAP = {
    "senin": 0, "selasa": 1, "rabu": 2, "kamis": 3,
    "jumat": 4, "sabtu": 5, "minggu": 6,
}


@app.get("/schedules")
def get_schedules():
    return {"schedules": schedules}


@app.post("/schedules")
def create_schedule(s: ScheduleCreate):
    schedule = {
        "id"              : str(uuid.uuid4()),
        "name"            : s.name,
        "time"            : s.time,
        "duration_minutes": s.duration_minutes,
        "days"            : s.days,
        "enabled"         : s.enabled,
        "created_at"      : datetime.now().isoformat(),
        "last_triggered"  : None,
    }
    schedules.append(schedule)
    return {"success": True, "schedule": schedule}


@app.put("/schedules/{schedule_id}")
def update_schedule(schedule_id: str, s: ScheduleUpdate):
    for sc in schedules:
        if sc["id"] == schedule_id:
            if s.name             is not None: sc["name"]             = s.name
            if s.time             is not None: sc["time"]             = s.time
            if s.duration_minutes is not None: sc["duration_minutes"] = s.duration_minutes
            if s.days             is not None: sc["days"]             = s.days
            if s.enabled          is not None: sc["enabled"]          = s.enabled
            return {"success": True, "schedule": sc}
    raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")


@app.delete("/schedules/{schedule_id}")
def delete_schedule(schedule_id: str):
    for i, sc in enumerate(schedules):
        if sc["id"] == schedule_id:
            schedules.pop(i)
            return {"success": True}
    raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")


# ── Background task: cek jadwal tiap menit ────────────────────────────────────
async def _schedule_checker():
    """Berjalan di background — cek tiap menit apakah ada jadwal yang harus dijalankan."""
    while True:
        await asyncio.sleep(60)   # cek setiap menit
        _run_due_schedules()


def _run_due_schedules():
    now    = datetime.now()
    h_m    = now.strftime("%H:%M")
    hari   = now.weekday()   # 0=senin … 6=minggu

    for sc in schedules:
        if not sc["enabled"]:
            continue
        if sc["time"] != h_m:
            continue

        # Cek hari
        days = sc.get("days", [])
        hari_aktif = [HARI_MAP[d] for d in days if d in HARI_MAP]
        if hari not in hari_aktif:
            continue

        # Jangan trigger 2x dalam 1 menit
        last = sc.get("last_triggered")
        if last and last[:16] == now.isoformat()[:16]:
            continue

        # Jalankan penyiraman
        system_state["pump_status"] = True
        system_state["mode"]        = "schedule"
        sc["last_triggered"]        = now.isoformat()

        print(f"[JADWAL] '{sc['name']}' — pompa ON selama {sc['duration_minutes']} menit")

        # Matikan pompa setelah durasi (non-blocking)
        duration = sc["duration_minutes"]
        asyncio.create_task(_stop_pump_after(duration, sc["name"]))


async def _stop_pump_after(minutes: int, name: str):
    """Matikan pompa setelah durasi jadwal selesai."""
    await asyncio.sleep(minutes * 60)
    system_state["pump_status"] = False
    print(f"[JADWAL] '{name}' selesai — pompa OFF")
