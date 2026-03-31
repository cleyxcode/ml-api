"""
FastAPI Backend - Sistem Penyiraman Tanaman Berbasis IoT + KNN
Storage  : Supabase (PostgreSQL)
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
import asyncio
import joblib
import numpy as np
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "model", "model_info.json")

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API",
    description="Sistem Penyiraman Tanaman IoT dengan Klasifikasi KNN",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State (model KNN tetap di memory — tidak perlu DB) ────────────────────────
knn_model  = None
scaler     = None
model_meta = {}

HARI_MAP = {
    "senin": 0, "selasa": 1, "rabu": 2, "kamis": 3,
    "jumat": 4, "sabtu": 5, "minggu": 6,
}


# ── Startup: load model + jalankan schedule checker ───────────────────────────
@app.on_event("startup")
async def startup():
    global knn_model, scaler, model_meta

    if not os.path.exists(MODEL_PATH):
        print("[WARN] Model belum ada! Jalankan train_model.py terlebih dahulu.")
        return

    knn_model = joblib.load(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            model_meta = json.load(f)
    else:
        model_meta = {"best_k": "?", "accuracy": "?"}

    print(f"[OK] Model KNN dimuat. K={model_meta.get('best_k')}, Akurasi={model_meta.get('accuracy')}%")
    asyncio.create_task(_schedule_checker())


# ── Schema ────────────────────────────────────────────────────────────────────
class SensorData(BaseModel):
    soil_moisture : float = Field(..., ge=0, le=100)
    temperature   : float = Field(..., ge=0, le=60)
    air_humidity  : float = Field(..., ge=0, le=100)


class ControlCommand(BaseModel):
    action : str = Field(..., description="'on' atau 'off'")
    mode   : Optional[str] = Field("manual", description="'manual' | 'auto' | 'schedule'")


class ScheduleCreate(BaseModel):
    name             : str        = Field(...)
    time             : str        = Field(..., description="HH:MM")
    duration_minutes : int        = Field(default=5, ge=1, le=60)
    days             : List[str]  = Field(default=["senin","selasa","rabu","kamis","jumat","sabtu","minggu"])
    enabled          : bool       = Field(default=True)


class ScheduleUpdate(BaseModel):
    name             : Optional[str]       = None
    time             : Optional[str]       = None
    duration_minutes : Optional[int]       = None
    days             : Optional[List[str]] = None
    enabled          : Optional[bool]      = None


# ── Helper: baca system_state dari Supabase ───────────────────────────────────
def _get_state() -> dict:
    res = supabase.table("system_state").select("*").eq("id", 1).single().execute()
    return res.data or {"pump_status": False, "mode": "auto", "last_label": None, "last_updated": None}


def _update_state(**kwargs):
    supabase.table("system_state").update(kwargs).eq("id", 1).execute()


# ── Helper: klasifikasi KNN ───────────────────────────────────────────────────
def classify(soil_moisture: float, temperature: float, air_humidity: float) -> dict:
    if knn_model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")

    features        = np.array([[soil_moisture, temperature, air_humidity]])
    features_scaled = scaler.transform(features)

    label      = knn_model.predict(features_scaled)[0]
    proba      = knn_model.predict_proba(features_scaled)[0]
    classes    = knn_model.classes_
    confidence = round(float(max(proba)) * 100, 2)
    proba_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}

    return {
        "label"         : label,
        "confidence"    : confidence,
        "probabilities" : proba_dict,
        "needs_watering": label == "Kering",
        "description"   : model_meta.get("label_desc", {}).get(label, ""),
    }


# ── GET / ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status"     : "online",
        "message"    : "Siram Pintar API berjalan",
        "model_ready": knn_model is not None,
    }


# ── POST /sensor ──────────────────────────────────────────────────────────────
@app.post("/sensor")
def receive_sensor(data: SensorData):
    result    = classify(data.soil_moisture, data.temperature, data.air_humidity)
    timestamp = datetime.now().isoformat()
    state     = _get_state()

    # Simpan ke sensor_readings
    supabase.table("sensor_readings").insert({
        "timestamp"    : timestamp,
        "soil_moisture": data.soil_moisture,
        "temperature"  : data.temperature,
        "air_humidity" : data.air_humidity,
        "label"        : result["label"],
        "confidence"   : result["confidence"],
        "needs_watering": result["needs_watering"],
        "description"  : result["description"],
        "probabilities": result["probabilities"],
        "pump_status"  : state["pump_status"],
        "mode"         : state["mode"],
    }).execute()

    # Update system_state
    _update_state(last_label=result["label"], last_updated=timestamp)

    # Kontrol pompa otomatis
    pump_action = None
    if state["mode"] == "auto":
        if result["needs_watering"] and not state["pump_status"]:
            _update_state(pump_status=True)
            pump_action = "on"
        elif not result["needs_watering"] and state["pump_status"]:
            _update_state(pump_status=False)
            pump_action = "off"

    # Baca state terbaru untuk response
    new_state = _get_state()

    return {
        "received"      : True,
        "timestamp"     : timestamp,
        "sensor"        : data.model_dump(),
        "classification": result,
        "pump_status"   : new_state["pump_status"],
        "pump_action"   : pump_action,
        "mode"          : new_state["mode"],
    }


# ── GET /status ───────────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    state = _get_state()

    # Ambil data sensor terbaru
    res = (
        supabase.table("sensor_readings")
        .select("*")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    latest = res.data[0] if res.data else None

    return {
        "pump_status" : state["pump_status"],
        "mode"        : state["mode"],
        "last_label"  : state["last_label"],
        "last_updated": state["last_updated"],
        "latest_data" : latest,
    }


# ── GET /history ──────────────────────────────────────────────────────────────
@app.get("/history")
def get_history(limit: int = 50):
    limit = min(limit, 500)

    res = (
        supabase.table("sensor_readings")
        .select("*")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    records = list(reversed(res.data))   # urut dari lama → baru (untuk grafik)

    return {
        "total"  : len(records),
        "records": records,
    }


# ── POST /control ─────────────────────────────────────────────────────────────
@app.post("/control")
def control_pump(cmd: ControlCommand):
    action = cmd.action.lower()
    if action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'")

    mode = (cmd.mode or "manual").lower()
    if mode not in ("auto", "manual", "schedule"):
        raise HTTPException(status_code=400, detail="Mode harus 'auto', 'manual', atau 'schedule'")

    _update_state(pump_status=(action == "on"), mode=mode)
    state = _get_state()

    return {
        "success"    : True,
        "pump_status": state["pump_status"],
        "mode"       : state["mode"],
        "timestamp"  : datetime.now().isoformat(),
    }


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: SensorData):
    result = classify(data.soil_moisture, data.temperature, data.air_humidity)
    return {"input": data.model_dump(), "result": result}


# ── GET /model-info ───────────────────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


# ── GET /schedules ────────────────────────────────────────────────────────────
@app.get("/schedules")
def get_schedules():
    res = supabase.table("schedules").select("*").order("created_at").execute()
    return {"schedules": res.data}


# ── POST /schedules ───────────────────────────────────────────────────────────
@app.post("/schedules")
def create_schedule(s: ScheduleCreate):
    res = supabase.table("schedules").insert({
        "name"            : s.name,
        "time"            : s.time,
        "duration_minutes": s.duration_minutes,
        "days"            : s.days,
        "enabled"         : s.enabled,
    }).execute()

    return {"success": True, "schedule": res.data[0]}


# ── PUT /schedules/{id} ───────────────────────────────────────────────────────
@app.put("/schedules/{schedule_id}")
def update_schedule(schedule_id: str, s: ScheduleUpdate):
    payload = {k: v for k, v in s.model_dump().items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="Tidak ada field yang diupdate")

    res = supabase.table("schedules").update(payload).eq("id", schedule_id).execute()

    if not res.data:
        raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")

    return {"success": True, "schedule": res.data[0]}


# ── DELETE /schedules/{id} ────────────────────────────────────────────────────
@app.delete("/schedules/{schedule_id}")
def delete_schedule(schedule_id: str):
    res = supabase.table("schedules").delete().eq("id", schedule_id).execute()

    if not res.data:
        raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")

    return {"success": True}


# ── Background: cek jadwal tiap menit ────────────────────────────────────────
async def _schedule_checker():
    while True:
        await asyncio.sleep(60)
        await _run_due_schedules()


async def _run_due_schedules():
    now   = datetime.now()
    h_m   = now.strftime("%H:%M")
    hari  = now.weekday()

    res = supabase.table("schedules").select("*").eq("enabled", True).execute()

    for sc in (res.data or []):
        if sc["time"] != h_m:
            continue

        hari_aktif = [HARI_MAP[d] for d in sc.get("days", []) if d in HARI_MAP]
        if hari not in hari_aktif:
            continue

        # Cek sudah trigger di menit ini?
        last = sc.get("last_triggered")
        if last and last[:16] == now.isoformat()[:16]:
            continue

        # Nyalakan pompa + catat log
        _update_state(pump_status=True, mode="schedule")
        supabase.table("schedules").update({"last_triggered": now.isoformat()}).eq("id", sc["id"]).execute()
        supabase.table("schedule_logs").insert({
            "schedule_id"    : sc["id"],
            "triggered_at"   : now.isoformat(),
            "duration_minutes": sc["duration_minutes"],
        }).execute()

        print(f"[JADWAL] '{sc['name']}' — pompa ON selama {sc['duration_minutes']} menit")
        asyncio.create_task(_stop_pump_after(sc["duration_minutes"], sc["id"], sc["name"]))


async def _stop_pump_after(minutes: int, schedule_id: str, name: str):
    await asyncio.sleep(minutes * 60)
    _update_state(pump_status=False)

    # Update completed_at di schedule_logs
    now = datetime.now().isoformat()
    res = (
        supabase.table("schedule_logs")
        .select("id")
        .eq("schedule_id", schedule_id)
        .is_("completed_at", "null")
        .order("triggered_at", desc=True)
        .limit(1)
        .execute()
    )
    if res.data:
        supabase.table("schedule_logs").update({"completed_at": now}).eq("id", res.data[0]["id"]).execute()

    print(f"[JADWAL] '{name}' selesai — pompa OFF")
