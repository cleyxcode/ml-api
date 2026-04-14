"""
FastAPI Backend - Sistem Penyiraman Tanaman Berbasis IoT + KNN
Storage  : MySQL (Hostinger)
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
from typing import Optional, List
from contextlib import contextmanager

import pymysql
import pymysql.cursors
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "model", "model_info.json")

# ── MySQL config ──────────────────────────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "srv1987.hstgr.io")
DB_PORT = int(os.environ.get("DB_PORT", 3306))
DB_USER = os.environ.get("DB_USER", "")
DB_PASS = os.environ.get("DB_PASS", "")
DB_NAME = os.environ.get("DB_NAME", "")

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

# ── KNN model (di memory) ─────────────────────────────────────────────────────
knn_model  = None
scaler     = None
model_meta = {}

HARI_MAP = {
    "senin": 0, "selasa": 1, "rabu": 2, "kamis": 3,
    "jumat": 4, "sabtu": 5, "minggu": 6,
}


# ── Helper: koneksi MySQL ─────────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15,
        ssl={"ssl": {}},        # aktifkan SSL — wajib untuk Hostinger remote
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Startup ───────────────────────────────────────────────────────────────────
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
    action : str           = Field(..., description="'on' atau 'off'")
    mode   : Optional[str] = Field("manual")


class ScheduleCreate(BaseModel):
    name             : str       = Field(...)
    time             : str       = Field(..., description="HH:MM")
    duration_minutes : int       = Field(default=5, ge=1, le=60)
    days             : List[str] = Field(default=["senin","selasa","rabu","kamis","jumat","sabtu","minggu"])
    enabled          : bool      = Field(default=True)


class ScheduleUpdate(BaseModel):
    name             : Optional[str]       = None
    time             : Optional[str]       = None
    duration_minutes : Optional[int]       = None
    days             : Optional[List[str]] = None
    enabled          : Optional[bool]      = None


# ── Helper: system_state ──────────────────────────────────────────────────────
def _get_state() -> dict:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_state WHERE id = 1")
            row = cur.fetchone()
    if row:
        row["pump_status"] = bool(row["pump_status"])
    return row or {"pump_status": False, "mode": "auto", "last_label": None, "last_updated": None}


def _update_state(**kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k} = %s" for k in kwargs)
    vals = list(kwargs.values())
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE system_state SET {sets} WHERE id = 1", vals)


# ── Helper: KNN classify ──────────────────────────────────────────────────────
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


# ── GET /db-test ── diagnosis koneksi database ────────────────────────────────
@app.get("/db-test")
def db_test():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
        return {"db_status": "connected", "result": row}
    except Exception as e:
        return {"db_status": "error", "detail": str(e)}


# ── POST /sensor ──────────────────────────────────────────────────────────────
@app.post("/sensor")
def receive_sensor(data: SensorData):
    result    = classify(data.soil_moisture, data.temperature, data.air_humidity)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state     = _get_state()
    row_id    = str(uuid.uuid4())

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sensor_readings
                    (id, timestamp, soil_moisture, temperature, air_humidity,
                     label, confidence, needs_watering, description, probabilities,
                     pump_status, mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row_id, timestamp,
                data.soil_moisture, data.temperature, data.air_humidity,
                result["label"], result["confidence"], result["needs_watering"],
                result["description"], json.dumps(result["probabilities"]),
                state["pump_status"], state["mode"],
            ))

    _update_state(last_label=result["label"], last_updated=timestamp)

    pump_action = None
    if state["mode"] == "auto":
        if result["needs_watering"] and not state["pump_status"]:
            _update_state(pump_status=True)
            pump_action = "on"
        elif not result["needs_watering"] and state["pump_status"]:
            _update_state(pump_status=False)
            pump_action = "off"

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

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM sensor_readings
                ORDER BY timestamp DESC LIMIT 1
            """)
            latest = cur.fetchone()

    if latest and isinstance(latest.get("probabilities"), str):
        latest["probabilities"] = json.loads(latest["probabilities"])
    if latest:
        latest["pump_status"]   = bool(latest["pump_status"])
        latest["needs_watering"]= bool(latest["needs_watering"])

    return {
        "pump_status" : state["pump_status"],
        "mode"        : state["mode"],
        "last_label"  : state["last_label"],
        "last_updated": str(state["last_updated"]) if state["last_updated"] else None,
        "latest_data" : latest,
    }


# ── GET /history ──────────────────────────────────────────────────────────────
@app.get("/history")
def get_history(limit: int = 50):
    limit = min(limit, 500)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM sensor_readings
                ORDER BY timestamp DESC LIMIT %s
            """, (limit,))
            rows = cur.fetchall()

    records = []
    for r in reversed(rows):
        if isinstance(r.get("probabilities"), str):
            r["probabilities"] = json.loads(r["probabilities"])
        r["pump_status"]    = bool(r["pump_status"])
        r["needs_watering"] = bool(r["needs_watering"])
        records.append(r)

    return {"total": len(records), "records": records}


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
    return {"input": data.model_dump(), "result": classify(data.soil_moisture, data.temperature, data.air_humidity)}


# ── GET /model-info ───────────────────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


# ── GET /schedules ────────────────────────────────────────────────────────────
@app.get("/schedules")
def get_schedules():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedules ORDER BY created_at")
            rows = cur.fetchall()
    for r in rows:
        if isinstance(r.get("days"), str):
            r["days"] = json.loads(r["days"])
        r["enabled"] = bool(r["enabled"])
    return {"schedules": rows}


# ── POST /schedules ───────────────────────────────────────────────────────────
@app.post("/schedules")
def create_schedule(s: ScheduleCreate):
    new_id = str(uuid.uuid4())
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO schedules (id, name, time, duration_minutes, days, enabled)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (new_id, s.name, s.time, s.duration_minutes, json.dumps(s.days), s.enabled))
            cur.execute("SELECT * FROM schedules WHERE id = %s", (new_id,))
            row = cur.fetchone()

    if isinstance(row.get("days"), str):
        row["days"] = json.loads(row["days"])
    row["enabled"] = bool(row["enabled"])
    return {"success": True, "schedule": row}


# ── PUT /schedules/{id} ───────────────────────────────────────────────────────
@app.put("/schedules/{schedule_id}")
def update_schedule(schedule_id: str, s: ScheduleUpdate):
    payload = {k: v for k, v in s.model_dump().items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="Tidak ada field yang diupdate")

    if "days" in payload:
        payload["days"] = json.dumps(payload["days"])

    sets = ", ".join(f"{k} = %s" for k in payload)
    vals = list(payload.values()) + [schedule_id]

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE schedules SET {sets} WHERE id = %s", vals)
            cur.execute("SELECT * FROM schedules WHERE id = %s", (schedule_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")

    if isinstance(row.get("days"), str):
        row["days"] = json.loads(row["days"])
    row["enabled"] = bool(row["enabled"])
    return {"success": True, "schedule": row}


# ── DELETE /schedules/{id} ────────────────────────────────────────────────────
@app.delete("/schedules/{schedule_id}")
def delete_schedule(schedule_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM schedules WHERE id = %s", (schedule_id,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")
    return {"success": True}


# ── Background: cek jadwal tiap menit ────────────────────────────────────────
async def _schedule_checker():
    while True:
        await asyncio.sleep(60)
        try:
            await _run_due_schedules()
        except Exception as e:
            print(f"[JADWAL ERR] {e}")


async def _run_due_schedules():
    now  = datetime.now()
    h_m  = now.strftime("%H:%M")
    hari = now.weekday()

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedules WHERE enabled = 1")
            schedules = cur.fetchall()

    for sc in schedules:
        if sc["time"] != h_m:
            continue

        days = json.loads(sc["days"]) if isinstance(sc["days"], str) else sc["days"]
        hari_aktif = [HARI_MAP[d] for d in days if d in HARI_MAP]
        if hari not in hari_aktif:
            continue

        last = sc.get("last_triggered")
        if last and str(last)[:16] == now.isoformat()[:16]:
            continue

        _update_state(pump_status=True, mode="schedule")

        log_id = str(uuid.uuid4())
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE schedules SET last_triggered = %s WHERE id = %s",
                            (now.strftime("%Y-%m-%d %H:%M:%S"), sc["id"]))
                cur.execute("""
                    INSERT INTO schedule_logs (id, schedule_id, triggered_at, duration_minutes)
                    VALUES (%s, %s, %s, %s)
                """, (log_id, sc["id"], now.strftime("%Y-%m-%d %H:%M:%S"), sc["duration_minutes"]))

        print(f"[JADWAL] '{sc['name']}' — pompa ON selama {sc['duration_minutes']} menit")
        asyncio.create_task(_stop_pump_after(sc["duration_minutes"], log_id, sc["name"]))


async def _stop_pump_after(minutes: int, log_id: str, name: str):
    await asyncio.sleep(minutes * 60)
    _update_state(pump_status=False, mode="auto")   # reset mode ke auto setelah jadwal selesai

    completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE schedule_logs SET completed_at = %s WHERE id = %s",
                        (completed, log_id))

    print(f"[JADWAL] '{name}' selesai — pompa OFF, mode kembali ke auto")
