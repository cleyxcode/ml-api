import os
import json
import uuid
import asyncio
import logging
import joblib
import numpy as np
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

import pymysql
import pymysql.cursors
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("siram-pintar")

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

# ── Konstanta ─────────────────────────────────────────────────────────────────
# Map hari dari angka ESP32 (Sunday=0) ke string database
DAY_MAP = {
    0: "minggu", 1: "senin", 2: "selasa",
    3: "rabu", 4: "kamis", 5: "jumat", 6: "sabtu"
}

# Map hari string → angka (untuk cek jadwal)
HARI_MAP = {
    "senin": 1, "selasa": 2, "rabu": 3, "kamis": 4,
    "jumat": 5, "sabtu": 6, "minggu": 0,
}

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI LOGIKA PENYIRAMAN CERDAS
# ══════════════════════════════════════════════════════════════════════════════
class WateringConfig:
    ALLOWED_WINDOWS             = [(5, 7), (16, 18)]  # Pagi & Sore
    RAIN_HUMIDITY_THRESHOLD     = 85.0
    HOT_TEMP_THRESHOLD          = 35.0
    CONFIDENCE_THRESHOLD_NORMAL = 60.0
    CONFIDENCE_THRESHOLD_HOT    = 40.0
    MAX_PUMP_DURATION_MINUTES   = 2    # Maks pompa nyala di mode AUTO
    COOLDOWN_MINUTES            = 30   # Jeda antar penyiraman di mode AUTO

CFG = WateringConfig()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API",
    description="Sistem Penyiraman Tanaman IoT dengan Klasifikasi KNN",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── KNN model (in-memory) ─────────────────────────────────────────────────────
knn_model  = None
scaler     = None
model_meta: dict = {}

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE  — pola dari v1 (terbukti jalan), tanpa bungkus HTTPException
# ══════════════════════════════════════════════════════════════════════════════
@contextmanager
def get_db():
    """
    Context manager koneksi MySQL.
    PENTING: tidak membungkus Exception menjadi HTTPException supaya
    stack trace asli tetap terlihat dan FastAPI bisa log dengan benar.
    """
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15,
        ssl={"ssl": {}},   # wajib untuk Hostinger remote
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise           # lempar ulang error asli — JANGAN wrap ke HTTPException
    finally:
        conn.close()


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global knn_model, scaler, model_meta

    if not os.path.exists(MODEL_PATH):
        log.warning("Model belum ada! Jalankan train_model.py terlebih dahulu.")
        return

    try:
        knn_model = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                model_meta = json.load(f)
        else:
            model_meta = {"best_k": "?", "accuracy": "?"}
        log.info("Model KNN dimuat. K=%s, Akurasi=%s%%",
                 model_meta.get("best_k"), model_meta.get("accuracy"))
    except Exception as exc:
        log.error("Gagal memuat model: %s", exc)

    # Jadwal checker seperti v1
    asyncio.create_task(_schedule_checker())


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA / MODEL
# ══════════════════════════════════════════════════════════════════════════════
class SensorData(BaseModel):
    soil_moisture : float = Field(..., ge=0, le=100)
    temperature   : float = Field(..., ge=0, le=60)
    air_humidity  : float = Field(..., ge=0, le=100)
    # FIX: hour/minute/day bersifat Optional dengan default fallback ke waktu server
    # Sehingga ESP32 lama yang tidak mengirim field ini tetap bisa diterima
    hour   : Optional[int] = Field(default=None, ge=0, le=23)
    minute : Optional[int] = Field(default=None, ge=0, le=59)
    day    : Optional[int] = Field(default=None, ge=0, le=6)


class ControlCommand(BaseModel):
    action : str           = Field(..., description="'on' atau 'off'")
    mode   : Optional[str] = Field(default="manual")


class ScheduleCreate(BaseModel):
    name             : str       = Field(...)
    time             : str       = Field(..., description="HH:MM")
    duration_minutes : int       = Field(default=5, ge=1, le=60)
    days             : List[str] = Field(
        default=["senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
    )
    enabled          : bool      = Field(default=True)


class ScheduleUpdate(BaseModel):
    name             : Optional[str]       = None
    time             : Optional[str]       = None
    duration_minutes : Optional[int]       = None
    days             : Optional[List[str]] = None
    enabled          : Optional[bool]      = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: waktu
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_time(hour: Optional[int], minute: Optional[int], day: Optional[int]):
    """
    Jika ESP32 mengirim hour/minute/day gunakan itu.
    Jika tidak, fallback ke waktu server (seperti v1).
    """
    if hour is not None and minute is not None and day is not None:
        return hour, minute, day
    now = datetime.now()
    return now.hour, now.minute, now.weekday()  # weekday(): Mon=0 … Sun=6


def get_elapsed_minutes(current_minutes: int, stored_minutes) -> int:
    """Selisih menit, tangani rollover 24 jam (1440 menit)."""
    if stored_minutes is None:
        return 999999
    diff = current_minutes - int(stored_minutes)
    return diff if diff >= 0 else diff + 1440


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: system_state  — pola dari v1
# ══════════════════════════════════════════════════════════════════════════════
def _get_state() -> dict:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_state WHERE id = 1")
            row = cur.fetchone()
    if row:
        row["pump_status"] = bool(row["pump_status"])
        return row
    return {
        "pump_status"        : False,
        "mode"               : "auto",
        "last_label"         : None,
        "last_updated"       : None,
        "pump_start_minute"  : None,
        "last_watered_minute": None,
    }


def _update_state(**kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k} = %s" for k in kwargs)
    vals = list(kwargs.values())
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE system_state SET {sets} WHERE id = 1", vals)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: KNN classify  — pola dari v1
# ══════════════════════════════════════════════════════════════════════════════
def classify(soil_moisture: float, temperature: float, air_humidity: float) -> dict:
    if knn_model is None or scaler is None:
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


# ══════════════════════════════════════════════════════════════════════════════
# LOGIKA SMART WATERING (MODE AUTO)
# ══════════════════════════════════════════════════════════════════════════════
def _evaluate_smart_watering(
    result: dict,
    hour: int,
    minute: int,
    air_humidity: float,
    temperature: float,
    state: dict,
    current_total_minutes: int,
) -> dict:
    response = {
        "action"        : None,
        "reason"        : "",
        "hot_mode"      : temperature >= CFG.HOT_TEMP_THRESHOLD,
        "blocked_reason": None,
    }

    # Pompa sedang nyala → cek apakah harus dimatikan
    if state["pump_status"]:
        elapsed_run = get_elapsed_minutes(
            current_total_minutes, state.get("pump_start_minute")
        )
        if elapsed_run >= CFG.MAX_PUMP_DURATION_MINUTES:
            _update_state(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                pump_start_minute=None,
            )
            response["action"] = "off"
            response["reason"] = (
                f"Auto-stop: pompa melebihi {CFG.MAX_PUMP_DURATION_MINUTES} menit."
            )
            return response

        if not result["needs_watering"]:
            _update_state(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                pump_start_minute=None,
            )
            response["action"] = "off"
            response["reason"] = f"Tanah sudah lembap ({result['label']})."
            return response

        response["reason"] = "Pompa tetap ON."
        return response

    # Pompa mati → cek apakah perlu dinyalakan
    in_window = any(s <= hour <= e for s, e in CFG.ALLOWED_WINDOWS)
    if not in_window:
        response["blocked_reason"] = (
            f"Di luar jam aman. Jam alat: {hour:02d}:{minute:02d}."
        )
        return response

    if air_humidity >= CFG.RAIN_HUMIDITY_THRESHOLD:
        response["blocked_reason"] = "Kelembapan udara tinggi (indikasi hujan)."
        return response

    elapsed_cd = get_elapsed_minutes(
        current_total_minutes, state.get("last_watered_minute")
    )
    if elapsed_cd < CFG.COOLDOWN_MINUTES:
        remaining = CFG.COOLDOWN_MINUTES - elapsed_cd
        response["blocked_reason"] = f"Cooldown. Sisa: {remaining} menit."
        return response

    if not result["needs_watering"]:
        response["blocked_reason"] = f"Tanah tidak kering ({result['label']})."
        return response

    threshold = (
        CFG.CONFIDENCE_THRESHOLD_HOT
        if response["hot_mode"]
        else CFG.CONFIDENCE_THRESHOLD_NORMAL
    )
    if result["confidence"] < threshold:
        response["blocked_reason"] = (
            f"Confidence KNN rendah ({result['confidence']}% < {threshold}%)."
        )
        return response

    _update_state(pump_status=True, pump_start_minute=current_total_minutes)
    response["action"] = "on"
    response["reason"] = (
        f"Siram Pintar ON: {result['label']} ({result['confidence']}%)."
    )
    return response


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS UMUM
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status"     : "online",
        "message"    : "Siram Pintar API berjalan",
        "version"    : "3.0.0",
        "model_ready": knn_model is not None,
    }


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


@app.get("/model-info")
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT SENSOR — logika utama ESP32
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/sensor")
def receive_sensor(data: SensorData):
    result    = classify(data.soil_moisture, data.temperature, data.air_humidity)
    state     = _get_state()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_id    = str(uuid.uuid4())

    # Resolve waktu: dari ESP32 atau dari server
    hour, minute, raw_day = _resolve_time(data.hour, data.minute, data.day)
    current_time_str      = f"{hour:02d}:{minute:02d}"
    current_total_minutes = hour * 60 + minute

    # Konversi day ESP32 (Sun=0) → nama hari
    # Jika waktu dari server: weekday() Mon=0…Sun=6, konversi ke format DAY_MAP
    if data.day is not None:
        current_day_str = DAY_MAP.get(raw_day, "minggu")
    else:
        # weekday() dari server: 0=Mon … 6=Sun → mapping ke HARI_MAP keys
        _server_day_names = ["senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
        current_day_str = _server_day_names[raw_day]

    final_action = None

    # ── LOGIKA JADWAL (mode manual) ──────────────────────────────────────────
    if state["mode"] == "manual":
        schedule_triggered = False

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM schedules WHERE enabled = 1")
                all_schedules = cur.fetchall()

                # A. Cek jadwal yang harus MENYALA
                for sch in all_schedules:
                    sch_days = (
                        json.loads(sch["days"])
                        if isinstance(sch["days"], str)
                        else sch["days"]
                    )
                    if (
                        current_day_str in sch_days
                        and sch["time"] == current_time_str
                        and sch.get("last_triggered_minute") != current_total_minutes
                    ):
                        schedule_triggered = True
                        cur.execute(
                            "UPDATE schedules SET last_triggered_minute = %s, "
                            "last_triggered = CURRENT_TIMESTAMP WHERE id = %s",
                            (current_total_minutes, sch["id"]),
                        )
                        _update_state(
                            pump_status=True,
                            pump_start_minute=current_total_minutes,
                        )
                        final_action = "on"
                        log.info("JADWAL MANUAL: '%s' pompa ON.", sch["name"])
                        break

                # B. Cek apakah jadwal harus MATI (durasi habis)
                if (
                    not schedule_triggered
                    and state["pump_status"]
                    and state.get("pump_start_minute") is not None
                ):
                    elapsed = get_elapsed_minutes(
                        current_total_minutes, state["pump_start_minute"]
                    )
                    triggering_sch = next(
                        (
                            s for s in all_schedules
                            if s.get("last_triggered_minute") == state["pump_start_minute"]
                        ),
                        None,
                    )
                    if triggering_sch:
                        duration = triggering_sch.get("duration_minutes", 5)
                        if elapsed >= duration:
                            _update_state(
                                pump_status=False,
                                pump_start_minute=None,
                                last_watered_minute=current_total_minutes,
                            )
                            final_action = "off"
                            log.info("JADWAL SELESAI: durasi %s menit tercapai.", duration)

    # ── LOGIKA CERDAS KNN (mode auto) ───────────────────────────────────────
    elif state["mode"] == "auto":
        smart_eval   = _evaluate_smart_watering(
            result, hour, minute,
            data.air_humidity, data.temperature,
            state, current_total_minutes,
        )
        final_action = smart_eval["action"]

    # ── Simpan riwayat sensor ────────────────────────────────────────────────
    pump_status_logged = (
        final_action == "on"
        if final_action is not None
        else state["pump_status"]
    )

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sensor_readings
                    (id, timestamp, soil_moisture, temperature, air_humidity,
                     label, confidence, needs_watering, description, probabilities,
                     pump_status, mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row_id, timestamp,
                    data.soil_moisture, data.temperature, data.air_humidity,
                    result["label"], result["confidence"], result["needs_watering"],
                    result["description"],
                    json.dumps(result["probabilities"]),   # FIX: simpan data KNN asli
                    pump_status_logged, state["mode"],
                ),
            )

    _update_state(last_label=result["label"], last_updated=timestamp)
    new_state = _get_state()

    return {
        "received"      : True,
        "timestamp"     : timestamp,
        "device_time"   : current_time_str,
        "sensor"        : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
        "classification": result,
        "pump_status"   : new_state["pump_status"],
        "pump_action"   : final_action,
        "mode"          : new_state["mode"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT STATUS & HISTORY
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/status")
def get_status():
    state = _get_state()

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1"
            )
            latest = cur.fetchone()

    if latest:
        if isinstance(latest.get("probabilities"), str):
            latest["probabilities"] = json.loads(latest["probabilities"])
        latest["pump_status"]    = bool(latest["pump_status"])
        latest["needs_watering"] = bool(latest["needs_watering"])

    return {
        "pump_status" : state["pump_status"],
        "mode"        : state["mode"],
        "last_label"  : state["last_label"],
        "last_updated": str(state["last_updated"]) if state["last_updated"] else None,
        "latest_data" : latest,
    }


@app.get("/history")
def get_history(limit: int = Query(default=50, ge=1, le=500)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()

    records = []
    for r in reversed(rows):
        if isinstance(r.get("probabilities"), str):
            r["probabilities"] = json.loads(r["probabilities"])
        r["pump_status"]    = bool(r["pump_status"])
        r["needs_watering"] = bool(r["needs_watering"])
        records.append(r)

    return {"total": len(records), "records": records}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT KONTROL POMPA (dari Flutter)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/control")
def control_pump(cmd: ControlCommand):
    action = (cmd.action or "").lower()
    if action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'")

    mode = (cmd.mode or "manual").lower()
    # FIX: paksa 'schedule' dari Flutter menjadi 'manual' (tidak ada mode schedule di DB)
    if mode not in ("auto", "manual"):
        mode = "manual"

    pump_on = action == "on"
    update_kwargs = {"pump_status": pump_on, "mode": mode}
    if not pump_on:
        update_kwargs["pump_start_minute"] = None

    _update_state(**update_kwargs)
    state = _get_state()

    return {
        "success"    : True,
        "pump_status": state["pump_status"],
        "mode"       : state["mode"],
        "timestamp"  : datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT JADWAL (CRUD)
# ══════════════════════════════════════════════════════════════════════════════
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
        # Normalisasi last_triggered agar bisa di-serialize JSON
        if r.get("last_triggered") and not isinstance(r["last_triggered"], str):
            r["last_triggered"] = str(r["last_triggered"])
    return {"schedules": rows}


@app.post("/schedules")
def create_schedule(s: ScheduleCreate):
    new_id = str(uuid.uuid4())
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO schedules (id, name, time, duration_minutes, days, enabled)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (new_id, s.name, s.time, s.duration_minutes,
                 json.dumps(s.days), s.enabled),
            )
            cur.execute("SELECT * FROM schedules WHERE id = %s", (new_id,))
            row = cur.fetchone()

    if isinstance(row.get("days"), str):
        row["days"] = json.loads(row["days"])
    row["enabled"] = bool(row["enabled"])
    if row.get("last_triggered") and not isinstance(row["last_triggered"], str):
        row["last_triggered"] = str(row["last_triggered"])
    return {"success": True, "schedule": row}


@app.put("/schedules/{schedule_id}")
def update_schedule(schedule_id: str, s: ScheduleUpdate):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedules WHERE id = %s", (schedule_id,))
            ext = cur.fetchone()
            if not ext:
                raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")

            # Ambil nilai lama jika field tidak diupdate
            new_name     = s.name             if s.name             is not None else ext["name"]
            new_time     = s.time             if s.time             is not None else ext["time"]
            new_duration = s.duration_minutes if s.duration_minutes is not None else ext["duration_minutes"]
            new_enabled  = s.enabled          if s.enabled          is not None else ext["enabled"]
            new_days     = (
                json.dumps(s.days)
                if s.days is not None
                else (ext["days"] if isinstance(ext["days"], str) else json.dumps(ext["days"]))
            )

            cur.execute(
                """
                UPDATE schedules
                SET name=%s, time=%s, duration_minutes=%s, days=%s, enabled=%s
                WHERE id=%s
                """,
                (new_name, new_time, new_duration, new_days, new_enabled, schedule_id),
            )
            cur.execute("SELECT * FROM schedules WHERE id = %s", (schedule_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")
    if isinstance(row.get("days"), str):
        row["days"] = json.loads(row["days"])
    row["enabled"] = bool(row["enabled"])
    if row.get("last_triggered") and not isinstance(row["last_triggered"], str):
        row["last_triggered"] = str(row["last_triggered"])
    return {"success": True, "schedule": row}


@app.delete("/schedules/{schedule_id}")
def delete_schedule(schedule_id: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM schedules WHERE id = %s", (schedule_id,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Jadwal tidak ditemukan")
    return {"success": True}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT PREDICT (uji manual KNN)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/predict")
def predict(data: SensorData):
    return {
        "input" : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
        "result": classify(data.soil_moisture, data.temperature, data.air_humidity),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND TASK: cek jadwal tiap menit (dari v1)
# ══════════════════════════════════════════════════════════════════════════════
async def _schedule_checker():
    while True:
        await asyncio.sleep(60)
        try:
            await _run_due_schedules()
        except Exception as e:
            log.error("[JADWAL ERR] %s", e)


async def _run_due_schedules():
    now  = datetime.now()
    h_m  = now.strftime("%H:%M")
    # weekday(): Mon=0 … Sun=6 → map ke format HARI_MAP
    _server_day_names = ["senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
    hari_str = _server_day_names[now.weekday()]

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schedules WHERE enabled = 1")
            schedules = cur.fetchall()

    for sc in schedules:
        if sc["time"] != h_m:
            continue

        days = json.loads(sc["days"]) if isinstance(sc["days"], str) else sc["days"]
        if hari_str not in days:
            continue

        last = sc.get("last_triggered")
        if last and str(last)[:16] == now.isoformat()[:16]:
            continue

        # Cek mode — hanya jalankan background scheduler jika mode auto
        state = _get_state()
        if state["mode"] != "auto":
            continue

        _update_state(pump_status=True, mode="auto")

        log_id = str(uuid.uuid4())
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE schedules SET last_triggered = %s WHERE id = %s",
                    (now.strftime("%Y-%m-%d %H:%M:%S"), sc["id"]),
                )
                cur.execute(
                    """
                    INSERT INTO schedule_logs
                        (id, schedule_id, triggered_at, duration_minutes)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (log_id, sc["id"],
                     now.strftime("%Y-%m-%d %H:%M:%S"), sc["duration_minutes"]),
                )

        log.info("[JADWAL] '%s' — pompa ON selama %s menit", sc["name"], sc["duration_minutes"])
        asyncio.create_task(_stop_pump_after(sc["duration_minutes"], log_id, sc["name"]))


async def _stop_pump_after(minutes: int, log_id: str, name: str):
    await asyncio.sleep(minutes * 60)
    _update_state(pump_status=False, mode="auto")

    completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE schedule_logs SET completed_at = %s WHERE id = %s",
                (completed, log_id),
            )

    log.info("[JADWAL] '%s' selesai — pompa OFF, mode kembali ke auto", name)