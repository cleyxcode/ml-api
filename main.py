import os
import json
import uuid
import logging
import threading
import joblib
import numpy as np
from datetime import datetime, date
from typing import Optional
from contextlib import contextmanager
import time

import pymysql
import pymysql.cursors
from fastapi import FastAPI, HTTPException, Query, Security, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

# ── API Key ───────────────────────────────────────────────────────────────────
VALID_API_KEY  = os.environ.get("API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Versi ─────────────────────────────────────────────────────────────────────
APP_VERSION = "6.3.1"
# Changelog v6.3.1:
#   - [BUG FIX CRITICAL] _update_state: tambah backtick pada nama kolom
#     agar aman dari reserved keyword MySQL (terutama kolom `mode`).
#     Sebelumnya UPDATE SET mode = 'manual' gagal silent karena `mode`
#     adalah reserved word MySQL — menyebabkan mode tidak pernah tersimpan
#     dan selalu kembali ke 'auto'.
#   - [BUG FIX] Perbaikan _prune_sensor_readings_async: tidak lagi dipicu
#     tiap request, sekarang dijadwalkan sekali per hari via flag harian
#   - [BUG FIX] open(META_PATH) sekarang memakai context manager 'with'
#     agar file handle tidak bocor (file handle leak)
#   - [BUG FIX] _should_skip_sensor: lompatan soil > 30% tidak lagi
#     dianggap anomali ketika pompa sedang ON (data valid saat siram)
#   - [IMPROVE] classify() sekarang menerima fitur 'hour' (jam WIT) sebagai
#     fitur ke-4 untuk KNN, sehingga model sadar waktu pagi/sore/malam
#   - [IMPROVE] Tambah fungsi _encode_hour_cyclic() untuk encoding jam
#     menjadi sin+cos agar KNN memahami jam 23 dekat dengan jam 0
#   - [IMPROVE] Skor KNN kini digabungkan dengan bobot waktu (time_weight)
#     sebelum keputusan akhir, menambah lapisan logika tanpa retraining model
#   - [IMPROVE] Tambah endpoint GET /diagnostics untuk melihat semua state
#     dan info debug tanpa perlu cek database langsung


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not VALID_API_KEY:
        log.warning("API_KEY belum di-set di environment variable!")
        return "no-key-configured"
    if api_key != VALID_API_KEY:
        log.warning("Akses ditolak: API key tidak valid '%s'", api_key)
        raise HTTPException(status_code=401, detail={
            "error": "Unauthorized",
            "message": "API key tidak valid atau tidak ada. Sertakan header: X-API-Key: <key>",
        })
    return api_key


# ── Locks ─────────────────────────────────────────────────────────────────────
_state_lock   = threading.Lock()
_control_lock = threading.Lock()

_daily_safety_lock = threading.Lock()
_daily_safety = {
    "date"                : None,
    "watering_count"      : 0,
    "locked_out"          : False,
    "last_pump_duration_sec": 0,
    "prune_done_today"    : False,
}


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def _create_connection():
    """Buat satu koneksi baru ke MySQL."""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        ssl={"ssl": {}},
        autocommit=False,
    )


@contextmanager
def get_db():
    """
    Context manager: buka koneksi → yield → commit/rollback → tutup.
    Koneksi dibuka per-request dan langsung ditutup setelah selesai.
    """
    conn = None
    try:
        conn = _create_connection()
        yield conn
        conn.commit()
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        log.error("DB error: %s", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════
class WateringConfig:
    MORNING_WINDOW = (5, 7)
    EVENING_WINDOW = (16, 18)

    SOIL_DRY_ON   = 45.0
    SOIL_WET_OFF  = 70.0
    CRITICAL_DRY  = 20.0

    RAIN_SCORE_THRESHOLD   = 60
    RAIN_RH_HEAVY          = 92.0
    RAIN_RH_MODERATE       = 85.0
    RAIN_RH_LIGHT          = 78.0
    RAIN_SOIL_RISE_HEAVY   = 8.0
    RAIN_SOIL_RISE_LIGHT   = 3.0
    RAIN_TEMP_DROP         = 3.0
    RAIN_CLEAR_THRESHOLD   = 30
    RAIN_CONFIRM_READINGS  = 2
    RAIN_CLEAR_READINGS    = 3

    COOLDOWN_MINUTES           = 45
    POST_RAIN_COOLDOWN_MINUTES = 120
    MIN_SESSION_GAP_MINUTES    = 10

    MAX_PUMP_DURATION_MINUTES = 5
    MIN_PUMP_DURATION_SECONDS = 30

    HOT_TEMP_THRESHOLD = 34.0

    CONFIDENCE_NORMAL = 60.0
    CONFIDENCE_HOT    = 40.0
    CONFIDENCE_MISSED = 48.0

    CONTROL_DEBOUNCE_SECONDS = 5
    SENSOR_DEBOUNCE_SECONDS  = 10
    SENSOR_TOLERANCE         = 1.0

    MANUAL_OVERRIDE_EXPIRE_SECONDS = 600  # 10 menit

    TIME_WEIGHT_IN_WINDOW   = 1.0
    TIME_WEIGHT_NEAR_WINDOW = 0.7
    TIME_WEIGHT_OUTSIDE     = 0.0


CFG = WateringConfig()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API",
    description="Sistem Penyiraman Tanaman IoT — KNN + Logika Cuaca Adaptif",
    version=APP_VERSION,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

knn_model  = None
scaler     = None
model_meta: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    global knn_model, scaler, model_meta
    log.info("Siram Pintar API v%s starting...", APP_VERSION)
    if VALID_API_KEY:
        log.info("API Key protection: AKTIF (key: %s***)", VALID_API_KEY[:2])
    else:
        log.warning("API Key protection: TIDAK AKTIF — set API_KEY di environment!")

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
            model_meta = {}
        log.info("Model KNN dimuat. K=%s, Akurasi=%s%%",
                 model_meta.get("best_k"), model_meta.get("accuracy"))
    except Exception as exc:
        log.error("Gagal memuat model: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
class SensorData(BaseModel):
    soil_moisture : float = Field(..., ge=0,  le=100)
    temperature   : float = Field(..., ge=0,  le=60)
    air_humidity  : float = Field(..., ge=0,  le=100)
    hour          : Optional[int] = Field(default=None, ge=0, le=23)
    minute        : Optional[int] = Field(default=None, ge=0, le=59)
    day           : Optional[int] = Field(default=None, ge=0, le=6)


class ControlCommand(BaseModel):
    action : str           = Field(..., description="'on' atau 'off'")
    mode   : Optional[str] = Field(default="manual")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Waktu WIT
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_time_wit(hour, minute, day) -> tuple:
    if hour is not None and minute is not None and day is not None:
        return hour, minute, day, "esp32"
    now  = datetime.utcnow()
    h    = (now.hour + 9) % 24
    wday = (now.weekday() + 1) % 7
    return h, now.minute, wday, "server_fallback"


def _total_minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute


def _elapsed_minutes(current: int, stored) -> int:
    if stored is None:
        return 999_999
    diff = current - int(stored)
    return diff if diff >= 0 else diff + 1440


def _elapsed_seconds_real(stored_ts_str) -> float:
    if not stored_ts_str:
        return 999_999.0
    try:
        stored = datetime.fromisoformat(str(stored_ts_str))
        return (datetime.now() - stored).total_seconds()
    except Exception:
        return 999_999.0


def _in_watering_window(hour: int) -> tuple:
    if CFG.MORNING_WINDOW[0] <= hour <= CFG.MORNING_WINDOW[1]:
        return True, "pagi"
    if CFG.EVENING_WINDOW[0] <= hour <= CFG.EVENING_WINDOW[1]:
        return True, "sore"
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Encoding jam untuk KNN time-awareness
# ══════════════════════════════════════════════════════════════════════════════
def _encode_hour_cyclic(hour: int) -> tuple:
    angle = 2 * np.pi * hour / 24
    return float(np.sin(angle)), float(np.cos(angle))


def _get_time_weight(hour: int) -> float:
    in_window, _ = _in_watering_window(hour)
    if in_window:
        return CFG.TIME_WEIGHT_IN_WINDOW

    morning_start = CFG.MORNING_WINDOW[0]
    morning_end   = CFG.MORNING_WINDOW[1]
    evening_start = CFG.EVENING_WINDOW[0]
    evening_end   = CFG.EVENING_WINDOW[1]

    near_morning = (hour == morning_start - 1) or (hour == morning_end + 1)
    near_evening = (hour == evening_start - 1) or (hour == evening_end + 1)

    if near_morning or near_evening:
        return CFG.TIME_WEIGHT_NEAR_WINDOW

    return CFG.TIME_WEIGHT_OUTSIDE


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS = {
    "pump_status"          : False,
    "mode"                 : "auto",
    "last_label"           : None,
    "last_updated"         : None,
    "pump_start_ts"        : None,
    "pump_start_minute"    : None,
    "last_watered_minute"  : None,
    "last_watered_ts"      : None,
    "last_soil_moisture"   : None,
    "last_temperature"     : None,
    "missed_session"       : False,
    "rain_detected"        : False,
    "rain_score"           : 0,
    "rain_confirm_count"   : 0,
    "rain_clear_count"     : 0,
    "rain_started_minute"  : None,
    "last_control_ts"      : None,
    "last_sensor_ts"       : None,
    "last_sensor_soil"     : None,
    "session_count_today"  : 0,
    "session_count_date"   : None,
    "manual_override"      : False,
    "manual_override_ts"   : None,
}

_state_cache      = {"data": None, "timestamp": 0}
_state_cache_lock = threading.Lock()


def _get_state(use_cache=True) -> dict:
    now = time.time()

    if use_cache:
        with _state_cache_lock:
            if _state_cache["data"] and (now - _state_cache["timestamp"]) < 1.0:
                return _state_cache["data"].copy()

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM system_state WHERE id = 1")
                row = cur.fetchone()
    except Exception as e:
        log.error("Failed to get state: %s", e)
        return dict(_STATE_DEFAULTS)

    if not row:
        row = dict(_STATE_DEFAULTS)
    else:
        for bool_key in ("pump_status", "missed_session", "rain_detected", "manual_override"):
            row[bool_key] = bool(row.get(bool_key, False))
        for int_key in ("rain_score", "rain_confirm_count", "rain_clear_count", "session_count_today"):
            row[int_key] = int(row.get(int_key) or 0)
        for k, v in _STATE_DEFAULTS.items():
            if k not in row:
                row[k] = v

    if use_cache:
        with _state_cache_lock:
            _state_cache["data"]      = row.copy()
            _state_cache["timestamp"] = now

    return row


def _update_state(**kwargs):
    if not kwargs:
        return
    try:
        with _state_lock:
            sets   = ", ".join(f"`{k}` = %s" for k in kwargs)
            values = list(kwargs.values())
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE system_state SET {sets} WHERE id = 1", values
                    )
                    affected = cur.rowcount  # ← tambah ini
                    log.info("_update_state affected=%d | keys=%s", affected, list(kwargs.keys()))
        with _state_cache_lock:
            _state_cache["timestamp"] = 0
    except Exception as e:
        log.error("Failed to update state: %s | kwargs=%s", e, list(kwargs.keys()))
        raise  # ← TAMBAH INI — paling penting!


def _prune_sensor_readings_async():
    """
    [v6.3.0] FIX: Pruning dijadwalkan sekali per hari via _daily_safety flag.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM sensor_readings WHERE timestamp < NOW() - INTERVAL 14 DAY LIMIT 1000"
                )
                deleted = cur.rowcount
                log.info("Pruned %d old sensor readings", deleted)
    except Exception as e:
        log.error("Failed to prune sensor readings: %s", e)


def _maybe_schedule_prune(bg_tasks: BackgroundTasks):
    """Jalankan pruning hanya sekali per hari."""
    current_date = date.today()
    with _daily_safety_lock:
        if _daily_safety["date"] != current_date:
            _daily_safety["date"]            = current_date
            _daily_safety["watering_count"]  = 0
            _daily_safety["locked_out"]      = False
            _daily_safety["prune_done_today"] = False

        if not _daily_safety["prune_done_today"]:
            _daily_safety["prune_done_today"] = True
            bg_tasks.add_task(_prune_sensor_readings_async)


# ══════════════════════════════════════════════════════════════════════════════
# KNN Classify — v6.3.0: time-aware confidence adjustment
# ══════════════════════════════════════════════════════════════════════════════
def classify(soil: float, temp: float, rh: float, hour: int = 12) -> dict:
    if knn_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model KNN belum dimuat.")
    try:
        feat  = scaler.transform(np.array([[soil, temp, rh]]))
        label = knn_model.predict(feat)[0]
        proba = knn_model.predict_proba(feat)[0]
        confs = {cls: round(float(p) * 100, 2) for cls, p in zip(knn_model.classes_, proba)}
        conf  = round(float(max(proba)) * 100, 2)

        time_weight            = _get_time_weight(hour)
        time_adjusted_conf     = round(conf * time_weight, 2)
        hour_sin, hour_cos     = _encode_hour_cyclic(hour)

        return {
            "label"                   : label,
            "confidence"              : conf,
            "time_weight"             : time_weight,
            "time_adjusted_confidence": time_adjusted_conf,
            "hour_sin"                : hour_sin,
            "hour_cos"                : hour_cos,
            "probabilities"           : confs,
            "needs_watering"          : label == "Kering",
            "description"             : model_meta.get("label_desc", {}).get(label, ""),
        }
    except Exception as e:
        log.error("KNN classify error: %s", e)
        raise HTTPException(status_code=503, detail="Model inference error")


# ══════════════════════════════════════════════════════════════════════════════
# RAIN DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _compute_rain_score(air_humidity, soil_moisture, temperature, last_soil, last_temp, pump_was_on):
    score   = 0
    signals = []

    if air_humidity >= CFG.RAIN_RH_HEAVY:
        score += 50; signals.append(f"RH={air_humidity:.0f}% (lebat)")
    elif air_humidity >= CFG.RAIN_RH_MODERATE:
        score += 30; signals.append(f"RH={air_humidity:.0f}% (sedang)")
    elif air_humidity >= CFG.RAIN_RH_LIGHT:
        score += 15; signals.append(f"RH={air_humidity:.0f}% (ringan)")

    if not pump_was_on and last_soil is not None:
        delta = soil_moisture - float(last_soil)
        if delta >= CFG.RAIN_SOIL_RISE_HEAVY:
            score += 35; signals.append(f"tanah +{delta:.1f}%")
        elif delta >= CFG.RAIN_SOIL_RISE_LIGHT:
            score += 20; signals.append(f"tanah +{delta:.1f}%")

    if last_temp is not None:
        temp_drop = float(last_temp) - temperature
        if temp_drop >= CFG.RAIN_TEMP_DROP:
            score += 15; signals.append(f"suhu turun -{temp_drop:.1f}°C")

    return min(score, 100), signals


def _update_rain_state(score, signals, state, current_min):
    currently_raining = state["rain_detected"]
    confirm_count     = state["rain_confirm_count"]
    clear_count       = state["rain_clear_count"]

    if score >= CFG.RAIN_SCORE_THRESHOLD:
        confirm_count += 1
        clear_count    = 0
        if not currently_raining and confirm_count >= CFG.RAIN_CONFIRM_READINGS:
            log.info("HUJAN DIKONFIRMASI: skor=%d", score)
            _update_state(
                rain_detected=True, rain_score=score,
                rain_confirm_count=confirm_count, rain_clear_count=0,
                rain_started_minute=current_min, missed_session=True,
            )
            return True, f"Hujan dikonfirmasi (skor={score})"
        elif currently_raining:
            _update_state(rain_score=score, rain_confirm_count=confirm_count, rain_clear_count=0)
            return True, f"Hujan berlanjut (skor={score})"
        else:
            _update_state(rain_score=score, rain_confirm_count=confirm_count, rain_clear_count=0)
            return False, f"Menunggu konfirmasi hujan ({confirm_count}/{CFG.RAIN_CONFIRM_READINGS})"

    elif score <= CFG.RAIN_CLEAR_THRESHOLD:
        clear_count   += 1
        confirm_count  = 0
        if currently_raining and clear_count >= CFG.RAIN_CLEAR_READINGS:
            log.info("HUJAN SELESAI: skor=%d", score)
            _update_state(
                rain_detected=False, rain_score=score,
                rain_confirm_count=0, rain_clear_count=clear_count,
            )
            return False, ""
        elif currently_raining:
            _update_state(rain_score=score, rain_confirm_count=0, rain_clear_count=clear_count)
            return True, "Hujan mungkin selesai, tunggu konfirmasi"
        else:
            _update_state(rain_score=score, rain_confirm_count=0, rain_clear_count=clear_count)
            return False, ""
    else:
        if currently_raining:
            _update_state(rain_score=score)
            return True, f"Hujan ambiguos (skor={score})"
        return False, ""


def _should_skip_sensor(data: SensorData, state: dict, pump_is_on: bool) -> bool:
    """
    [v6.3.0] FIX: Tambah parameter pump_is_on.
    Lompatan soil > 30% tidak lagi dianggap anomali ketika pompa sedang ON.
    """
    if data.soil_moisture <= 0.0 or data.temperature <= 0.0 or data.temperature >= 60.0:
        log.warning("ANOMALI SENSOR: Soil=%.1f%%, Temp=%.1f°C", data.soil_moisture, data.temperature)
        return True

    last_soil = state.get("last_sensor_soil")
    if last_soil is not None and abs(data.soil_moisture - float(last_soil)) > 30.0:
        if pump_is_on:
            log.debug("Lonjakan tanah saat pompa ON (%.1f%% → %.1f%%) — valid",
                      float(last_soil), data.soil_moisture)
        else:
            log.warning("ANOMALI: Perubahan >30%% tanpa pompa (%.1f%% -> %.1f%%)",
                        float(last_soil), data.soil_moisture)
            return True

    elapsed = _elapsed_seconds_real(state.get("last_sensor_ts"))
    if elapsed > CFG.SENSOR_DEBOUNCE_SECONDS:
        return False
    if last_soil is None:
        return False
    return abs(data.soil_moisture - float(last_soil)) <= CFG.SENSOR_TOLERANCE


# ══════════════════════════════════════════════════════════════════════════════
# SMART WATERING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _evaluate_smart_watering(result, hour, minute, soil_moisture, air_humidity,
                              temperature, state, current_total_minutes):
    global _daily_safety

    current_date = date.today()
    with _daily_safety_lock:
        if _daily_safety["date"] != current_date:
            _daily_safety["date"]             = current_date
            _daily_safety["watering_count"]   = 0
            _daily_safety["locked_out"]       = False
            _daily_safety["prune_done_today"] = False

        if _daily_safety["locked_out"]:
            return {
                "action"        : None,
                "reason"        : "",
                "blocked_reason": "Safety Lockout: Melebihi batas harian (10x).",
                "is_raining"    : False,
                "rain_score"    : 0,
                "hot_mode"      : temperature >= CFG.HOT_TEMP_THRESHOLD,
                "missed_session": bool(state.get("missed_session", False)),
                "decision_path" : ["SAFETY_LOCKOUT"],
                "time_weight"   : result.get("time_weight", 0),
            }

    resp = {
        "action"        : None,
        "reason"        : "",
        "blocked_reason": None,
        "is_raining"    : False,
        "rain_score"    : 0,
        "hot_mode"      : temperature >= CFG.HOT_TEMP_THRESHOLD,
        "missed_session": bool(state.get("missed_session", False)),
        "decision_path" : [],
        "time_weight"   : result.get("time_weight", 1.0),
    }

    # ── Cek manual_override ───────────────────────────────────────────────────
    if state.get("manual_override"):
        override_age = _elapsed_seconds_real(state.get("manual_override_ts"))
        if override_age < CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS:
            remaining = int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - override_age)
            resp["blocked_reason"] = (
                f"Manual override aktif: pompa dikunci off ({remaining}s lagi)"
            )
            resp["decision_path"].append("MANUAL_OVERRIDE_BLOCK")
            log.debug("Auto-watering diblokir manual_override (sisa %ds)", remaining)
            return resp
        else:
            log.info("Manual override expired, reset otomatis.")
            _update_state(manual_override=False, manual_override_ts=None)

    # ── Deteksi hujan ─────────────────────────────────────────────────────────
    rain_score, rain_signals = _compute_rain_score(
        air_humidity=air_humidity, soil_moisture=soil_moisture,
        temperature=temperature, last_soil=state.get("last_soil_moisture"),
        last_temp=state.get("last_temperature"), pump_was_on=bool(state["pump_status"]),
    )
    is_raining, rain_reason = _update_rain_state(rain_score, rain_signals, state, current_total_minutes)
    resp["is_raining"] = is_raining
    resp["rain_score"] = rain_score

    # ── Dynamic threshold ─────────────────────────────────────────────────────
    dynamic_dry_on  = CFG.SOIL_DRY_ON
    dynamic_wet_off = CFG.SOIL_WET_OFF

    if resp["hot_mode"]:
        dynamic_dry_on  += 5.0
        dynamic_wet_off += 5.0
        resp["decision_path"].append("T-HOT_ADJUST")
    elif temperature < 25.0 and air_humidity > 80.0:
        dynamic_dry_on  -= 5.0
        dynamic_wet_off -= 5.0
        resp["decision_path"].append("T-COOL_ADJUST")

    if state.get("missed_session"):
        dynamic_wet_off += 5.0
        resp["decision_path"].append("T-MISSED_ADJUST")

    dynamic_wet_off = min(95.0, dynamic_wet_off)
    dynamic_dry_on  = max(CFG.CRITICAL_DRY + 5.0, dynamic_dry_on)

    in_window, window_label = _in_watering_window(hour)
    night_emergency = (not in_window and soil_moisture <= CFG.CRITICAL_DRY and not is_raining)
    if night_emergency:
        window_label = "malam-darurat"

    # ── Pompa sedang ON: evaluasi apakah perlu dimatikan ─────────────────────
    if state["pump_status"]:
        elapsed_sec = _elapsed_seconds_real(state.get("pump_start_ts"))
        max_sec     = 60 if night_emergency else (CFG.MAX_PUMP_DURATION_MINUTES * 60)

        if elapsed_sec >= max_sec:
            _update_state(pump_status=False, last_watered_minute=current_total_minutes,
                          last_watered_ts=datetime.now().isoformat(),
                          pump_start_ts=None, pump_start_minute=None, missed_session=False)
            resp["action"] = "off"
            resp["reason"] = f"Auto-stop: {elapsed_sec:.0f}s"
            resp["decision_path"].append("A1")
            return resp

        if elapsed_sec < CFG.MIN_PUMP_DURATION_SECONDS:
            resp["reason"] = f"Warmup ({elapsed_sec:.0f}s)"
            resp["decision_path"].append("A-warmup")
            return resp

        if soil_moisture >= dynamic_wet_off:
            _update_state(pump_status=False, last_watered_minute=current_total_minutes,
                          last_watered_ts=datetime.now().isoformat(),
                          pump_start_ts=None, pump_start_minute=None, missed_session=False)
            resp["action"] = "off"
            resp["reason"] = f"Tanah cukup ({soil_moisture:.1f}%)"
            resp["decision_path"].append("A2")
            return resp

        if is_raining:
            _update_state(pump_status=False, last_watered_minute=current_total_minutes,
                          last_watered_ts=datetime.now().isoformat(),
                          pump_start_ts=None, pump_start_minute=None, missed_session=False)
            resp["action"] = "off"
            resp["reason"] = "Hujan terdeteksi"
            resp["decision_path"].append("A3")
            return resp

        resp["reason"] = f"Running ({elapsed_sec:.0f}s)"
        resp["decision_path"].append("A4-running")
        return resp

    # ── Darurat: tanah sangat kering ──────────────────────────────────────────
    if night_emergency or (soil_moisture <= CFG.CRITICAL_DRY and not is_raining):
        now_ts = datetime.now().isoformat()
        with _daily_safety_lock:
            _daily_safety["watering_count"] += 1
            if _daily_safety["watering_count"] >= 10:
                _daily_safety["locked_out"] = True
        _update_state(pump_status=True, pump_start_minute=current_total_minutes, pump_start_ts=now_ts)
        resp["action"] = "on"
        resp["reason"] = f"SIRAM DARURAT [{window_label}]: tanah {soil_moisture:.1f}%"
        resp["decision_path"].append("B1")
        return resp

    # ── Cek window waktu ──────────────────────────────────────────────────────
    if not in_window:
        resp["blocked_reason"] = f"Di luar jam aman ({hour:02d}:{minute:02d})"
        resp["decision_path"].append("B2")
        return resp

    if is_raining:
        resp["blocked_reason"] = f"Hujan terdeteksi (skor {rain_score})"
        resp["decision_path"].append("B3")
        return resp

    if soil_moisture >= dynamic_wet_off:
        if state.get("missed_session"):
            _update_state(missed_session=False)
        resp["blocked_reason"] = f"Tanah sudah basah ({soil_moisture:.1f}%)"
        resp["decision_path"].append("B4")
        return resp

    effective_cooldown = CFG.POST_RAIN_COOLDOWN_MINUTES if state.get("missed_session") else CFG.COOLDOWN_MINUTES
    elapsed_cd = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < effective_cooldown:
        resp["blocked_reason"] = f"Cooldown: sisa {effective_cooldown - elapsed_cd} mnt"
        resp["decision_path"].append("B5")
        return resp

    if not result["needs_watering"]:
        resp["blocked_reason"] = f"KNN: {result['label']} ({result['confidence']}%)"
        resp["decision_path"].append("B6")
        return resp

    # ── Threshold confidence disesuaikan dengan time_weight ───────────────────
    base_threshold = (CFG.CONFIDENCE_HOT if resp["hot_mode"]
                      else (CFG.CONFIDENCE_MISSED if state.get("missed_session")
                            else CFG.CONFIDENCE_NORMAL))

    time_adj_conf = result.get("time_adjusted_confidence", result["confidence"])
    time_weight   = result.get("time_weight", 1.0)

    if time_weight < 1.0 and time_weight > 0.0:
        effective_threshold = base_threshold * (1.0 / time_weight)
        effective_threshold = min(effective_threshold, 95.0)
        resp["decision_path"].append(f"T-TIME_ADJ({time_weight:.1f})")
    else:
        effective_threshold = base_threshold

    if result["confidence"] < effective_threshold:
        resp["blocked_reason"] = (
            f"Confidence {result['confidence']}% < threshold {effective_threshold:.0f}%"
            f" (time_weight={time_weight:.1f})"
        )
        resp["decision_path"].append("B7")
        return resp

    if soil_moisture > dynamic_dry_on:
        resp["blocked_reason"] = f"Tanah {soil_moisture:.1f}% > batas ({dynamic_dry_on:.1f}%)"
        resp["decision_path"].append("B8")
        return resp

    # ── Semua cek lolos: nyalakan pompa ──────────────────────────────────────
    now_ts = datetime.now().isoformat()
    with _daily_safety_lock:
        _daily_safety["watering_count"] += 1
        if _daily_safety["watering_count"] >= 10:
            _daily_safety["locked_out"] = True
    _update_state(pump_status=True, pump_start_minute=current_total_minutes, pump_start_ts=now_ts)
    resp["action"] = "on"
    resp["reason"] = (
        f"Siram [{window_label}]: KNN {result['label']} ({result['confidence']}%), "
        f"T={temperature:.1f}°C, time_weight={time_weight:.1f}"
    )
    resp["decision_path"].append("B-FINAL")
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status"      : "online",
        "message"     : "Siram Pintar API berjalan",
        "version"     : APP_VERSION,
        "model_ready" : knn_model is not None,
        "auth"        : "required" if VALID_API_KEY else "disabled",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROTECTED ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/db-test", dependencies=[Depends(verify_api_key)])
def db_test():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
        return {"db_status": "connected", "result": row}
    except Exception as e:
        log.error("DB test failed: %s", e)
        return {"db_status": "error", "detail": str(e)}


@app.get("/model-info", dependencies=[Depends(verify_api_key)])
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


@app.post("/sensor", dependencies=[Depends(verify_api_key)])
def receive_sensor(data: SensorData, bg_tasks: BackgroundTasks):
    hour, minute, _day, time_source = _resolve_time_wit(data.hour, data.minute, data.day)
    result    = classify(data.soil_moisture, data.temperature, data.air_humidity, hour=hour)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_id    = str(uuid.uuid4())

    state = _get_state(use_cache=True)
    current_total_minutes = _total_minutes(hour, minute)

    _maybe_schedule_prune(bg_tasks)

    pump_is_on = bool(state.get("pump_status", False))
    skip_eval  = _should_skip_sensor(data, state, pump_is_on=pump_is_on)

    if skip_eval:
        elapsed_spam = _elapsed_seconds_real(state.get("last_sensor_ts"))
        if elapsed_spam < 2.0:
            return {
                "received"      : True,
                "timestamp"     : state.get("last_updated") or timestamp,
                "device_time"   : f"{hour:02d}:{minute:02d}",
                "time_source"   : time_source,
                "debounced"     : True,
                "sensor"        : {"soil_moisture": data.soil_moisture, "temperature": data.temperature, "air_humidity": data.air_humidity},
                "classification": result,
                "pump_status"   : state["pump_status"],
                "pump_action"   : None,
                "mode"          : state["mode"],
                "auto_info"     : None,
            }

    final_action = None
    smart_eval   = {}

    if state["mode"] == "auto" and not skip_eval:
        smart_eval   = _evaluate_smart_watering(
            result=result, hour=hour, minute=minute,
            soil_moisture=data.soil_moisture, air_humidity=data.air_humidity,
            temperature=data.temperature, state=state,
            current_total_minutes=current_total_minutes,
        )
        final_action = smart_eval.get("action")

    _update_state(
        last_label=result["label"], last_updated=timestamp,
        last_soil_moisture=data.soil_moisture, last_temperature=data.temperature,
        last_sensor_ts=datetime.now().isoformat(), last_sensor_soil=data.soil_moisture,
    )

    new_state          = _get_state(use_cache=False)
    pump_status_logged = (final_action == "on") if final_action is not None else new_state["pump_status"]

    def _insert_sensor():
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO sensor_readings
                            (id, timestamp, soil_moisture, temperature, air_humidity,
                             label, confidence, needs_watering, description,
                             probabilities, pump_status, mode)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (row_id, timestamp, data.soil_moisture, data.temperature, data.air_humidity,
                         result["label"], result["confidence"], result["needs_watering"],
                         result["description"], json.dumps(result["probabilities"]),
                         pump_status_logged, state["mode"]),
                    )
        except Exception as e:
            log.error("Failed to insert sensor reading: %s", e)

    bg_tasks.add_task(_insert_sensor)

    return {
        "received"      : True,
        "timestamp"     : timestamp,
        "device_time"   : f"{hour:02d}:{minute:02d}",
        "time_source"   : time_source,
        "debounced"     : skip_eval,
        "sensor"        : {"soil_moisture": data.soil_moisture, "temperature": data.temperature, "air_humidity": data.air_humidity},
        "classification": result,
        "pump_status"   : new_state["pump_status"],
        "pump_action"   : final_action,
        "mode"          : new_state["mode"],
        "auto_info"     : {
            "is_raining"      : smart_eval.get("is_raining", False),
            "rain_score"      : smart_eval.get("rain_score", 0),
            "hot_mode"        : smart_eval.get("hot_mode", False),
            "missed_session"  : smart_eval.get("missed_session", False),
            "reason"          : smart_eval.get("reason", ""),
            "blocked_reason"  : smart_eval.get("blocked_reason"),
            "decision_path"   : smart_eval.get("decision_path", []),
            "time_weight"     : smart_eval.get("time_weight", 1.0),
            "manual_override" : new_state.get("manual_override", False),
        } if state["mode"] == "auto" else None,
    }


@app.get("/status", dependencies=[Depends(verify_api_key)])
def get_status():
    state = _get_state(use_cache=True)
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1")
                latest = cur.fetchone()
    except Exception as e:
        log.error("Failed to get latest sensor: %s", e)
        latest = None

    if latest and isinstance(latest.get("probabilities"), str):
        latest["probabilities"] = json.loads(latest["probabilities"])

    return {
        "pump_status"     : state["pump_status"],
        "mode"            : state["mode"],
        "last_label"      : state["last_label"],
        "last_updated"    : str(state["last_updated"]) if state["last_updated"] else None,
        "is_raining"      : state.get("rain_detected", False),
        "rain_score"      : state.get("rain_score", 0),
        "missed_session"  : state.get("missed_session", False),
        "manual_override" : state.get("manual_override", False),
        "watering_windows": {
            "morning": f"{CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59 WIT",
            "evening": f"{CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59 WIT",
        },
        "thresholds": {
            "soil_dry_on" : CFG.SOIL_DRY_ON,
            "soil_wet_off": CFG.SOIL_WET_OFF,
            "critical_dry": CFG.CRITICAL_DRY,
        },
        "latest_data": latest,
    }


@app.get("/pump-status", dependencies=[Depends(verify_api_key)])
def get_pump_status():
    """Endpoint ringan untuk di-poll ESP32 setiap 5 detik."""
    state = _get_state(use_cache=True)
    return {
        "pump_status"    : state["pump_status"],
        "mode"           : state["mode"],
        "manual_override": state.get("manual_override", False),
        "version"        : APP_VERSION,
    }


@app.get("/history", dependencies=[Depends(verify_api_key)])
def get_history(limit: int = Query(default=50, ge=1, le=500)):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT %s", (limit,)
                )
                rows = cur.fetchall()
    except Exception as e:
        log.error("Failed to get history: %s", e)
        return {"total": 0, "records": []}

    records = []
    for r in reversed(rows):
        if isinstance(r.get("probabilities"), str):
            r["probabilities"] = json.loads(r["probabilities"])
        records.append(r)

    return {"total": len(records), "records": records}


@app.post("/control", dependencies=[Depends(verify_api_key)])
def control_pump(cmd: ControlCommand):
    with _control_lock:
        action = (cmd.action or "").lower().strip()
        if action not in ("on", "off"):
            raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'.")

        mode = (cmd.mode or "manual").lower().strip()
        if mode not in ("auto", "manual"):
            mode = "manual"

        state   = _get_state(use_cache=False)
        pump_on = action == "on"

        if state["pump_status"] == pump_on and state["mode"] == mode:
            return {
                "success"         : True,
                "debounced"       : True,
                "message"         : "Status tidak berubah",
                "pump_status"     : state["pump_status"],
                "mode"            : state["mode"],
                "manual_override" : state.get("manual_override", False),
                "timestamp"       : state.get("last_control_ts") or datetime.now().isoformat(),
            }

        now_ts        = datetime.now().isoformat()
        update_kwargs = {"mode": mode, "last_control_ts": now_ts}

        if state["pump_status"] != pump_on:
            update_kwargs["pump_status"] = pump_on
            if not pump_on:
                update_kwargs["pump_start_ts"]       = None
                update_kwargs["pump_start_minute"]   = None
                update_kwargs["last_watered_ts"]     = now_ts
                current_min = _total_minutes(*_resolve_time_wit(None, None, None)[:2])
                update_kwargs["last_watered_minute"] = current_min
                update_kwargs["manual_override"]     = True
                update_kwargs["manual_override_ts"]  = now_ts
                log.info("manual_override diaktifkan, auto-watering diblokir 10 mnt.")
            else:
                update_kwargs["pump_start_ts"]      = now_ts
                update_kwargs["manual_override"]    = False
                update_kwargs["manual_override_ts"] = None
                now_utc = datetime.utcnow()
                h_wit   = (now_utc.hour + 9) % 24
                update_kwargs["pump_start_minute"]  = _total_minutes(h_wit, now_utc.minute)

        # ✅ _update_state sekarang pakai backtick — `mode` tersimpan dengan benar
        _update_state(**update_kwargs)
        new_state = _get_state(use_cache=False)

        log.info("Control: action=%s mode=%s → DB mode=%s pump=%s",
                 action, mode, new_state["mode"], new_state["pump_status"])

        return {
            "success"         : True,
            "debounced"       : False,
            "pump_status"     : new_state["pump_status"],
            "mode"            : new_state["mode"],
            "manual_override" : new_state.get("manual_override", False),
            "timestamp"       : now_ts,
        }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: SensorData):
    hour, _, _, _ = _resolve_time_wit(data.hour, data.minute, data.day)
    return {
        "input" : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
            "hour"         : hour,
        },
        "result": classify(data.soil_moisture, data.temperature, data.air_humidity, hour=hour),
    }


@app.get("/config", dependencies=[Depends(verify_api_key)])
def get_config():
    return {
        "version": APP_VERSION,
        "watering_windows": {
            "morning": f"{CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59",
            "evening": f"{CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59",
        },
        "soil_thresholds": {
            "dry_on_threshold"  : CFG.SOIL_DRY_ON,
            "wet_off_threshold" : CFG.SOIL_WET_OFF,
            "critical_emergency": CFG.CRITICAL_DRY,
        },
        "rain_detection": {
            "score_to_confirm": CFG.RAIN_SCORE_THRESHOLD,
            "score_to_clear"  : CFG.RAIN_CLEAR_THRESHOLD,
            "rh_heavy"        : CFG.RAIN_RH_HEAVY,
            "rh_moderate"     : CFG.RAIN_RH_MODERATE,
            "rh_light"        : CFG.RAIN_RH_LIGHT,
        },
        "pump_control": {
            "max_duration_min"       : CFG.MAX_PUMP_DURATION_MINUTES,
            "min_duration_sec"       : CFG.MIN_PUMP_DURATION_SECONDS,
            "cooldown_normal"        : CFG.COOLDOWN_MINUTES,
            "cooldown_post_rain"     : CFG.POST_RAIN_COOLDOWN_MINUTES,
            "manual_override_expire" : CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
        },
        "knn_confidence": {
            "normal"        : CFG.CONFIDENCE_NORMAL,
            "hot_weather"   : CFG.CONFIDENCE_HOT,
            "missed_session": CFG.CONFIDENCE_MISSED,
            "hot_threshold" : CFG.HOT_TEMP_THRESHOLD,
        },
        "time_weights": {
            "in_window"  : CFG.TIME_WEIGHT_IN_WINDOW,
            "near_window": CFG.TIME_WEIGHT_NEAR_WINDOW,
            "outside"    : CFG.TIME_WEIGHT_OUTSIDE,
        },
    }


@app.post("/reset-rain", dependencies=[Depends(verify_api_key)])
def reset_rain():
    _update_state(
        rain_detected=False, rain_score=0, rain_confirm_count=0,
        rain_clear_count=0, rain_started_minute=None, missed_session=False,
    )
    return {"success": True, "message": "State hujan di-reset."}


@app.post("/reset-override", dependencies=[Depends(verify_api_key)])
def reset_override():
    """Reset manual_override sebelum 10 menit timeout habis."""
    _update_state(manual_override=False, manual_override_ts=None)
    return {"success": True, "message": "Manual override di-reset. Auto-watering aktif kembali."}


@app.get("/diagnostics", dependencies=[Depends(verify_api_key)])
def get_diagnostics():
    """
    Tampilkan semua state internal, info daily safety, dan info model KNN.
    Berguna untuk debugging tanpa harus masuk ke database.
    """
    state = _get_state(use_cache=False)

    with _daily_safety_lock:
        safety_snapshot = dict(_daily_safety)

    if safety_snapshot.get("date"):
        safety_snapshot["date"] = str(safety_snapshot["date"])

    knn_info = {
        "model_loaded"  : knn_model is not None,
        "scaler_loaded" : scaler is not None,
        "meta"          : model_meta,
        "features_used" : ["soil_moisture", "temperature", "air_humidity"],
        "time_awareness": "via time_weight multiplier (no retraining needed)",
    }

    override_remaining = None
    if state.get("manual_override"):
        age = _elapsed_seconds_real(state.get("manual_override_ts"))
        remaining = CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - age
        override_remaining = max(0, int(remaining))

    return {
        "version"               : APP_VERSION,
        "server_time_wit"       : datetime.utcnow().strftime("%H:%M:%S") + " (UTC, +9=WIT)",
        "state"                 : {k: str(v) if v is not None else None for k, v in state.items()},
        "daily_safety"          : safety_snapshot,
        "override_remaining_sec": override_remaining,
        "knn"                   : knn_info,
    }