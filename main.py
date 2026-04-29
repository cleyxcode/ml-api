import os
import json
import uuid
import logging
import asyncio
import concurrent.futures
import joblib
import numpy as np
from datetime import datetime, date
from typing import Optional
import time

import firebase_admin
from firebase_admin import credentials, db as firebase_db

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

# ── Firebase config ───────────────────────────────────────────────────────────
FIREBASE_CRED_PATH    = os.environ.get(
    "FIREBASE_CRED_PATH",
    os.path.join(BASE_DIR, "iot-project-8494e-firebase-adminsdk-fbsvc-dcf9e0e4b6.json")
)
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DATABASE_URL",
    "https://iot-project-8494e-default-rtdb.asia-southeast1.firebasedatabase.app/"
)

# ── API Key ───────────────────────────────────────────────────────────────────
VALID_API_KEY  = os.environ.get("API_KEY", "yuli1")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Versi ─────────────────────────────────────────────────────────────────────
APP_VERSION = "8.0.0"
# ═══════════════════════════════════════════════════════════════════════════
# v8.0.0 — Real-time state, async-safe control, Firebase listener cache
# ═══════════════════════════════════════════════════════════════════════════
# Perubahan utama:
#   - _state_cache sekarang diperbarui via Firebase on_value listener (real-time)
#   - Cache TTL diturunkan 2s → 0.3s sebagai fallback jika listener lambat
#   - _control_lock diganti asyncio.Lock() agar benar-benar non-blocking
#   - lambda di run_in_executor diganti fungsi eksplisit (silent-bug fix)
#   - Debounce sensor dilonggarkan: hanya blokir jika <1 detik DAN delta <0.5%
#   - Mode switch: force invalidate cache + re-read Firebase setelah write
#   - _update_state_async tidak lagi memanggil _get_state di dalam lock
#   - Semua logika bisnis, safety, rain detection TIDAK BERUBAH
# ═══════════════════════════════════════════════════════════════════════════


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not VALID_API_KEY:
        log.warning("API_KEY belum di-set di environment variable!")
        return "no-key-configured"
    if api_key != VALID_API_KEY:
        log.warning("Akses ditolak: API key tidak valid '%s'", api_key)
        raise HTTPException(status_code=401, detail={
            "error"  : "Unauthorized",
            "message": "API key tidak valid atau tidak ada. Sertakan header: X-API-Key: <key>",
        })
    return api_key


# ── Locks ─────────────────────────────────────────────────────────────────────
# PERBAIKAN: _control_lock kini asyncio.Lock agar aman di event-loop async
_control_lock = asyncio.Lock()

_daily_safety_lock = asyncio.Lock()
_daily_safety = {
    "date"                 : None,
    "watering_count"       : 0,
    "locked_out"           : False,
    "last_pump_duration_sec": 0,
    "prune_done_today"     : False,
}

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="fb-worker")


# ══════════════════════════════════════════════════════════════════════════════
# FIREBASE INIT
# ══════════════════════════════════════════════════════════════════════════════
def _init_firebase():
    """Inisialisasi Firebase Admin SDK (idempotent)."""
    if firebase_admin._apps:
        return
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        log.info("Firebase Realtime DB terhubung: %s", FIREBASE_DATABASE_URL)
    except Exception as e:
        log.error("Gagal inisialisasi Firebase: %s", e)
        raise


def _ref_state() -> firebase_db.Reference:
    return firebase_db.reference("/system_state")


def _ref_sensor_readings() -> firebase_db.Reference:
    return firebase_db.reference("/sensor_readings")


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

    # PERBAIKAN: debounce kontrol dikurangi agar mode switch terasa instan
    CONTROL_DEBOUNCE_SECONDS = 1
    # PERBAIKAN: debounce sensor dilonggarkan
    SENSOR_DEBOUNCE_SECONDS  = 1
    SENSOR_TOLERANCE         = 0.5   # dari 1.0 → 0.5 agar data lebih sensitif

    MANUAL_OVERRIDE_EXPIRE_SECONDS = 600

    TIME_WEIGHT_IN_WINDOW   = 1.0
    TIME_WEIGHT_NEAR_WINDOW = 0.7
    TIME_WEIGHT_OUTSIDE     = 0.0


CFG = WateringConfig()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API — Firebase RT",
    description="Sistem Penyiraman Tanaman IoT — KNN + Firebase Realtime DB (v8.0.0)",
    version=APP_VERSION,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

knn_model  = None
scaler     = None
model_meta: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME STATE CACHE — diperbarui via Firebase listener
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

# Cache thread-safe via asyncio — diupdate oleh listener Firebase
_rt_cache: dict = {"data": None, "timestamp": 0.0}
_rt_cache_lock  = asyncio.Lock()   # digunakan hanya di coroutine
_listener_ref   = None             # handle listener agar bisa di-detach


def _normalize_state(raw: dict) -> dict:
    """Normalisasi tipe dari data Firebase ke Python native."""
    row = dict(_STATE_DEFAULTS)
    row.update(raw)
    for k in ("pump_status", "missed_session", "rain_detected", "manual_override"):
        row[k] = bool(row.get(k, False))
    for k in ("rain_score", "rain_confirm_count", "rain_clear_count", "session_count_today"):
        row[k] = int(row.get(k) or 0)
    return row


def _on_state_changed(event):
    """
    Firebase on_value listener — dipanggil setiap ada perubahan di /system_state.
    Runs di thread Firebase SDK, bukan event-loop. Gunakan thread-safe update.
    PERBAIKAN: cache diperbarui langsung tanpa butuh polling.
    """
    try:
        data = event.data
        if data is None:
            return
        normalized = _normalize_state(data)
        # Update cache secara thread-safe (tidak pakai asyncio.Lock di sini)
        _rt_cache["data"]      = normalized
        _rt_cache["timestamp"] = time.monotonic()
        log.debug("RT-cache updated via listener: pump=%s mode=%s",
                  normalized["pump_status"], normalized["mode"])
    except Exception as e:
        log.error("Firebase listener error: %s", e)


def _start_firebase_listener():
    """Pasang listener real-time ke /system_state."""
    global _listener_ref
    try:
        ref = _ref_state()
        _listener_ref = ref.listen(_on_state_changed)
        log.info("Firebase real-time listener terpasang di /system_state")
    except Exception as e:
        log.error("Gagal memasang Firebase listener: %s", e)


def _stop_firebase_listener():
    global _listener_ref
    if _listener_ref:
        try:
            _listener_ref.close()
            log.info("Firebase listener dihentikan.")
        except Exception:
            pass
        _listener_ref = None


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    global knn_model, scaler, model_meta
    log.info("Siram Pintar API v%s (Firebase RT) starting...", APP_VERSION)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _init_firebase)

    if VALID_API_KEY:
        log.info("API Key protection: AKTIF (key: %s***)", VALID_API_KEY[:2])

    await loop.run_in_executor(_executor, _ensure_state_node)

    # Pasang real-time listener di background thread
    await loop.run_in_executor(_executor, _start_firebase_listener)

    # Beri waktu listener populate cache pertama kali
    await asyncio.sleep(0.5)

    await _sync_daily_safety_from_db()

    if not os.path.exists(MODEL_PATH):
        log.warning("Model belum ada! Jalankan train_model.py terlebih dahulu.")
        return
    try:
        knn_model = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                model_meta = json.load(f)
        log.info("Model KNN dimuat. K=%s, Akurasi=%s%%",
                 model_meta.get("best_k"), model_meta.get("accuracy"))
    except Exception as exc:
        log.error("Gagal memuat model: %s", exc)


@app.on_event("shutdown")
async def shutdown():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _stop_firebase_listener)
    _executor.shutdown(wait=False)
    log.info("Siram Pintar API shutdown selesai.")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
class SensorData(BaseModel):
    soil_moisture : float = Field(..., ge=0, le=100)
    temperature   : float = Field(..., ge=0, le=60)
    air_humidity  : float = Field(..., ge=0, le=100)
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
# HELPER: KNN time encoding
# ══════════════════════════════════════════════════════════════════════════════
def _encode_hour_cyclic(hour: int) -> tuple:
    angle = 2 * np.pi * hour / 24
    return float(np.sin(angle)), float(np.cos(angle))


def _get_time_weight(hour: int) -> float:
    in_window, _ = _in_watering_window(hour)
    if in_window:
        return CFG.TIME_WEIGHT_IN_WINDOW
    ms, me = CFG.MORNING_WINDOW[0], CFG.MORNING_WINDOW[1]
    es, ee = CFG.EVENING_WINDOW[0], CFG.EVENING_WINDOW[1]
    if hour == ms - 1 or hour == me + 1 or hour == es - 1 or hour == ee + 1:
        return CFG.TIME_WEIGHT_NEAR_WINDOW
    return CFG.TIME_WEIGHT_OUTSIDE


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _ensure_state_node():
    try:
        ref  = _ref_state()
        data = ref.get()
        if data is None:
            ref.set(_STATE_DEFAULTS)
            log.info("system_state node dibuat di Firebase.")
    except Exception as e:
        log.error("Gagal memastikan system_state node: %s", e)


def _fb_get_state_sync() -> dict:
    """Baca state langsung dari Firebase (bypass cache) — hanya untuk fallback."""
    try:
        data = _ref_state().get()
    except Exception as e:
        log.error("Firebase get state error: %s", e)
        data = None
    if data is None:
        return dict(_STATE_DEFAULTS)
    return _normalize_state(data)


def _get_state(force_fresh: bool = False) -> dict:
    """
    Ambil state dari cache real-time.
    PERBAIKAN: cache TTL 0.3s sebagai safety-net; listener Firebase adalah sumber utama.
    force_fresh=True: paksa baca langsung dari Firebase (digunakan setelah write penting).
    """
    if force_fresh:
        row = _fb_get_state_sync()
        _rt_cache["data"]      = row
        _rt_cache["timestamp"] = time.monotonic()
        return row

    cached = _rt_cache["data"]
    age    = time.monotonic() - _rt_cache["timestamp"]

    if cached and age < 0.3:
        return cached.copy()

    # Cache stale atau kosong → baca langsung
    try:
        row = _fb_get_state_sync()
        _rt_cache["data"]      = row
        _rt_cache["timestamp"] = time.monotonic()
        return row
    except Exception as e:
        log.error("Fallback get state gagal: %s", e)
        return cached.copy() if cached else dict(_STATE_DEFAULTS)


def _fb_update_state_sync(**kwargs):
    """Update field di /system_state secara atomic (sync, untuk thread pool)."""
    if not kwargs:
        return
    _ref_state().update(kwargs)
    log.info("State updated: %s", list(kwargs.keys()))


async def _update_state_async(**kwargs):
    """
    Update state async.
    PERBAIKAN: tidak ada lock global; Firebase .update() sudah atomic per field.
    Setelah write, cache di-force-refresh agar tidak tunggu listener.
    """
    if not kwargs:
        return
    loop = asyncio.get_event_loop()

    def _do():
        _fb_update_state_sync(**kwargs)
        # Segera baca balik untuk perbarui cache lokal
        fresh = _fb_get_state_sync()
        _rt_cache["data"]      = fresh
        _rt_cache["timestamp"] = time.monotonic()

    await loop.run_in_executor(_executor, _do)


# ══════════════════════════════════════════════════════════════════════════════
# DAILY SAFETY
# ══════════════════════════════════════════════════════════════════════════════
async def _sync_daily_safety_from_db():
    loop = asyncio.get_event_loop()
    row  = await loop.run_in_executor(_executor, _fb_get_state_sync)

    db_count    = int(row.get("session_count_today") or 0)
    db_date_raw = row.get("session_count_date")
    db_date     = None
    if db_date_raw:
        try:
            db_date = date.fromisoformat(str(db_date_raw)[:10])
        except Exception:
            pass

    today = date.today()
    async with _daily_safety_lock:
        if db_date == today:
            _daily_safety["date"]           = today
            _daily_safety["watering_count"] = db_count
            _daily_safety["locked_out"]     = (db_count >= 10)
            log.info("_sync_daily_safety: recovered watering_count=%d", db_count)
        else:
            _daily_safety["date"]           = today
            _daily_safety["watering_count"] = 0
            _daily_safety["locked_out"]     = False
            log.info("_sync_daily_safety: hari baru, counter direset.")


def _daily_safety_reset_if_new_day():
    """Harus dipanggil di dalam _daily_safety_lock."""
    today = date.today()
    if _daily_safety["date"] != today:
        _daily_safety["date"]             = today
        _daily_safety["watering_count"]   = 0
        _daily_safety["locked_out"]       = False
        _daily_safety["prune_done_today"] = False
        return True
    return False


def _prune_sensor_readings():
    """Hapus sensor_readings > 14 hari."""
    try:
        cutoff = datetime.now().timestamp() - (14 * 86400)
        ref    = _ref_sensor_readings()
        data   = ref.order_by_child("timestamp_unix").end_at(cutoff).get()
        if data:
            ref.update({k: None for k in data.keys()})
            log.info("Pruned %d old sensor readings.", len(data))
    except Exception as e:
        log.error("Prune error: %s", e)


async def _maybe_schedule_prune(bg_tasks: BackgroundTasks):
    async with _daily_safety_lock:
        _daily_safety_reset_if_new_day()
        if not _daily_safety["prune_done_today"]:
            _daily_safety["prune_done_today"] = True
            bg_tasks.add_task(_prune_sensor_readings)


# ══════════════════════════════════════════════════════════════════════════════
# KNN Classify
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
        tw    = _get_time_weight(hour)
        hs, hc = _encode_hour_cyclic(hour)
        return {
            "label"                   : label,
            "confidence"              : conf,
            "time_weight"             : tw,
            "time_adjusted_confidence": round(conf * tw, 2),
            "hour_sin"                : hs,
            "hour_cos"                : hc,
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
def _compute_rain_score(air_humidity, soil_moisture, temperature,
                        last_soil, last_temp, pump_was_on):
    score, signals = 0, []
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
        drop = float(last_temp) - temperature
        if drop >= CFG.RAIN_TEMP_DROP:
            score += 15; signals.append(f"suhu turun -{drop:.1f}°C")

    return min(score, 100), signals


def _update_rain_state_batched(score, signals, state, current_min) -> tuple:
    currently = state["rain_detected"]
    confirm   = state["rain_confirm_count"]
    clear     = state["rain_clear_count"]
    updates   = {}

    if score >= CFG.RAIN_SCORE_THRESHOLD:
        confirm += 1; clear = 0
        if not currently and confirm >= CFG.RAIN_CONFIRM_READINGS:
            updates = dict(rain_detected=True, rain_score=score,
                           rain_confirm_count=confirm, rain_clear_count=0,
                           rain_started_minute=current_min, missed_session=True)
            return True, f"Hujan dikonfirmasi (skor={score})", updates
        elif currently:
            updates = dict(rain_score=score, rain_confirm_count=confirm, rain_clear_count=0)
            return True, f"Hujan berlanjut (skor={score})", updates
        else:
            updates = dict(rain_score=score, rain_confirm_count=confirm, rain_clear_count=0)
            return False, f"Menunggu konfirmasi ({confirm}/{CFG.RAIN_CONFIRM_READINGS})", updates

    elif score <= CFG.RAIN_CLEAR_THRESHOLD:
        clear += 1; confirm = 0
        if currently and clear >= CFG.RAIN_CLEAR_READINGS:
            updates = dict(rain_detected=False, rain_score=score,
                           rain_confirm_count=0, rain_clear_count=clear)
            return False, "", updates
        elif currently:
            updates = dict(rain_score=score, rain_confirm_count=0, rain_clear_count=clear)
            return True, "Hujan mungkin selesai, tunggu konfirmasi", updates
        else:
            updates = dict(rain_score=score, rain_confirm_count=0, rain_clear_count=clear)
            return False, "", updates
    else:
        if currently:
            updates = dict(rain_score=score)
            return True, f"Hujan ambiguos (skor={score})", updates
        return False, "", {}


def _should_skip_sensor(data: SensorData, state: dict, pump_is_on: bool) -> bool:
    """
    PERBAIKAN: debounce diperketat — hanya skip jika:
    - data anomali (sensor mati / suhu ekstrem), ATAU
    - selisih sangat kecil (<0.5%) DAN baru saja dikirim (<1 detik)
    """
    # Anomali sensor — skip selalu
    if data.soil_moisture <= 0.0 or data.temperature <= 0.0 or data.temperature >= 60.0:
        log.warning("ANOMALI SENSOR: Soil=%.1f%% Temp=%.1f°C",
                    data.soil_moisture, data.temperature)
        return True

    last_soil = state.get("last_sensor_soil")
    if last_soil is not None and abs(data.soil_moisture - float(last_soil)) > 30.0:
        if not pump_is_on:
            log.warning("ANOMALI: Perubahan >30%% tanpa pompa (%.1f%% → %.1f%%)",
                        float(last_soil), data.soil_moisture)
            return True

    elapsed = _elapsed_seconds_real(state.get("last_sensor_ts"))

    # Terlalu cepat DAN perubahan tidak signifikan → skip
    if elapsed < CFG.SENSOR_DEBOUNCE_SECONDS:
        if last_soil is None:
            return False
        if abs(data.soil_moisture - float(last_soil)) <= CFG.SENSOR_TOLERANCE:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# SMART WATERING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
async def _evaluate_smart_watering_async(result, hour, minute, soil_moisture,
                                         air_humidity, temperature, state,
                                         current_total_minutes) -> dict:
    resp = {
        "action"         : None,
        "reason"         : "",
        "blocked_reason" : None,
        "is_raining"     : False,
        "rain_score"     : 0,
        "hot_mode"       : temperature >= CFG.HOT_TEMP_THRESHOLD,
        "missed_session" : bool(state.get("missed_session", False)),
        "decision_path"  : [],
        "time_weight"    : result.get("time_weight", 1.0),
        "pending_updates": {},
    }

    # PERBAIKAN: gunakan asyncio.Lock agar tidak blocking event loop
    async with _daily_safety_lock:
        _daily_safety_reset_if_new_day()
        if _daily_safety["locked_out"]:
            resp["blocked_reason"] = "Safety Lockout: Melebihi batas harian (10x)."
            resp["decision_path"].append("SAFETY_LOCKOUT")
            return resp

    if state.get("manual_override"):
        age = _elapsed_seconds_real(state.get("manual_override_ts"))
        if age < CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS:
            remaining = int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - age)
            resp["blocked_reason"] = (
                f"Manual override aktif: pompa dikunci off ({remaining}s lagi)"
            )
            resp["decision_path"].append("MANUAL_OVERRIDE_BLOCK")
            return resp
        else:
            log.info("Manual override expired, reset otomatis.")
            resp["pending_updates"].update(manual_override=False, manual_override_ts=None)

    rain_score, rain_signals = _compute_rain_score(
        air_humidity=air_humidity, soil_moisture=soil_moisture,
        temperature=temperature, last_soil=state.get("last_soil_moisture"),
        last_temp=state.get("last_temperature"), pump_was_on=bool(state["pump_status"]),
    )
    is_raining, rain_reason, rain_updates = _update_rain_state_batched(
        rain_score, rain_signals, state, current_total_minutes
    )
    resp["pending_updates"].update(rain_updates)
    resp["is_raining"] = is_raining
    resp["rain_score"] = rain_score

    dynamic_dry_on  = CFG.SOIL_DRY_ON
    dynamic_wet_off = CFG.SOIL_WET_OFF

    if resp["hot_mode"]:
        dynamic_dry_on  += 5.0; dynamic_wet_off += 5.0
        resp["decision_path"].append("T-HOT_ADJUST")
    elif temperature < 25.0 and air_humidity > 80.0:
        dynamic_dry_on  -= 5.0; dynamic_wet_off -= 5.0
        resp["decision_path"].append("T-COOL_ADJUST")

    if state.get("missed_session"):
        dynamic_wet_off += 5.0
        resp["decision_path"].append("T-MISSED_ADJUST")

    dynamic_wet_off = min(95.0, dynamic_wet_off)
    dynamic_dry_on  = max(CFG.CRITICAL_DRY + 5.0, dynamic_dry_on)

    in_window, window_label = _in_watering_window(hour)
    night_emergency = not in_window and soil_moisture <= CFG.CRITICAL_DRY and not is_raining
    if night_emergency:
        window_label = "malam-darurat"

    # ── Pompa sedang ON ───────────────────────────────────────────────────────
    if state["pump_status"]:
        elapsed_sec = _elapsed_seconds_real(state.get("pump_start_ts"))
        max_sec     = 60 if night_emergency else (CFG.MAX_PUMP_DURATION_MINUTES * 60)

        if elapsed_sec >= max_sec:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = f"Auto-stop: {elapsed_sec:.0f}s"
            resp["decision_path"].append("A1")
            return resp

        if elapsed_sec < CFG.MIN_PUMP_DURATION_SECONDS:
            resp["reason"] = f"Warmup ({elapsed_sec:.0f}s)"
            resp["decision_path"].append("A-warmup")
            return resp

        if soil_moisture >= dynamic_wet_off:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = f"Tanah cukup ({soil_moisture:.1f}%)"
            resp["decision_path"].append("A2")
            return resp

        if is_raining:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = "Hujan terdeteksi"
            resp["decision_path"].append("A3")
            return resp

        resp["reason"] = f"Running ({elapsed_sec:.0f}s)"
        resp["decision_path"].append("A4-running")
        return resp

    # ── Helper tambah sesi ────────────────────────────────────────────────────
    async def _add_pump_on_updates(updates: dict):
        async with _daily_safety_lock:
            _daily_safety["watering_count"] += 1
            new_count = _daily_safety["watering_count"]
            if new_count >= 10:
                _daily_safety["locked_out"] = True
        updates["session_count_today"] = new_count
        updates["session_count_date"]  = date.today().isoformat()

    # ── Darurat ───────────────────────────────────────────────────────────────
    if night_emergency or (soil_moisture <= CFG.CRITICAL_DRY and not is_raining):
        now_ts = datetime.now().isoformat()
        pump_u = dict(pump_status=True, pump_start_minute=current_total_minutes,
                      pump_start_ts=now_ts)
        await _add_pump_on_updates(pump_u)
        resp["pending_updates"].update(pump_u)
        resp["action"] = "on"
        resp["reason"] = f"SIRAM DARURAT [{window_label}]: tanah {soil_moisture:.1f}%"
        resp["decision_path"].append("B1")
        return resp

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
            resp["pending_updates"]["missed_session"] = False
        resp["blocked_reason"] = f"Tanah sudah basah ({soil_moisture:.1f}%)"
        resp["decision_path"].append("B4")
        return resp

    effective_cd = (CFG.POST_RAIN_COOLDOWN_MINUTES if state.get("missed_session")
                    else CFG.COOLDOWN_MINUTES)
    elapsed_cd = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < effective_cd:
        resp["blocked_reason"] = f"Cooldown: sisa {effective_cd - elapsed_cd} mnt"
        resp["decision_path"].append("B5")
        return resp

    if not result["needs_watering"]:
        resp["blocked_reason"] = f"KNN: {result['label']} ({result['confidence']}%)"
        resp["decision_path"].append("B6")
        return resp

    base_thr = (CFG.CONFIDENCE_HOT if resp["hot_mode"]
                else (CFG.CONFIDENCE_MISSED if state.get("missed_session")
                      else CFG.CONFIDENCE_NORMAL))
    tw = result.get("time_weight", 1.0)
    eff_thr = min(base_thr * (1.0 / tw), 95.0) if 0.0 < tw < 1.0 else base_thr
    if 0.0 < tw < 1.0:
        resp["decision_path"].append(f"T-TIME_ADJ({tw:.1f})")

    if result["confidence"] < eff_thr:
        resp["blocked_reason"] = (
            f"Confidence {result['confidence']}% < threshold {eff_thr:.0f}%"
            f" (time_weight={tw:.1f})"
        )
        resp["decision_path"].append("B7")
        return resp

    if soil_moisture > dynamic_dry_on:
        resp["blocked_reason"] = f"Tanah {soil_moisture:.1f}% > batas ({dynamic_dry_on:.1f}%)"
        resp["decision_path"].append("B8")
        return resp

    now_ts = datetime.now().isoformat()
    pump_u = dict(pump_status=True, pump_start_minute=current_total_minutes,
                  pump_start_ts=now_ts)
    await _add_pump_on_updates(pump_u)
    resp["pending_updates"].update(pump_u)
    resp["action"] = "on"
    resp["reason"] = (
        f"Siram [{window_label}]: KNN {result['label']} ({result['confidence']}%), "
        f"T={temperature:.1f}°C, time_weight={tw:.1f}"
    )
    resp["decision_path"].append("B-FINAL")
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS PUBLIC
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status"      : "online",
        "message"     : "Siram Pintar API berjalan (Firebase RT)",
        "version"     : APP_VERSION,
        "model_ready" : knn_model is not None,
        "auth"        : "required" if VALID_API_KEY else "disabled",
        "database"    : "Firebase Realtime DB + listener",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS PROTECTED
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/db-test", dependencies=[Depends(verify_api_key)])
async def db_test():
    loop = asyncio.get_event_loop()
    def _test():
        ref = firebase_db.reference("/ping_test")
        ref.set({"ts": datetime.now().isoformat(), "ok": True})
        val = ref.get()
        ref.delete()
        return val
    try:
        result = await loop.run_in_executor(_executor, _test)
        return {"db_status": "connected", "firebase_url": FIREBASE_DATABASE_URL, "result": result}
    except Exception as e:
        return {"db_status": "error", "detail": str(e)}


@app.get("/model-info", dependencies=[Depends(verify_api_key)])
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


@app.post("/sensor", dependencies=[Depends(verify_api_key)])
async def receive_sensor(data: SensorData, bg_tasks: BackgroundTasks):
    hour, minute, _day, time_source = _resolve_time_wit(data.hour, data.minute, data.day)
    result    = classify(data.soil_moisture, data.temperature, data.air_humidity, hour=hour)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_id    = str(uuid.uuid4())

    state                = _get_state()
    current_total_min    = _total_minutes(hour, minute)

    await _maybe_schedule_prune(bg_tasks)

    pump_is_on = bool(state.get("pump_status", False))
    skip_eval  = _should_skip_sensor(data, state, pump_is_on)

    # Debounce sangat ketat (< 1 detik persis sama) → return cepat
    if skip_eval:
        elapsed_spam = _elapsed_seconds_real(state.get("last_sensor_ts"))
        if elapsed_spam < 1.0:
            return {
                "received"      : True,
                "timestamp"     : state.get("last_updated") or timestamp,
                "device_time"   : f"{hour:02d}:{minute:02d}",
                "time_source"   : time_source,
                "debounced"     : True,
                "sensor"        : {
                    "soil_moisture": data.soil_moisture,
                    "temperature"  : data.temperature,
                    "air_humidity" : data.air_humidity,
                },
                "classification": result,
                "pump_status"   : state["pump_status"],
                "pump_action"   : None,
                "mode"          : state["mode"],
                "auto_info"     : None,
            }

    final_action = None
    smart_eval   = {}

    if state["mode"] == "auto" and not skip_eval:
        smart_eval = await _evaluate_smart_watering_async(
            result=result, hour=hour, minute=minute,
            soil_moisture=data.soil_moisture, air_humidity=data.air_humidity,
            temperature=data.temperature, state=state,
            current_total_minutes=current_total_min,
        )
        final_action = smart_eval.get("action")

    pump_status_logged = (
        (final_action == "on") if final_action is not None else state["pump_status"]
    )
    sensor_updates = dict(
        last_label=result["label"], last_updated=timestamp,
        last_soil_moisture=data.soil_moisture, last_temperature=data.temperature,
        last_sensor_ts=datetime.now().isoformat(), last_sensor_soil=data.soil_moisture,
    )
    pending     = smart_eval.get("pending_updates", {})
    all_updates = {**sensor_updates, **pending}

    loop = asyncio.get_event_loop()

    # PERBAIKAN: fungsi eksplisit (bukan lambda) — bug silent tuple return
    def _save_all():
        _fb_update_state_sync(**all_updates)
        fresh = _fb_get_state_sync()
        _rt_cache["data"]      = fresh
        _rt_cache["timestamp"] = time.monotonic()

        _ref_sensor_readings().child(row_id).set({
            "id"            : row_id,
            "timestamp"     : timestamp,
            "timestamp_unix": datetime.now().timestamp(),
            "soil_moisture" : data.soil_moisture,
            "temperature"   : data.temperature,
            "air_humidity"  : data.air_humidity,
            "label"         : result["label"],
            "confidence"    : result["confidence"],
            "needs_watering": result["needs_watering"],
            "description"   : result["description"],
            "probabilities" : result["probabilities"],
            "pump_status"   : pump_status_logged,
            "mode"          : state["mode"],
        })

    try:
        await loop.run_in_executor(_executor, _save_all)
    except Exception as e:
        log.error("Failed to save to Firebase: %s", e)

    new_state = _get_state()
    return {
        "received"      : True,
        "timestamp"     : timestamp,
        "device_time"   : f"{hour:02d}:{minute:02d}",
        "time_source"   : time_source,
        "debounced"     : skip_eval,
        "sensor"        : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
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
async def get_status():
    # PERBAIKAN: tidak force-fresh di sini; listener sudah jaga kebaruan cache
    state = _get_state()
    loop  = asyncio.get_event_loop()

    def _get_latest():
        try:
            data = (
                _ref_sensor_readings()
                .order_by_child("timestamp_unix")
                .limit_to_last(1)
                .get()
            )
            if data:
                return list(data.values())[0]
        except Exception as e:
            log.error("Gagal ambil latest sensor: %s", e)
        return None

    latest = await loop.run_in_executor(_executor, _get_latest)

    async with _daily_safety_lock:
        watering_today = _daily_safety["watering_count"]
        locked_out     = _daily_safety["locked_out"]

    return {
        "pump_status"     : state["pump_status"],
        "mode"            : state["mode"],
        "last_label"      : state["last_label"],
        "last_updated"    : str(state["last_updated"]) if state["last_updated"] else None,
        "is_raining"      : state.get("rain_detected", False),
        "rain_score"      : state.get("rain_score", 0),
        "missed_session"  : state.get("missed_session", False),
        "manual_override" : state.get("manual_override", False),
        "watering_today"  : watering_today,
        "safety_locked"   : locked_out,
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
    state = _get_state()
    return {
        "pump_status"    : state["pump_status"],
        "mode"           : state["mode"],
        "manual_override": state.get("manual_override", False),
        "version"        : APP_VERSION,
    }


@app.get("/history", dependencies=[Depends(verify_api_key)])
async def get_history(limit: int = Query(default=50, ge=1, le=500)):
    loop = asyncio.get_event_loop()

    def _fetch():
        try:
            data = (
                _ref_sensor_readings()
                .order_by_child("timestamp_unix")
                .limit_to_last(limit)
                .get()
            )
            if not data:
                return []
            return sorted(data.values(), key=lambda x: x.get("timestamp_unix", 0))
        except Exception as e:
            log.error("History error: %s", e)
            return []

    records = await loop.run_in_executor(_executor, _fetch)
    return {"total": len(records), "records": records}


@app.post("/control", dependencies=[Depends(verify_api_key)])
async def control_pump(cmd: ControlCommand):
    """
    PERBAIKAN UTAMA:
    - asyncio.Lock (bukan threading.Lock) → tidak blocking event loop
    - Setelah write, force-refresh cache dari Firebase
    - Deteksi mode-change saja (tanpa harus ubah pump_status) sudah benar disimpan
    - Return state yang sudah fresh, bukan state sebelum write
    """
    action = (cmd.action or "").lower().strip()
    if action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'.")

    mode = (cmd.mode or "manual").lower().strip()
    if mode not in ("auto", "manual"):
        mode = "manual"

    loop = asyncio.get_event_loop()

    async with _control_lock:
        # Baca state fresh dari Firebase (bukan cache) untuk hindari race condition
        state   = await loop.run_in_executor(_executor, _fb_get_state_sync)
        pump_on = action == "on"
        now_ts  = datetime.now().isoformat()

        # Cek apakah benar-benar ada perubahan
        pump_changed = state["pump_status"] != pump_on
        mode_changed = state["mode"] != mode

        if not pump_changed and not mode_changed:
            return {
                "success"        : True,
                "debounced"      : True,
                "message"        : "Status tidak berubah",
                "pump_status"    : state["pump_status"],
                "mode"           : state["mode"],
                "manual_override": state.get("manual_override", False),
                "timestamp"      : state.get("last_control_ts") or now_ts,
            }

        update_kwargs: dict = {"last_control_ts": now_ts}

        # Update mode jika berubah
        if mode_changed:
            update_kwargs["mode"] = mode
            log.info("Mode berubah: %s → %s", state["mode"], mode)

        # Update pump jika berubah
        if pump_changed:
            update_kwargs["pump_status"] = pump_on

            if not pump_on:
                # Matikan pompa secara manual
                current_min = _total_minutes(*_resolve_time_wit(None, None, None)[:2])
                update_kwargs.update(
                    pump_start_ts=None,
                    pump_start_minute=None,
                    last_watered_ts=now_ts,
                    last_watered_minute=current_min,
                    manual_override=True,
                    manual_override_ts=now_ts,
                )
                log.info("Pompa OFF manual — manual_override diaktifkan.")
            else:
                # Nyalakan pompa secara manual
                now_utc = datetime.utcnow()
                h_wit   = (now_utc.hour + 9) % 24
                update_kwargs.update(
                    pump_start_ts=now_ts,
                    pump_start_minute=_total_minutes(h_wit, now_utc.minute),
                    manual_override=False,
                    manual_override_ts=None,
                )
                log.info("Pompa ON manual.")

        # Tulis ke Firebase
        def _write_and_refresh():
            _fb_update_state_sync(**update_kwargs)
            # Force-refresh cache segera setelah write
            fresh = _fb_get_state_sync()
            _rt_cache["data"]      = fresh
            _rt_cache["timestamp"] = time.monotonic()
            return fresh

        try:
            new_state = await loop.run_in_executor(_executor, _write_and_refresh)
        except Exception as e:
            log.error("Control write gagal: %s", e)
            raise HTTPException(status_code=503, detail="Gagal menyimpan ke Firebase.")

        log.info("Control OK: action=%s mode=%s → pump=%s mode=%s",
                 action, mode, new_state["pump_status"], new_state["mode"])

        return {
            "success"        : True,
            "debounced"      : False,
            "pump_status"    : new_state["pump_status"],
            "mode"           : new_state["mode"],
            "manual_override": new_state.get("manual_override", False),
            "timestamp"      : now_ts,
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
        "version"  : APP_VERSION,
        "database" : "Firebase Realtime DB + RT listener",
        "firebase_url": FIREBASE_DATABASE_URL,
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
            "max_duration_min"      : CFG.MAX_PUMP_DURATION_MINUTES,
            "min_duration_sec"      : CFG.MIN_PUMP_DURATION_SECONDS,
            "cooldown_normal"       : CFG.COOLDOWN_MINUTES,
            "cooldown_post_rain"    : CFG.POST_RAIN_COOLDOWN_MINUTES,
            "manual_override_expire": CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
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
        "realtime": {
            "listener_active"   : _listener_ref is not None,
            "cache_ttl_fallback": "0.3s",
            "sensor_debounce"   : f"{CFG.SENSOR_DEBOUNCE_SECONDS}s",
            "sensor_tolerance"  : f"{CFG.SENSOR_TOLERANCE}%",
        },
    }


@app.post("/reset-rain", dependencies=[Depends(verify_api_key)])
async def reset_rain():
    await _update_state_async(
        rain_detected=False, rain_score=0, rain_confirm_count=0,
        rain_clear_count=0, rain_started_minute=None, missed_session=False,
    )
    return {"success": True, "message": "State hujan di-reset."}


@app.post("/reset-override", dependencies=[Depends(verify_api_key)])
async def reset_override():
    await _update_state_async(manual_override=False, manual_override_ts=None)
    return {"success": True, "message": "Manual override di-reset. Auto-watering aktif kembali."}


@app.get("/diagnostics", dependencies=[Depends(verify_api_key)])
async def get_diagnostics():
    state = _get_state(force_fresh=True)

    async with _daily_safety_lock:
        safety_snapshot = {
            k: str(v) if isinstance(v, date) else v
            for k, v in _daily_safety.items()
        }

    override_remaining = None
    if state.get("manual_override"):
        age = _elapsed_seconds_real(state.get("manual_override_ts"))
        override_remaining = max(0, int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - age))

    cache_age = round(time.monotonic() - _rt_cache["timestamp"], 3)

    return {
        "version"               : APP_VERSION,
        "server_time_wit"       : datetime.utcnow().strftime("%H:%M:%S") + " (UTC+9=WIT)",
        "state"                 : {k: str(v) if v is not None else None
                                   for k, v in state.items()},
        "daily_safety"          : safety_snapshot,
        "override_remaining_sec": override_remaining,
        "knn"                   : {
            "model_loaded" : knn_model is not None,
            "scaler_loaded": scaler is not None,
            "meta"         : model_meta,
        },
        "realtime_cache"        : {
            "listener_active": _listener_ref is not None,
            "cache_age_sec"  : cache_age,
            "cache_valid"    : cache_age < 1.0,
        },
        "database": {
            "type"         : "Firebase Realtime Database",
            "url"          : FIREBASE_DATABASE_URL,
            "state_path"   : "/system_state",
            "readings_path": "/sensor_readings",
        },
        "migrations_from": "v7.0.0 (sync lock) → v8.0.0 (async lock + RT listener)",
    }