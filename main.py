import os
import json
import uuid
import random
import logging
import threading
import joblib
import numpy as np
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

import pymysql
import pymysql.cursors
from fastapi import FastAPI, HTTPException, Query, Security, Depends
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
# Simpan di environment variable: API_KEY=xxxxx (5 huruf+angka, uppercase)
# Contoh: API_KEY=A7K2M
VALID_API_KEY = os.environ.get("API_KEY", "")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """
    Validasi API Key dari header X-API-Key.
    Jika salah/tidak ada → 401 Unauthorized.
    """
    if not VALID_API_KEY:
        # Jika API_KEY belum di-set di env, log warning dan tetap izinkan
        log.warning("API_KEY belum di-set di environment variable!")
        return "no-key-configured"
    if api_key != VALID_API_KEY:
        log.warning("Akses ditolak: API key tidak valid '%s'", api_key)
        raise HTTPException(
            status_code=401,
            detail={
                "error"  : "Unauthorized",
                "message": "API key tidak valid atau tidak ada. Sertakan header: X-API-Key: <key>",
            },
        )
    return api_key


# ── Global lock: mencegah race condition saat ESP32 kirim data cepat ──────────
_sensor_lock = threading.Lock()
_control_lock = threading.Lock()

# ── Global Safety State ──────────────
_daily_safety = {
    "date": None,
    "watering_count": 0,
    "locked_out": False,
    "last_pump_duration_sec": 0
}


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
    SENSOR_DEBOUNCE_SECONDS = 10
    SENSOR_TOLERANCE        = 1.0


CFG = WateringConfig()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API",
    description="Sistem Penyiraman Tanaman IoT — KNN + Logika Cuaca Adaptif v6",
    version="6.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

knn_model  = None
scaler     = None
model_meta: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
@contextmanager
def get_db():
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASS, database=DB_NAME,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15, ssl={"ssl": {}},
        autocommit=False,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@app.on_event("startup")
async def startup():
    global knn_model, scaler, model_meta
    if VALID_API_KEY:
        log.info("API Key protection: AKTIF (key: %s***)", VALID_API_KEY[:2])
    else:
        log.warning("API Key protection: TIDAK AKTIF — set API_KEY di environment!")

    if not os.path.exists(MODEL_PATH):
        log.warning("Model belum ada! Jalankan train_model.py terlebih dahulu.")
        return
    try:
        knn_model  = joblib.load(MODEL_PATH)
        scaler     = joblib.load(SCALER_PATH)
        model_meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {"best_k": "?", "accuracy": "?"}
        log.info("Model KNN dimuat. K=%s, Akurasi=%s%%", model_meta.get("best_k"), model_meta.get("accuracy"))
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
    now        = datetime.utcnow()
    h          = (now.hour + 9) % 24
    wday       = (now.weekday() + 1) % 7
    log.warning("Fallback waktu server WIT: %02d:%02d", h, now.minute)
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
# HELPER: system_state
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
}


def _get_state() -> dict:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_state WHERE id = 1")
            row = cur.fetchone()
    if not row:
        return dict(_STATE_DEFAULTS)

    for bool_key in ("pump_status", "missed_session", "rain_detected"):
        row[bool_key] = bool(row.get(bool_key, False))

    for int_key in ("rain_score", "rain_confirm_count", "rain_clear_count", "session_count_today"):
        row[int_key] = int(row.get(int_key) or 0)

    for k, v in _STATE_DEFAULTS.items():
        if k not in row:
            row[k] = v

    return row


def _update_state(**kwargs):
    if not kwargs:
        return
    sets   = ", ".join(f"{k} = %s" for k in kwargs)
    values = list(kwargs.values())
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE system_state SET {sets} WHERE id = 1", values)


# ══════════════════════════════════════════════════════════════════════════════
# KNN Classify
# ══════════════════════════════════════════════════════════════════════════════
def classify(soil: float, temp: float, rh: float) -> dict:
    if knn_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model KNN belum dimuat.")
    feat   = scaler.transform(np.array([[soil, temp, rh]]))
    label  = knn_model.predict(feat)[0]
    proba  = knn_model.predict_proba(feat)[0]
    confs  = {cls: round(float(p) * 100, 2) for cls, p in zip(knn_model.classes_, proba)}
    conf   = round(float(max(proba)) * 100, 2)
    return {
        "label"         : label,
        "confidence"    : conf,
        "probabilities" : confs,
        "needs_watering": label == "Kering",
        "description"   : model_meta.get("label_desc", {}).get(label, ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DETEKSI HUJAN
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
            score += 35; signals.append(f"tanah +{delta:.1f}% (cepat)")
        elif delta >= CFG.RAIN_SOIL_RISE_LIGHT:
            score += 20; signals.append(f"tanah +{delta:.1f}% (perlahan)")

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
            log.info("HUJAN DIKONFIRMASI: skor=%d, sinyal=%s", score, signals)
            _update_state(
                rain_detected=True, rain_score=score,
                rain_confirm_count=confirm_count, rain_clear_count=0,
                rain_started_minute=current_min, missed_session=True,
            )
            return True, f"Hujan dikonfirmasi (skor={score}, {', '.join(signals)})"
        elif currently_raining:
            _update_state(rain_score=score, rain_confirm_count=confirm_count, rain_clear_count=0)
            return True, f"Hujan berlanjut (skor={score})"
        else:
            _update_state(rain_score=score, rain_confirm_count=confirm_count, rain_clear_count=0)
            return False, f"Menunggu konfirmasi hujan ({confirm_count}/{CFG.RAIN_CONFIRM_READINGS}, skor={score})"

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
            return True, f"Hujan mungkin selesai, tunggu konfirmasi ({clear_count}/{CFG.RAIN_CLEAR_READINGS})"
        else:
            _update_state(rain_score=score, rain_confirm_count=0, rain_clear_count=clear_count)
            return False, ""
    else:
        if currently_raining:
            _update_state(rain_score=score)
            return True, f"Hujan ambiguos (skor={score}), tetap aktif"
        return False, ""


def _should_skip_sensor(data: SensorData, state: dict) -> bool:
    if data.soil_moisture <= 0.0 or data.temperature <= 0.0 or data.temperature >= 60.0:
        log.warning("ANOMALI SENSOR: Nilai tidak masuk akal (Soil=%.1f%%, Temp=%.1f°C). Data diabaikan.", data.soil_moisture, data.temperature)
        return True

    last_soil = state.get("last_sensor_soil")
    if last_soil is not None:
        if abs(data.soil_moisture - float(last_soil)) > 30.0:
            log.warning("ANOMALI SENSOR: Perubahan drastis >30%% (%.1f%% -> %.1f%%). Data diabaikan.", float(last_soil), data.soil_moisture)
            return True

    elapsed = _elapsed_seconds_real(state.get("last_sensor_ts"))
    if elapsed > CFG.SENSOR_DEBOUNCE_SECONDS:
        return False
    if last_soil is None:
        return False
    return abs(data.soil_moisture - float(last_soil)) <= CFG.SENSOR_TOLERANCE


# ══════════════════════════════════════════════════════════════════════════════
# MESIN KEPUTUSAN AUTO
# ══════════════════════════════════════════════════════════════════════════════
def _evaluate_smart_watering(result, hour, minute, soil_moisture, air_humidity,
                              temperature, state, current_total_minutes):
    global _daily_safety

    current_date = datetime.now().date()
    if _daily_safety["date"] != current_date:
        _daily_safety["date"] = current_date
        _daily_safety["watering_count"] = 0
        _daily_safety["locked_out"] = False

    resp = {
        "action": None, "reason": "", "blocked_reason": None,
        "is_raining": False, "rain_score": 0,
        "hot_mode": temperature >= CFG.HOT_TEMP_THRESHOLD,
        "missed_session": bool(state.get("missed_session", False)),
        "decision_path": [],
    }

    if _daily_safety["locked_out"]:
        resp["blocked_reason"] = "Safety Lockout: Melebihi batas harian penyiraman maksimum (10x)."
        resp["decision_path"].append("SAFETY_LOCKOUT")
        return resp

    def _block(code, reason):
        resp["blocked_reason"] = reason
        resp["decision_path"].append(code)

    def _act_on(code, reason):
        resp["action"] = "on"; resp["reason"] = reason
        resp["decision_path"].append(code)

    def _act_off(code, reason):
        resp["action"] = "off"; resp["reason"] = reason
        resp["decision_path"].append(code)

    rain_score, rain_signals = _compute_rain_score(
        air_humidity=air_humidity, soil_moisture=soil_moisture,
        temperature=temperature, last_soil=state.get("last_soil_moisture"),
        last_temp=state.get("last_temperature"), pump_was_on=bool(state["pump_status"]),
    )
    is_raining, rain_reason = _update_rain_state(rain_score, rain_signals, state, current_total_minutes)
    resp["is_raining"] = is_raining
    resp["rain_score"] = rain_score

    # -- Dynamic Thresholds (Logika lebih pintar & adaptif) --
    dynamic_dry_on  = CFG.SOIL_DRY_ON
    dynamic_wet_off = CFG.SOIL_WET_OFF

    if resp["hot_mode"]:
        dynamic_dry_on  += 5.0  # Cuaca panas, mulai siram lebih awal
        dynamic_wet_off += 5.0  # Butuh lebih banyak air
        resp["decision_path"].append("T-HOT_ADJUST")
    elif temperature < 25.0 and air_humidity > 80.0:
        dynamic_dry_on  -= 5.0  # Cuaca dingin & lembab, tunda siram
        dynamic_wet_off -= 5.0
        resp["decision_path"].append("T-COOL_ADJUST")

    if state.get("missed_session"):
        dynamic_wet_off += 5.0  # Kompensasi sesi yang terlewat (hujan palsu)
        resp["decision_path"].append("T-MISSED_ADJUST")

    # Pastikan batas tidak melebihi 100% atau kurang dari batas minimal
    dynamic_wet_off = min(95.0, dynamic_wet_off)
    dynamic_dry_on  = max(CFG.CRITICAL_DRY + 5.0, dynamic_dry_on)

    in_window, window_label = _in_watering_window(hour)
    night_emergency = (not in_window and soil_moisture <= CFG.CRITICAL_DRY and not is_raining)
    if night_emergency:
        window_label = "malam-darurat"

    if state["pump_status"]:
        elapsed_sec = _elapsed_seconds_real(state.get("pump_start_ts"))
        max_sec     = 60 if night_emergency else (CFG.MAX_PUMP_DURATION_MINUTES * 60)

        if elapsed_sec >= max_sec:
            _daily_safety["last_pump_duration_sec"] = elapsed_sec
            _update_state(pump_status=False, last_watered_minute=current_total_minutes,
                          last_watered_ts=datetime.now().isoformat(),
                          pump_start_ts=None, pump_start_minute=None, missed_session=False)
            _act_off("A1", f"Auto-stop: batas maksimal ({elapsed_sec:.0f}s).")
            return resp

        if elapsed_sec < CFG.MIN_PUMP_DURATION_SECONDS:
            resp["reason"] = f"Pompa ON, warmup ({elapsed_sec:.0f}s < {CFG.MIN_PUMP_DURATION_SECONDS}s)."
            resp["decision_path"].append("A-warmup")
            return resp

        if soil_moisture >= dynamic_wet_off:
            _daily_safety["last_pump_duration_sec"] = elapsed_sec
            _update_state(pump_status=False, last_watered_minute=current_total_minutes,
                          last_watered_ts=datetime.now().isoformat(),
                          pump_start_ts=None, pump_start_minute=None, missed_session=False)
            _act_off("A2", f"Auto-stop: tanah cukup ({soil_moisture:.1f}% >= {dynamic_wet_off:.1f}%).")
            return resp

        if is_raining:
            _daily_safety["last_pump_duration_sec"] = elapsed_sec
            _update_state(pump_status=False, last_watered_minute=current_total_minutes,
                          last_watered_ts=datetime.now().isoformat(),
                          pump_start_ts=None, pump_start_minute=None, missed_session=False)
            _act_off("A3", f"Auto-stop: {rain_reason}. Hujan menggantikan siram.")
            return resp

        resp["reason"] = f"Pompa ON ({elapsed_sec:.0f}s/{max_sec:.0f}s). Tanah={soil_moisture:.1f}%."
        resp["decision_path"].append("A4-running")
        return resp

    has_missed = bool(state.get("missed_session", False))

    if night_emergency or (soil_moisture <= CFG.CRITICAL_DRY and not is_raining):
        now_ts = datetime.now().isoformat()
        _daily_safety["watering_count"] += 1
        if _daily_safety["watering_count"] >= 10:
            _daily_safety["locked_out"] = True
            
        _update_state(pump_status=True, pump_start_minute=current_total_minutes, pump_start_ts=now_ts)
        _act_on("B1", f"SIRAM DARURAT [{window_label}]: tanah {soil_moisture:.1f}% <= {CFG.CRITICAL_DRY}%.")
        return resp

    if not in_window:
        _block("B2", f"Di luar jam aman. WIT={hour:02d}:{minute:02d}.")
        return resp

    if is_raining:
        _block("B3", f"{rain_reason}. Ditunda.")
        return resp

    if soil_moisture >= dynamic_wet_off:
        if has_missed:
            _update_state(missed_session=False)
        _block("B4", f"Tanah sudah basah ({soil_moisture:.1f}%).")
        return resp

    effective_cooldown = CFG.POST_RAIN_COOLDOWN_MINUTES if has_missed else CFG.COOLDOWN_MINUTES
    if _daily_safety.get("last_pump_duration_sec", 999) < 120 and not has_missed:
        effective_cooldown = 15  # Cooldown adaptif
        resp["decision_path"].append("ADAPTIVE_COOLDOWN")
    elapsed_cd = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < effective_cooldown:
        _block("B5", f"Cooldown: sisa {effective_cooldown - elapsed_cd} mnt.")
        return resp

    elapsed_gap = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_gap < CFG.MIN_SESSION_GAP_MINUTES:
        _block("B6", f"Gap minimum belum tercapai ({elapsed_gap} mnt).")
        return resp

    if not result["needs_watering"]:
        _block("B7", f"KNN label='{result['label']}' ({result['confidence']}%).")
        return resp

    if resp["hot_mode"]:
        threshold, ctx = CFG.CONFIDENCE_HOT, "suhu panas"
    elif has_missed:
        threshold, ctx = CFG.CONFIDENCE_MISSED, "hutang siram"
    else:
        threshold, ctx = CFG.CONFIDENCE_NORMAL, "normal"

    if result["confidence"] < threshold:
        _block("B8", f"Confidence {result['confidence']}% < {threshold}% ({ctx}).")
        return resp

    if soil_moisture > dynamic_dry_on:
        _block("B9", f"Tanah {soil_moisture:.1f}% > batas on ({dynamic_dry_on:.1f}%).")
        return resp

    now_ts = datetime.now().isoformat()
    _daily_safety["watering_count"] += 1
    if _daily_safety["watering_count"] >= 10:
        _daily_safety["locked_out"] = True
        
    _update_state(pump_status=True, pump_start_minute=current_total_minutes, pump_start_ts=now_ts)
    _act_on("B10", (
        f"Siram [{window_label}]: KNN={result['label']} ({result['confidence']}%), "
        f"suhu={temperature:.1f}°C, tanah={soil_moisture:.1f}%."
    ))
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: Public (tanpa auth)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    """Health check publik — tidak butuh API key."""
    return {
        "status"      : "online",
        "message"     : "Siram Pintar API berjalan",
        "version"     : "6.0.0",
        "model_ready" : knn_model is not None,
        "auth"        : "required" if VALID_API_KEY else "disabled",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: Protected (butuh X-API-Key header)
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
        return {"db_status": "error", "detail": str(e)}


@app.get("/model-info", dependencies=[Depends(verify_api_key)])
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


@app.post("/sensor", dependencies=[Depends(verify_api_key)])
def receive_sensor(data: SensorData):
    with _sensor_lock:
        result    = classify(data.soil_moisture, data.temperature, data.air_humidity)
        state     = _get_state()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_id    = str(uuid.uuid4())

        hour, minute, _day, time_source = _resolve_time_wit(data.hour, data.minute, data.day)
        current_total_minutes = _total_minutes(hour, minute)

        if random.random() < 0.05:
            with get_db() as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute("DELETE FROM sensor_readings WHERE timestamp < NOW() - INTERVAL 14 DAY")
                    except Exception as e:
                        log.error("Gagal auto-prune database: %s", e)

        skip_eval    = _should_skip_sensor(data, state)
        
        # Mencegah spam database / bug ESP32 ngirim data berulang kali dengan cepat
        if skip_eval:
            elapsed_spam = _elapsed_seconds_real(state.get("last_sensor_ts"))
            if elapsed_spam < 2.0:
                log.debug("Spam filter: Request terlalu cepat (%.1fs), abaikan operasi DB.", elapsed_spam)
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

        if state["mode"] == "manual":
            pass
        elif state["mode"] == "auto" and not skip_eval:
            smart_eval   = _evaluate_smart_watering(
                result=result, hour=hour, minute=minute,
                soil_moisture=data.soil_moisture, air_humidity=data.air_humidity,
                temperature=data.temperature, state=state,
                current_total_minutes=current_total_minutes,
            )
            final_action = smart_eval.get("action")
        elif state["mode"] == "auto" and skip_eval:
            log.debug("Sensor debounce: skip evaluasi.")

        _update_state(
            last_label=result["label"], last_updated=timestamp,
            last_soil_moisture=data.soil_moisture, last_temperature=data.temperature,
            last_sensor_ts=datetime.now().isoformat(), last_sensor_soil=data.soil_moisture,
        )

        new_state          = _get_state()
        pump_status_logged = (final_action == "on") if final_action is not None else new_state["pump_status"]

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
                "is_raining"    : smart_eval.get("is_raining", False),
                "rain_score"    : smart_eval.get("rain_score", 0),
                "hot_mode"      : smart_eval.get("hot_mode", False),
                "missed_session": smart_eval.get("missed_session", False),
                "reason"        : smart_eval.get("reason", ""),
                "blocked_reason": smart_eval.get("blocked_reason"),
                "decision_path" : smart_eval.get("decision_path", []),
            } if state["mode"] == "auto" else None,
        }


@app.get("/status", dependencies=[Depends(verify_api_key)])
def get_status():
    state = _get_state()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1")
            latest = cur.fetchone()

    if latest:
        if isinstance(latest.get("probabilities"), str):
            latest["probabilities"] = json.loads(latest["probabilities"])
        latest["pump_status"]    = bool(latest["pump_status"])
        latest["needs_watering"] = bool(latest["needs_watering"])

    return {
        "pump_status"   : state["pump_status"],
        "mode"          : state["mode"],
        "last_label"    : state["last_label"],
        "last_updated"  : str(state["last_updated"]) if state["last_updated"] else None,
        "is_raining"    : state.get("rain_detected", False),
        "rain_score"    : state.get("rain_score", 0),
        "missed_session": state.get("missed_session", False),
        "watering_windows": {
            "morning": f"{CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59 WIT",
            "evening": f"{CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59 WIT",
        },
        "thresholds": {
            "soil_dry_on"  : CFG.SOIL_DRY_ON,
            "soil_wet_off" : CFG.SOIL_WET_OFF,
            "critical_dry" : CFG.CRITICAL_DRY,
        },
        "latest_data"   : latest,
    }


@app.get("/history", dependencies=[Depends(verify_api_key)])
def get_history(limit: int = Query(default=50, ge=1, le=500)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT %s", (limit,)
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


@app.post("/control", dependencies=[Depends(verify_api_key)])
def control_pump(cmd: ControlCommand):
    with _control_lock:
        action = (cmd.action or "").lower().strip()
        if action not in ("on", "off"):
            raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'.")

        mode = (cmd.mode or "manual").lower().strip()
        if mode not in ("auto", "manual"):
            mode = "manual"

        state   = _get_state()
        pump_on = action == "on"

        same_state  = (state["pump_status"] == pump_on)
        same_mode   = (state["mode"] == mode)

        if same_state and same_mode:
            # Cegah bug perintah duplikat mereset timer (pump_start_ts)
            return {
                "success"    : True,
                "debounced"  : True,
                "message"    : f"Status pompa dan mode tidak berubah, perintah duplikat diabaikan.",
                "pump_status": state["pump_status"],
                "mode"       : state["mode"],
                "timestamp"  : state.get("last_control_ts") or datetime.now().isoformat(),
            }

        now_ts        = datetime.now().isoformat()
        update_kwargs = {"mode": mode, "last_control_ts": now_ts}
        
        if not same_state:
            update_kwargs["pump_status"] = pump_on
            if not pump_on:
                update_kwargs["pump_start_ts"]      = None
                update_kwargs["pump_start_minute"]  = None
                update_kwargs["last_watered_ts"]    = now_ts
                current_min = _total_minutes(*_resolve_time_wit(None, None, None)[:2])
                update_kwargs["last_watered_minute"] = current_min
            else:
                update_kwargs["pump_start_ts"] = now_ts
                now_utc = datetime.utcnow()
                h_wit   = (now_utc.hour + 9) % 24
                update_kwargs["pump_start_minute"] = _total_minutes(h_wit, now_utc.minute)

        _update_state(**update_kwargs)
        new_state = _get_state()

        return {
            "success"    : True,
            "debounced"  : False,
            "pump_status": new_state["pump_status"],
            "mode"       : new_state["mode"],
            "timestamp"  : now_ts,
        }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: SensorData):
    return {
        "input" : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
        "result": classify(data.soil_moisture, data.temperature, data.air_humidity),
    }


@app.get("/config", dependencies=[Depends(verify_api_key)])
def get_config():
    return {
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
            "max_duration_min"  : CFG.MAX_PUMP_DURATION_MINUTES,
            "min_duration_sec"  : CFG.MIN_PUMP_DURATION_SECONDS,
            "cooldown_normal"   : CFG.COOLDOWN_MINUTES,
            "cooldown_post_rain": CFG.POST_RAIN_COOLDOWN_MINUTES,
        },
        "knn_confidence": {
            "normal"         : CFG.CONFIDENCE_NORMAL,
            "hot_weather"    : CFG.CONFIDENCE_HOT,
            "missed_session" : CFG.CONFIDENCE_MISSED,
            "hot_threshold"  : CFG.HOT_TEMP_THRESHOLD,
        },
    }


@app.post("/reset-rain", dependencies=[Depends(verify_api_key)])
def reset_rain():
    _update_state(
        rain_detected=False, rain_score=0, rain_confirm_count=0,
        rain_clear_count=0, rain_started_minute=None, missed_session=False,
    )
    return {"success": True, "message": "State hujan dan hutang siram di-reset."}