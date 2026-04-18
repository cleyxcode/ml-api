import os
import json
import uuid
import logging
import threading
import joblib
import numpy as np
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

import pymysql
import pymysql.cursors
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

# ── Global lock: mencegah race condition saat ESP32 kirim data cepat ──────────
_sensor_lock = threading.Lock()
_control_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════
class WateringConfig:
    # ── Jendela waktu aman (inklusif) ─────────────────────────────────────────
    MORNING_WINDOW = (5, 7)    # 05:00–07:59 WIT
    EVENING_WINDOW = (16, 18)  # 16:00–18:59 WIT

    # ── Soil hysteresis — mencegah on/off jitter ──────────────────────────────
    # Pompa nyala jika tanah <= SOIL_DRY_ON, mati jika >= SOIL_WET_OFF
    # Gap antara keduanya adalah zona aman (tidak bolak-balik)
    SOIL_DRY_ON   = 45.0   # % — mulai siram jika di bawah ini
    SOIL_WET_OFF  = 70.0   # % — stop siram jika sudah di atas ini
    CRITICAL_DRY  = 20.0   # % — darurat, bypass jam aman

    # ── Deteksi hujan multi-faktor (skor 0-100) ───────────────────────────────
    RAIN_SCORE_THRESHOLD   = 60    # Skor min untuk konfirmasi hujan
    RAIN_RH_HEAVY          = 92.0  # RH → skor +50 (hujan lebat)
    RAIN_RH_MODERATE       = 85.0  # RH → skor +30 (hujan sedang)
    RAIN_RH_LIGHT          = 78.0  # RH → skor +15 (gerimis/kabut tebal)
    RAIN_SOIL_RISE_HEAVY   = 8.0   # Kenaikan tanah → skor +35
    RAIN_SOIL_RISE_LIGHT   = 3.0   # Kenaikan tanah → skor +20
    RAIN_TEMP_DROP         = 3.0   # Penurunan suhu → skor +15
    RAIN_CLEAR_THRESHOLD   = 30    # Skor < ini → hujan selesai
    RAIN_CONFIRM_READINGS  = 2     # Berapa pembacaan berturut yg harus konfirmasi
    RAIN_CLEAR_READINGS    = 3     # Berapa pembacaan berturut yg harus clear

    # ── Cooldown ──────────────────────────────────────────────────────────────
    COOLDOWN_MINUTES           = 45
    POST_RAIN_COOLDOWN_MINUTES = 120  # Lebih lama, tanah masih basah
    MIN_SESSION_GAP_MINUTES    = 10   # Gap minimum antar sesi dalam 1 jendela

    # ── Durasi pompa ──────────────────────────────────────────────────────────
    MAX_PUMP_DURATION_MINUTES = 5    # Hard cap auto mode
    MIN_PUMP_DURATION_SECONDS = 30   # Pompa tidak mati < 30 detik (mencegah flip-flop)

    # ── Suhu ekstrem ──────────────────────────────────────────────────────────
    HOT_TEMP_THRESHOLD = 34.0

    # ── KNN Confidence threshold (dinamis) ────────────────────────────────────
    CONFIDENCE_NORMAL = 60.0
    CONFIDENCE_HOT    = 40.0   # Lebih permisif saat panas
    CONFIDENCE_MISSED = 48.0   # Lebih permisif saat ada hutang siram

    # ── Debounce /control (manual) ────────────────────────────────────────────
    # Perintah yang sama tidak diproses ulang jika pompa sudah dalam kondisi itu
    # dan belum lewat CONTROL_DEBOUNCE_SECONDS sejak perintah terakhir
    CONTROL_DEBOUNCE_SECONDS = 5

    # ── Sensor reading debounce ───────────────────────────────────────────────
    # Jika data sensor sama persis (dalam toleransi) & waktu < ini, skip evaluasi
    SENSOR_DEBOUNCE_SECONDS = 10
    SENSOR_TOLERANCE        = 1.0   # Toleransi nilai sensor dianggap "sama"


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
    """
    Prioritas: jam ESP32 (WIT via NTP GMT+9).
    Fallback: konversi UTC server → WIT.
    Return: (hour, minute, wday, source)
    """
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
    """Elapsed dengan rollover 24 jam."""
    if stored is None:
        return 999_999
    diff = current - int(stored)
    return diff if diff >= 0 else diff + 1440


def _elapsed_seconds_real(stored_ts_str) -> float:
    """Elapsed detik nyata dari timestamp ISO string."""
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
# HELPER: system_state — dengan locking di DB
# ══════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS = {
    "pump_status"          : False,
    "mode"                 : "auto",
    "last_label"           : None,
    "last_updated"         : None,
    "pump_start_ts"        : None,   # v6: timestamp real (bukan menit) untuk durasi pompa
    "pump_start_minute"    : None,
    "last_watered_minute"  : None,
    "last_watered_ts"      : None,   # v6: timestamp real untuk cooldown akurat
    "last_soil_moisture"   : None,
    "last_temperature"     : None,   # v6: untuk deteksi penurunan suhu (hujan)
    "missed_session"       : False,
    "rain_detected"        : False,
    "rain_score"           : 0,      # v6: rain confidence score 0-100
    "rain_confirm_count"   : 0,      # v6: consecutive readings confirm
    "rain_clear_count"     : 0,      # v6: consecutive readings clear
    "rain_started_minute"  : None,
    "last_control_ts"      : None,   # v6: debounce /control endpoint
    "last_sensor_ts"       : None,   # v6: debounce /sensor endpoint
    "last_sensor_soil"     : None,   # v6: nilai sensor terakhir untuk debounce
    "session_count_today"  : 0,      # v6: jumlah sesi siram hari ini
    "session_count_date"   : None,   # v6: tanggal untuk reset session count
}


def _get_state() -> dict:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_state WHERE id = 1")
            row = cur.fetchone()
    if not row:
        return dict(_STATE_DEFAULTS)

    # Normalisasi tipe boolean
    for bool_key in ("pump_status", "missed_session", "rain_detected"):
        row[bool_key] = bool(row.get(bool_key, False))

    # Normalisasi int
    for int_key in ("rain_score", "rain_confirm_count", "rain_clear_count", "session_count_today"):
        row[int_key] = int(row.get(int_key) or 0)

    # Isi default jika kolom belum ada (migrasi bertahap)
    for k, v in _STATE_DEFAULTS.items():
        if k not in row:
            row[k] = v

    return row


def _update_state(**kwargs):
    """Update kolom state secara atomik. Hanya kolom yang ada di DB yang di-update."""
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
# DETEKSI HUJAN — Multi-faktor Scoring (v6)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_rain_score(
    air_humidity   : float,
    soil_moisture  : float,
    temperature    : float,
    last_soil      ,
    last_temp      ,
    pump_was_on    : bool,
) -> tuple:
    """
    Hitung skor hujan 0-100 dari beberapa sinyal.
    Jika pompa baru mati, kenaikan tanah DIABAIKAN (false positive).

    Return: (score: int, signals: list[str])
    """
    score   = 0
    signals = []

    # ── Sinyal 1: RH udara ────────────────────────────────────────────────────
    if air_humidity >= CFG.RAIN_RH_HEAVY:
        score += 50; signals.append(f"RH={air_humidity:.0f}% (lebat)")
    elif air_humidity >= CFG.RAIN_RH_MODERATE:
        score += 30; signals.append(f"RH={air_humidity:.0f}% (sedang)")
    elif air_humidity >= CFG.RAIN_RH_LIGHT:
        score += 15; signals.append(f"RH={air_humidity:.0f}% (ringan)")

    # ── Sinyal 2: Kenaikan tanah (hanya jika bukan dari pompa) ───────────────
    if not pump_was_on and last_soil is not None:
        delta = soil_moisture - float(last_soil)
        if delta >= CFG.RAIN_SOIL_RISE_HEAVY:
            score += 35; signals.append(f"tanah +{delta:.1f}% (cepat)")
        elif delta >= CFG.RAIN_SOIL_RISE_LIGHT:
            score += 20; signals.append(f"tanah +{delta:.1f}% (perlahan)")

    # ── Sinyal 3: Penurunan suhu tiba-tiba ────────────────────────────────────
    if last_temp is not None:
        temp_drop = float(last_temp) - temperature
        if temp_drop >= CFG.RAIN_TEMP_DROP:
            score += 15; signals.append(f"suhu turun -{temp_drop:.1f}°C")

    return min(score, 100), signals


def _update_rain_state(
    score         : int,
    signals       : list,
    state         : dict,
    current_min   : int,
) -> tuple:
    """
    Konfirmasi/clear hujan berdasarkan consecutive readings.
    Return: (is_raining: bool, reason: str)
    """
    currently_raining = state["rain_detected"]
    confirm_count     = state["rain_confirm_count"]
    clear_count       = state["rain_clear_count"]
    rain_score_prev   = state["rain_score"]

    if score >= CFG.RAIN_SCORE_THRESHOLD:
        # Skor tinggi → tambah confirm count, reset clear count
        confirm_count += 1
        clear_count    = 0

        if not currently_raining and confirm_count >= CFG.RAIN_CONFIRM_READINGS:
            log.info("HUJAN DIKONFIRMASI: skor=%d, sinyal=%s", score, signals)
            _update_state(
                rain_detected       = True,
                rain_score          = score,
                rain_confirm_count  = confirm_count,
                rain_clear_count    = 0,
                rain_started_minute = current_min,
                missed_session      = True,
            )
            return True, f"Hujan dikonfirmasi (skor={score}, {', '.join(signals)})"
        elif currently_raining:
            _update_state(rain_score=score, rain_confirm_count=confirm_count, rain_clear_count=0)
            return True, f"Hujan berlanjut (skor={score})"
        else:
            _update_state(rain_score=score, rain_confirm_count=confirm_count, rain_clear_count=0)
            return False, f"Menunggu konfirmasi hujan ({confirm_count}/{CFG.RAIN_CONFIRM_READINGS}, skor={score})"

    elif score <= CFG.RAIN_CLEAR_THRESHOLD:
        # Skor rendah → tambah clear count
        clear_count   += 1
        confirm_count  = 0

        if currently_raining and clear_count >= CFG.RAIN_CLEAR_READINGS:
            log.info("HUJAN SELESAI: skor=%d", score)
            _update_state(
                rain_detected      = False,
                rain_score         = score,
                rain_confirm_count = 0,
                rain_clear_count   = clear_count,
                # missed_session tetap True sampai berhasil siram
            )
            return False, ""
        elif currently_raining:
            _update_state(rain_score=score, rain_confirm_count=0, rain_clear_count=clear_count)
            return True, f"Hujan mungkin selesai, tunggu konfirmasi ({clear_count}/{CFG.RAIN_CLEAR_READINGS})"
        else:
            _update_state(rain_score=score, rain_confirm_count=0, rain_clear_count=clear_count)
            return False, ""

    else:
        # Skor ambiguos (di antara threshold) — pertahankan state
        if currently_raining:
            _update_state(rain_score=score)
            return True, f"Hujan ambiguos (skor={score}), tetap aktif"
        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# CEK DEBOUNCE SENSOR
# ══════════════════════════════════════════════════════════════════════════════
def _should_skip_sensor(data: SensorData, state: dict) -> bool:
    """
    True jika pembacaan terlalu mirip dengan sebelumnya dan waktu terlalu dekat.
    Mencegah evaluasi berulang akibat ESP32 kirim terlalu cepat.
    """
    elapsed = _elapsed_seconds_real(state.get("last_sensor_ts"))
    if elapsed > CFG.SENSOR_DEBOUNCE_SECONDS:
        return False
    last_soil = state.get("last_sensor_soil")
    if last_soil is None:
        return False
    return abs(data.soil_moisture - float(last_soil)) <= CFG.SENSOR_TOLERANCE


# ══════════════════════════════════════════════════════════════════════════════
# MESIN KEPUTUSAN AUTO (v6 — State Machine)
# ══════════════════════════════════════════════════════════════════════════════
def _evaluate_smart_watering(
    result               : dict,
    hour                 : int,
    minute               : int,
    soil_moisture        : float,
    air_humidity         : float,
    temperature          : float,
    state                : dict,
    current_total_minutes: int,
) -> dict:
    """
    State Machine penyiraman cerdas.

    STATE A — Pompa ON:
      [A1] Auto-stop: durasi hard cap tercapai
      [A2] Auto-stop: tanah sudah melewati SOIL_WET_OFF (hysteresis)
      [A3] Auto-stop: hujan dikonfirmasi (pompa tidak berguna)
      [A4] Pompa berjalan normal → maintain

    STATE B — Pompa OFF:
      [B1] DARURAT: tanah <= CRITICAL_DRY dan tidak hujan → siram paksa
      [B2] Di luar jendela waktu aman → blokir
      [B3] Hujan aktif → blokir, catat hutang siram
      [B4] Tanah sudah basah (>= SOIL_WET_OFF) → blokir, reset hutang jika hujan yg mengisi
      [B5] Cooldown belum selesai → blokir
      [B6] Sesi terlalu mepet (MIN_SESSION_GAP) → blokir anti-jitter
      [B7] KNN: bukan label Kering → blokir
      [B8] Confidence KNN < threshold dinamis → blokir
      [B9] Tanah belum cukup kering (> SOIL_DRY_ON, hysteresis) → blokir
      [B10] Semua lulus → NYALAKAN POMPA
    """
    resp = {
        "action"         : None,
        "reason"         : "",
        "blocked_reason" : None,
        "is_raining"     : False,
        "rain_score"     : 0,
        "hot_mode"       : temperature >= CFG.HOT_TEMP_THRESHOLD,
        "missed_session" : bool(state.get("missed_session", False)),
        "decision_path"  : [],
    }

    def _block(code: str, reason: str):
        resp["blocked_reason"] = reason
        resp["decision_path"].append(code)

    def _act_on(code: str, reason: str):
        resp["action"] = "on"
        resp["reason"] = reason
        resp["decision_path"].append(code)

    def _act_off(code: str, reason: str):
        resp["action"] = "off"
        resp["reason"] = reason
        resp["decision_path"].append(code)

    # ── Deteksi hujan ─────────────────────────────────────────────────────────
    rain_score, rain_signals = _compute_rain_score(
        air_humidity  = air_humidity,
        soil_moisture = soil_moisture,
        temperature   = temperature,
        last_soil     = state.get("last_soil_moisture"),
        last_temp     = state.get("last_temperature"),
        pump_was_on   = bool(state["pump_status"]),   # Jika pompa baru nyala, abaikan kenaikan tanah
    )
    is_raining, rain_reason = _update_rain_state(
        rain_score, rain_signals, state, current_total_minutes
    )
    resp["is_raining"] = is_raining
    resp["rain_score"] = rain_score

    # ══════════════════════════════════════════════════════════════════════════
    # STATE A: Pompa sedang ON
    # ══════════════════════════════════════════════════════════════════════════
    if state["pump_status"]:

        # [A1] Hard cap durasi pompa (pakai timestamp real, bukan menit)
        elapsed_sec = _elapsed_seconds_real(state.get("pump_start_ts"))
        max_sec     = CFG.MAX_PUMP_DURATION_MINUTES * 60

        if elapsed_sec >= max_sec:
            _update_state(
                pump_status         = False,
                last_watered_minute = current_total_minutes,
                last_watered_ts     = datetime.now().isoformat(),
                pump_start_ts       = None,
                pump_start_minute   = None,
                missed_session      = False,
            )
            _act_off("A1", f"Auto-stop: batas {CFG.MAX_PUMP_DURATION_MINUTES} menit tercapai ({elapsed_sec:.0f}s).")
            log.info("AUTO-STOP [A1-durasi]: %s", resp["reason"])
            return resp

        # Jangan matikan pompa terlalu cepat (anti flip-flop)
        if elapsed_sec < CFG.MIN_PUMP_DURATION_SECONDS:
            resp["reason"] = f"Pompa ON, terlalu singkat untuk dievaluasi ({elapsed_sec:.0f}s < {CFG.MIN_PUMP_DURATION_SECONDS}s)."
            resp["decision_path"].append("A-warmup")
            return resp

        # [A2] Tanah sudah basah (hysteresis: matikan di WET_OFF, bukan DRY_ON)
        if soil_moisture >= CFG.SOIL_WET_OFF:
            _update_state(
                pump_status         = False,
                last_watered_minute = current_total_minutes,
                last_watered_ts     = datetime.now().isoformat(),
                pump_start_ts       = None,
                pump_start_minute   = None,
                missed_session      = False,
            )
            _act_off("A2", f"Auto-stop: tanah cukup ({soil_moisture:.1f}% >= {CFG.SOIL_WET_OFF}%).")
            log.info("AUTO-STOP [A2-basah]: %s", resp["reason"])
            return resp

        # [A3] Hujan saat pompa nyala → matikan, hujan gantikan siram
        if is_raining:
            _update_state(
                pump_status         = False,
                last_watered_minute = current_total_minutes,
                last_watered_ts     = datetime.now().isoformat(),
                pump_start_ts       = None,
                pump_start_minute   = None,
                missed_session      = False,
            )
            _act_off("A3", f"Auto-stop: {rain_reason}. Hujan menggantikan siram.")
            log.info("AUTO-STOP [A3-hujan]: %s", resp["reason"])
            return resp

        # [A4] Pompa berjalan normal
        resp["reason"] = (
            f"Pompa ON ({elapsed_sec:.0f}s / {max_sec:.0f}s maks). "
            f"Tanah={soil_moisture:.1f}%, target matikan >= {CFG.SOIL_WET_OFF}%."
        )
        resp["decision_path"].append("A4-running")
        return resp

    # ══════════════════════════════════════════════════════════════════════════
    # STATE B: Pompa OFF — evaluasi apakah perlu nyala
    # ══════════════════════════════════════════════════════════════════════════

    has_missed = bool(state.get("missed_session", False))

    # [B1] Kekeringan DARURAT — bypass jam aman
    if soil_moisture <= CFG.CRITICAL_DRY and not is_raining:
        log.warning("DARURAT KERING: %.1f%% — siram paksa.", soil_moisture)
        now_ts = datetime.now().isoformat()
        _update_state(
            pump_status       = True,
            pump_start_minute = current_total_minutes,
            pump_start_ts     = now_ts,
        )
        _act_on("B1", (
            f"SIRAM DARURAT: tanah sangat kering ({soil_moisture:.1f}% <= {CFG.CRITICAL_DRY}%). "
            f"Jam WIT {hour:02d}:{minute:02d} diabaikan."
        ))
        return resp

    # [B2] Di luar jendela waktu aman
    in_window, window_label = _in_watering_window(hour)
    if not in_window:
        _block("B2", (
            f"Di luar jam aman. WIT={hour:02d}:{minute:02d}. "
            f"Pagi {CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59 / "
            f"Sore {CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59."
        ))
        return resp

    # [B3] Hujan aktif
    if is_raining:
        _block("B3", f"{rain_reason}. Ditunda — hutang siram dicatat.")
        return resp

    # [B4] Tanah sudah basah (>= SOIL_WET_OFF)
    if soil_moisture >= CFG.SOIL_WET_OFF:
        if has_missed:
            _update_state(missed_session=False)
            log.info("Hutang siram di-reset: hujan sudah mengisi tanah (%.1f%%).", soil_moisture)
        _block("B4", f"Tanah sudah basah ({soil_moisture:.1f}% >= {CFG.SOIL_WET_OFF}%).")
        return resp

    # [B5] Cooldown antar sesi
    effective_cooldown = CFG.POST_RAIN_COOLDOWN_MINUTES if has_missed else CFG.COOLDOWN_MINUTES
    elapsed_cd         = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < effective_cooldown:
        remaining = effective_cooldown - elapsed_cd
        _block("B5", (
            f"Cooldown {'pasca-hujan' if has_missed else 'normal'}: "
            f"sisa {remaining} mnt (dari {effective_cooldown} mnt)."
        ))
        return resp

    # [B6] Gap minimum antar sesi (anti-jitter dalam 1 jendela)
    elapsed_gap = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_gap < CFG.MIN_SESSION_GAP_MINUTES:
        _block("B6", f"Terlalu cepat sejak sesi terakhir ({elapsed_gap} mnt < {CFG.MIN_SESSION_GAP_MINUTES} mnt).")
        return resp

    # [B7] KNN: label bukan Kering
    if not result["needs_watering"]:
        _block("B7", f"KNN label='{result['label']}' ({result['confidence']}%) — tidak perlu siram.")
        return resp

    # [B8] Confidence KNN dinamis
    if resp["hot_mode"]:
        threshold, ctx = CFG.CONFIDENCE_HOT, "suhu panas"
    elif has_missed:
        threshold, ctx = CFG.CONFIDENCE_MISSED, "hutang siram"
    else:
        threshold, ctx = CFG.CONFIDENCE_NORMAL, "normal"

    if result["confidence"] < threshold:
        _block("B8", f"Confidence {result['confidence']}% < {threshold}% (konteks: {ctx}).")
        return resp

    # [B9] Hysteresis: tanah belum cukup kering untuk mulai siram
    # Pompa baru boleh nyala jika tanah <= SOIL_DRY_ON
    # (berbeda dengan matinya di SOIL_WET_OFF — ini mencegah on/off jitter)
    if soil_moisture > CFG.SOIL_DRY_ON:
        _block("B9", (
            f"Tanah {soil_moisture:.1f}% > threshold siram {CFG.SOIL_DRY_ON}%. "
            f"Tunggu hingga <= {CFG.SOIL_DRY_ON}% untuk memulai sesi baru."
        ))
        return resp

    # [B10] Semua lulus → NYALAKAN POMPA
    now_ts = datetime.now().isoformat()
    _update_state(
        pump_status       = True,
        pump_start_minute = current_total_minutes,
        pump_start_ts     = now_ts,
    )
    _act_on("B10", (
        f"Siram [{window_label}]: KNN={result['label']} ({result['confidence']}%), "
        f"suhu={temperature:.1f}°C, tanah={soil_moisture:.1f}%, "
        f"{'HUTANG SIRAM TERBAYAR' if has_missed else 'sesi normal'}."
    ))
    log.info("POMPA ON [B10]: %s", resp["reason"])
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: Root
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status"      : "online",
        "message"     : "Siram Pintar API berjalan",
        "version"     : "6.0.0",
        "model_ready" : knn_model is not None,
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
# ENDPOINT: /sensor — logika utama dari ESP32
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/sensor")
def receive_sensor(data: SensorData):
    # Lock: cegah race condition jika ESP32 kirim request simultan
    with _sensor_lock:
        result    = classify(data.soil_moisture, data.temperature, data.air_humidity)
        state     = _get_state()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_id    = str(uuid.uuid4())

        hour, minute, _day, time_source = _resolve_time_wit(data.hour, data.minute, data.day)
        current_total_minutes = _total_minutes(hour, minute)

        # ── Sensor debounce: skip evaluasi jika data sama & terlalu cepat ────
        skip_eval = _should_skip_sensor(data, state)

        final_action = None
        smart_eval   = {}

        if state["mode"] == "manual":
            pass  # Manual: pompa hanya dari /control

        elif state["mode"] == "auto" and not skip_eval:
            smart_eval   = _evaluate_smart_watering(
                result                = result,
                hour                  = hour,
                minute                = minute,
                soil_moisture         = data.soil_moisture,
                air_humidity          = data.air_humidity,
                temperature           = data.temperature,
                state                 = state,
                current_total_minutes = current_total_minutes,
            )
            final_action = smart_eval.get("action")

        elif state["mode"] == "auto" and skip_eval:
            log.debug("Sensor debounce: data sama, skip evaluasi.")

        # ── Update state sensor terbaru ───────────────────────────────────────
        _update_state(
            last_label        = result["label"],
            last_updated      = timestamp,
            last_soil_moisture= data.soil_moisture,
            last_temperature  = data.temperature,
            last_sensor_ts    = datetime.now().isoformat(),
            last_sensor_soil  = data.soil_moisture,
        )

        # ── Logging ke DB sensor_readings ─────────────────────────────────────
        new_state          = _get_state()
        pump_status_logged = (
            (final_action == "on") if final_action is not None else new_state["pump_status"]
        )

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sensor_readings
                        (id, timestamp, soil_moisture, temperature, air_humidity,
                         label, confidence, needs_watering, description,
                         probabilities, pump_status, mode)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        row_id, timestamp,
                        data.soil_moisture, data.temperature, data.air_humidity,
                        result["label"], result["confidence"],
                        result["needs_watering"], result["description"],
                        json.dumps(result["probabilities"]),
                        pump_status_logged, state["mode"],
                    ),
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


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /status
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/status")
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


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /history
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/history")
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


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /control — dari Flutter (manual override)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/control")
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

        # ── Debounce: jika kondisi sudah sama, tolak perintah duplikat ────────
        same_state  = (state["pump_status"] == pump_on)
        elapsed_sec = _elapsed_seconds_real(state.get("last_control_ts"))

        if same_state and elapsed_sec < CFG.CONTROL_DEBOUNCE_SECONDS:
            log.info(
                "CONTROL DEBOUNCE: pompa sudah %s, perintah '%s' diabaikan (%.1fs < %ds).",
                "ON" if pump_on else "OFF", action, elapsed_sec, CFG.CONTROL_DEBOUNCE_SECONDS
            )
            return {
                "success"    : True,
                "debounced"  : True,
                "message"    : f"Pompa sudah {action.upper()}, perintah duplikat diabaikan.",
                "pump_status": state["pump_status"],
                "mode"       : state["mode"],
                "timestamp"  : datetime.now().isoformat(),
            }

        # ── Terapkan perintah ─────────────────────────────────────────────────
        now_ts        = datetime.now().isoformat()
        update_kwargs = {
            "pump_status"    : pump_on,
            "mode"           : mode,
            "last_control_ts": now_ts,
        }
        if not pump_on:
            # Saat matikan: catat waktu selesai, reset pump_start
            update_kwargs["pump_start_ts"]     = None
            update_kwargs["pump_start_minute"] = None
            update_kwargs["last_watered_ts"]   = now_ts
            current_min = _total_minutes(
                *_resolve_time_wit(None, None, None)[:2]
            )
            update_kwargs["last_watered_minute"] = current_min
        else:
            # Saat nyalakan: catat waktu mulai
            update_kwargs["pump_start_ts"] = now_ts
            now_utc = datetime.utcnow()
            h_wit   = (now_utc.hour + 9) % 24
            update_kwargs["pump_start_minute"] = _total_minutes(h_wit, now_utc.minute)

        _update_state(**update_kwargs)
        new_state = _get_state()

        log.info("CONTROL: pompa %s, mode=%s.", action.upper(), mode)

        return {
            "success"    : True,
            "debounced"  : False,
            "pump_status": new_state["pump_status"],
            "mode"       : new_state["mode"],
            "timestamp"  : now_ts,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /predict — uji KNN manual
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
# ENDPOINT: /config — lihat konfigurasi aktif (debugging)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/config")
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
            "score_to_confirm"  : CFG.RAIN_SCORE_THRESHOLD,
            "score_to_clear"    : CFG.RAIN_CLEAR_THRESHOLD,
            "confirm_readings"  : CFG.RAIN_CONFIRM_READINGS,
            "clear_readings"    : CFG.RAIN_CLEAR_READINGS,
            "rh_heavy"          : CFG.RAIN_RH_HEAVY,
            "rh_moderate"       : CFG.RAIN_RH_MODERATE,
            "rh_light"          : CFG.RAIN_RH_LIGHT,
        },
        "pump_control": {
            "max_duration_min"  : CFG.MAX_PUMP_DURATION_MINUTES,
            "min_duration_sec"  : CFG.MIN_PUMP_DURATION_SECONDS,
            "cooldown_normal"   : CFG.COOLDOWN_MINUTES,
            "cooldown_post_rain": CFG.POST_RAIN_COOLDOWN_MINUTES,
            "min_session_gap"   : CFG.MIN_SESSION_GAP_MINUTES,
        },
        "knn_confidence": {
            "normal"      : CFG.CONFIDENCE_NORMAL,
            "hot_weather" : CFG.CONFIDENCE_HOT,
            "missed_session": CFG.CONFIDENCE_MISSED,
            "hot_threshold": CFG.HOT_TEMP_THRESHOLD,
        },
        "debounce": {
            "control_sec": CFG.CONTROL_DEBOUNCE_SECONDS,
            "sensor_sec" : CFG.SENSOR_DEBOUNCE_SECONDS,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /reset-rain — reset state hujan manual (admin)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/reset-rain")
def reset_rain():
    _update_state(
        rain_detected      = False,
        rain_score         = 0,
        rain_confirm_count = 0,
        rain_clear_count   = 0,
        rain_started_minute= None,
        missed_session     = False,
    )
    return {"success": True, "message": "State hujan dan hutang siram di-reset."}