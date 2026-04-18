import os
import json
import uuid
import logging
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

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI LOGIKA PENYIRAMAN CERDAS
# Semua nilai di sini dapat diubah tanpa menyentuh logika utama.
# ══════════════════════════════════════════════════════════════════════════════
class WateringConfig:
    # Jendela waktu aman — jam WIT (dari ESP32 via NTP, bukan server)
    # (jam_mulai_inklusif, jam_selesai_inklusif)
    MORNING_WINDOW  = (5, 7)    # 05:00–07:59 WIT
    EVENING_WINDOW  = (16, 18)  # 16:00–18:59 WIT

    # Deteksi hujan dua lapis:
    #   Sinyal 1 → RH udara >= threshold ini
    #   Sinyal 2 → kelembapan tanah NAIK sendiri >= threshold ini
    # Keduanya harus terpenuhi agar dikonfirmasi hujan (bukan sekadar lembap/kabut)
    RAIN_HUMIDITY_THRESHOLD  = 83.0  # RH udara (%)
    RAIN_SOIL_RISE_THRESHOLD = 5.0   # Kenaikan tanah min (%) untuk konfirmasi

    # Cooldown normal antar sesi penyiraman
    COOLDOWN_MINUTES = 45

    # Cooldown khusus setelah hujan (tanah masih basah, tunggu lebih lama)
    POST_RAIN_COOLDOWN_MINUTES = 90

    # Durasi maksimum pompa menyala di mode AUTO sebelum auto-stop
    MAX_PUMP_DURATION_MINUTES = 3

    # Suhu ekstrem — confidence KNN diturunkan agar lebih agresif siram
    HOT_TEMP_THRESHOLD = 34.0

    # Threshold confidence KNN (dinamis per konteks)
    CONFIDENCE_NORMAL  = 62.0  # Kondisi biasa
    CONFIDENCE_HOT     = 42.0  # Saat suhu panas ekstrem
    CONFIDENCE_MISSED  = 50.0  # Saat ada hutang siram (lebih permisif)

    # Kekeringan kritis — siram paksa walau di luar jam aman
    CRITICAL_DRY_SOIL = 20.0   # % kelembapan tanah

    # Batas atas tanah basah — jangan siram jika sudah di atas ini
    WET_SOIL_THRESHOLD = 70.0  # % kelembapan tanah

CFG = WateringConfig()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API",
    description="Sistem Penyiraman Tanaman IoT dengan KNN + Logika Cuaca Adaptif",
    version="5.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

knn_model  = None
scaler     = None
model_meta: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
@contextmanager
def get_db():
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15, ssl={"ssl": {}},
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
        knn_model = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                model_meta = json.load(f)
        else:
            model_meta = {"best_k": "?", "accuracy": "?"}
        log.info("Model KNN dimuat. K=%s, Akurasi=%s%%", model_meta.get("best_k"), model_meta.get("accuracy"))
    except Exception as exc:
        log.error("Gagal memuat model: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
class SensorData(BaseModel):
    soil_moisture : float = Field(..., ge=0, le=100)
    temperature   : float = Field(..., ge=0, le=60)
    air_humidity  : float = Field(..., ge=0, le=100)
    # Waktu lokal WIT dari ESP32 (sudah disinkron NTP GMT+9)
    # Prioritas utama. Jika tidak ada → fallback konversi waktu server ke WIT
    hour   : Optional[int] = Field(default=None, ge=0, le=23)
    minute : Optional[int] = Field(default=None, ge=0, le=59)
    day    : Optional[int] = Field(default=None, ge=0, le=6)


class ControlCommand(BaseModel):
    action : str           = Field(..., description="'on' atau 'off'")
    mode   : Optional[str] = Field(default="manual")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: waktu WIT
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_time_wit(
    hour: Optional[int], minute: Optional[int], day: Optional[int]
) -> tuple:
    """
    Prioritas: jam ESP32 (WIT via NTP).
    Fallback darurat: konversi UTC server → WIT (UTC+9).
    Return: (hour_wit, minute_wit, wday_0sun, source)
    """
    if hour is not None and minute is not None and day is not None:
        return hour, minute, day, "esp32"

    now_utc    = datetime.utcnow()
    wit_hour   = (now_utc.hour + 9) % 24
    wit_minute = now_utc.minute
    # Python weekday() Mon=0..Sun=6 → konversi ke Sun=0..Sat=6 (tm_wday)
    wit_wday   = (now_utc.weekday() + 1) % 7
    log.warning("Fallback ke waktu server WIT: %02d:%02d", wit_hour, wit_minute)
    return wit_hour, wit_minute, wit_wday, "server_fallback"


def _total_minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute


def _elapsed_minutes(current_total: int, stored_total) -> int:
    """Selisih menit dengan penanganan rollover 24 jam."""
    if stored_total is None:
        return 999_999
    diff = current_total - int(stored_total)
    return diff if diff >= 0 else diff + 1440


def _in_watering_window(hour: int) -> tuple:
    """Cek apakah jam WIT masuk jendela aman penyiraman."""
    if CFG.MORNING_WINDOW[0] <= hour <= CFG.MORNING_WINDOW[1]:
        return True, "pagi"
    if CFG.EVENING_WINDOW[0] <= hour <= CFG.EVENING_WINDOW[1]:
        return True, "sore"
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: system_state
# ══════════════════════════════════════════════════════════════════════════════
def _get_state() -> dict:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_state WHERE id = 1")
            row = cur.fetchone()
    if row:
        row["pump_status"]    = bool(row["pump_status"])
        row["missed_session"] = bool(row.get("missed_session", False))
        row["rain_detected"]  = bool(row.get("rain_detected", False))
        return row
    return {
        "pump_status"        : False,
        "mode"               : "auto",
        "last_label"         : None,
        "last_updated"       : None,
        "pump_start_minute"  : None,
        "last_watered_minute": None,
        "last_soil_moisture" : None,   # Untuk deteksi kenaikan tanah (hujan)
        "missed_session"     : False,  # Flag hutang siram akibat hujan
        "rain_detected"      : False,  # Flag hujan aktif saat ini
        "rain_started_minute": None,
    }


def _update_state(**kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k} = %s" for k in kwargs)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE system_state SET {sets} WHERE id = 1", list(kwargs.values()))


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: KNN classify
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
# DETEKSI HUJAN — logika dua lapis
# ══════════════════════════════════════════════════════════════════════════════
def _detect_rain(
    air_humidity: float,
    soil_moisture: float,
    last_soil_moisture,
    current_total_minutes: int,
    state: dict,
) -> tuple:
    """
    Hujan DIKONFIRMASI hanya jika DUA sinyal muncul bersamaan:
      1. RH udara >= 83% (udara jenuh)
      2. Kelembapan tanah NAIK sendiri >= 5% (bukan dari pompa)

    Mencegah false positive dari kabut/lembap biasa yang tidak ada airnya.

    Hujan SELESAI jika: RH < 75% DAN tanah tidak lagi naik.

    Return: (is_raining: bool, reason: str)
    """
    rh_high     = air_humidity >= CFG.RAIN_HUMIDITY_THRESHOLD
    soil_rising = False

    if last_soil_moisture is not None:
        delta       = soil_moisture - float(last_soil_moisture)
        soil_rising = delta >= CFG.RAIN_SOIL_RISE_THRESHOLD

    # Konfirmasi hujan baru (dua sinyal terpenuhi)
    if rh_high and soil_rising:
        if not state.get("rain_detected"):
            delta_val = soil_moisture - float(last_soil_moisture)
            log.info("Hujan terdeteksi: RH=%.1f%%, tanah naik +%.1f%%", air_humidity, delta_val)
            _update_state(
                rain_detected=True,
                rain_started_minute=current_total_minutes,
                missed_session=True,    # Catat hutang siram
            )
        return True, f"Hujan aktif (RH={air_humidity:.0f}%, tanah naik)"

    # RH masih tinggi tapi tanah tidak naik → kabut/lembap biasa atau
    # hujan sudah reda tapi udara belum kering
    if rh_high and not soil_rising and state.get("rain_detected"):
        return True, f"Pasca-hujan, RH masih tinggi ({air_humidity:.0f}%)"

    # Hujan benar-benar selesai
    if state.get("rain_detected") and air_humidity < 75.0 and not soil_rising:
        log.info("Hujan selesai. RH turun ke %.1f%%.", air_humidity)
        _update_state(rain_detected=False)
        # missed_session tetap True — akan direset setelah berhasil menyiram

    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# MESIN KEPUTUSAN MODE AUTO
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
    Urutan evaluasi (tiap langkah bisa mengembalikan hasil lebih awal):

    Jika pompa ON:
      [A] Auto-stop: durasi maksimum tercapai
      [B] Auto-stop: tanah sudah basah
      [C] Auto-stop: hujan tiba-tiba

    Jika pompa OFF:
      [1] Kekeringan kritis → siram paksa (bypass jam aman)
      [2] Di luar jam aman WIT → blokir
      [3] Hujan aktif → blokir + catat hutang siram
      [4] Tanah sudah basah → blokir (reset hutang jika ada)
      [5] Cooldown (normal / pasca-hujan) → blokir
      [6] KNN: label bukan Kering → blokir
      [7] Confidence < threshold dinamis → blokir
      [8] Semua lulus → NYALAKAN POMPA
    """
    response = {
        "action"        : None,
        "reason"        : "",
        "blocked_reason": None,
        "is_raining"    : False,
        "hot_mode"      : temperature >= CFG.HOT_TEMP_THRESHOLD,
        "missed_session": bool(state.get("missed_session", False)),
    }

    last_soil = state.get("last_soil_moisture")

    # Selalu jalankan deteksi hujan setiap pembacaan
    is_raining, rain_reason = _detect_rain(
        air_humidity, soil_moisture, last_soil, current_total_minutes, state
    )
    response["is_raining"] = is_raining

    # ── Pompa sedang ON ───────────────────────────────────────────────────────
    if state["pump_status"]:
        elapsed_run = _elapsed_minutes(current_total_minutes, state.get("pump_start_minute"))

        # [A] Durasi maksimum
        if elapsed_run >= CFG.MAX_PUMP_DURATION_MINUTES:
            _update_state(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                pump_start_minute=None,
                missed_session=False,
            )
            response["action"] = "off"
            response["reason"] = f"Auto-stop: batas {CFG.MAX_PUMP_DURATION_MINUTES} menit tercapai."
            log.info("AUTO-STOP [durasi]: %s", response["reason"])
            return response

        # [B] Tanah sudah cukup basah
        if soil_moisture >= CFG.WET_SOIL_THRESHOLD:
            _update_state(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                pump_start_minute=None,
                missed_session=False,
            )
            response["action"] = "off"
            response["reason"] = f"Auto-stop: tanah basah ({soil_moisture:.1f}% >= {CFG.WET_SOIL_THRESHOLD}%)."
            log.info("AUTO-STOP [tanah basah]: %s", response["reason"])
            return response

        # [C] Hujan tiba-tiba saat pompa nyala
        if is_raining:
            _update_state(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                pump_start_minute=None,
                missed_session=False,   # Hujan menggantikan siram
            )
            response["action"] = "off"
            response["reason"] = f"Auto-stop: {rain_reason}."
            log.info("AUTO-STOP [hujan]: %s", response["reason"])
            return response

        # Pompa masih berjalan normal
        response["reason"] = f"Pompa ON ({elapsed_run:.0f}/{CFG.MAX_PUMP_DURATION_MINUTES} mnt). Tanah={soil_moisture:.1f}%."
        return response

    # ── Pompa OFF — evaluasi apakah perlu dinyalakan ──────────────────────────

    # [1] Kekeringan kritis — bypass jam aman
    if soil_moisture <= CFG.CRITICAL_DRY_SOIL and not is_raining:
        log.warning("DARURAT KERING: %.1f%% — siram paksa.", soil_moisture)
        _update_state(pump_status=True, pump_start_minute=current_total_minutes)
        response["action"] = "on"
        response["reason"] = (
            f"SIRAM DARURAT: tanah sangat kering ({soil_moisture:.1f}% <= {CFG.CRITICAL_DRY_SOIL}%). "
            f"Jam WIT {hour:02d}:{minute:02d} diabaikan."
        )
        return response

    # [2] Cek jendela jam aman WIT dari ESP32
    in_window, window_label = _in_watering_window(hour)
    if not in_window:
        response["blocked_reason"] = (
            f"Di luar jam aman. Jam WIT ESP32: {hour:02d}:{minute:02d}. "
            f"Pagi {CFG.MORNING_WINDOW[0]:02d}:00-{CFG.MORNING_WINDOW[1]:02d}:59 / "
            f"Sore {CFG.EVENING_WINDOW[0]:02d}:00-{CFG.EVENING_WINDOW[1]:02d}:59."
        )
        return response

    # [3] Sedang hujan → blokir, hutang siram sudah dicatat di _detect_rain
    if is_raining:
        response["blocked_reason"] = f"{rain_reason}. Penyiraman ditunda — hutang siram dicatat."
        return response

    # [4] Tanah sudah basah
    if soil_moisture >= CFG.WET_SOIL_THRESHOLD:
        if state.get("missed_session"):
            _update_state(missed_session=False)
            log.info("Hutang siram di-reset: tanah sudah basah (%.1f%%) — hujan mengisi.", soil_moisture)
        response["blocked_reason"] = f"Tanah sudah basah ({soil_moisture:.1f}% >= {CFG.WET_SOIL_THRESHOLD}%)."
        return response

    # [5] Cooldown — lebih panjang jika pasca-hujan
    has_missed         = bool(state.get("missed_session", False))
    effective_cooldown = CFG.POST_RAIN_COOLDOWN_MINUTES if has_missed else CFG.COOLDOWN_MINUTES
    elapsed_cd         = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < effective_cooldown:
        remaining = effective_cooldown - elapsed_cd
        response["blocked_reason"] = (
            f"Cooldown {'pasca-hujan' if has_missed else 'normal'}: "
            f"sisa {remaining} menit (total {effective_cooldown} mnt)."
        )
        return response

    # [6] KNN: label bukan Kering
    if not result["needs_watering"]:
        response["blocked_reason"] = (
            f"KNN: label={result['label']}, conf={result['confidence']}% — tidak kering."
        )
        return response

    # [7] Threshold confidence dinamis
    if response["hot_mode"]:
        threshold, ctx = CFG.CONFIDENCE_HOT, "suhu panas"
    elif has_missed:
        threshold, ctx = CFG.CONFIDENCE_MISSED, "hutang siram"
    else:
        threshold, ctx = CFG.CONFIDENCE_NORMAL, "normal"

    if result["confidence"] < threshold:
        response["blocked_reason"] = (
            f"Confidence {result['confidence']}% < {threshold}% (konteks: {ctx})."
        )
        return response

    # [8] Semua kondisi lulus → NYALAKAN POMPA
    _update_state(pump_status=True, pump_start_minute=current_total_minutes)
    response["action"] = "on"
    response["reason"] = (
        f"Siram Pintar ON [{window_label}]: KNN={result['label']} ({result['confidence']}%), "
        f"suhu={temperature:.1f}C, tanah={soil_moisture:.1f}%, "
        f"{'HUTANG SIRAM TERBAYAR' if has_missed else 'sesi normal'}."
    )
    log.info("POMPA ON: %s", response["reason"])
    return response


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS UMUM
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "online", "message": "Siram Pintar API berjalan", "version": "5.0.0", "model_ready": knn_model is not None}


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

    # Selalu gunakan jam WIT dari ESP32 — bukan jam server
    hour, minute, _day, time_source = _resolve_time_wit(data.hour, data.minute, data.day)
    current_total_minutes = _total_minutes(hour, minute)

    final_action = None
    smart_eval   = {}

    if state["mode"] == "manual":
        pass   # Manual: pompa hanya merespons POST /control

    elif state["mode"] == "auto":
        smart_eval   = _evaluate_smart_watering(
            result=result,
            hour=hour,
            minute=minute,
            soil_moisture=data.soil_moisture,
            air_humidity=data.air_humidity,
            temperature=data.temperature,
            state=state,
            current_total_minutes=current_total_minutes,
        )
        final_action = smart_eval.get("action")

    # Simpan soil_moisture saat ini untuk perbandingan pembacaan berikutnya
    # (dibutuhkan oleh detektor hujan — apakah tanah naik sendiri?)
    _update_state(
        last_label=result["label"],
        last_updated=timestamp,
        last_soil_moisture=data.soil_moisture,
    )

    pump_status_logged = (
        (final_action == "on") if final_action is not None else state["pump_status"]
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
                    result["description"], json.dumps(result["probabilities"]),
                    pump_status_logged, state["mode"],
                ),
            )

    new_state = _get_state()

    return {
        "received"      : True,
        "timestamp"     : timestamp,
        "device_time"   : f"{hour:02d}:{minute:02d}",
        "time_source"   : time_source,   # "esp32" atau "server_fallback"
        "sensor"        : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
        "classification": result,
        "pump_status"   : new_state["pump_status"],
        "pump_action"   : final_action,
        "mode"          : new_state["mode"],
        # Info debugging & dashboard Flutter
        "auto_info": {
            "is_raining"    : smart_eval.get("is_raining", False),
            "hot_mode"      : smart_eval.get("hot_mode", False),
            "missed_session": smart_eval.get("missed_session", False),
            "reason"        : smart_eval.get("reason", ""),
            "blocked_reason": smart_eval.get("blocked_reason"),
        } if state["mode"] == "auto" else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT STATUS & HISTORY
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
        "missed_session": state.get("missed_session", False),
        "latest_data"   : latest,
    }


@app.get("/history")
def get_history(limit: int = Query(default=50, ge=1, le=500)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT %s", (limit,))
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
    if mode not in ("auto", "manual"):
        mode = "manual"
    pump_on       = action == "on"
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
# ENDPOINT PREDICT (uji manual KNN)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/predict")
def predict(data: SensorData):
    return {
        "input" : {"soil_moisture": data.soil_moisture, "temperature": data.temperature, "air_humidity": data.air_humidity},
        "result": classify(data.soil_moisture, data.temperature, data.air_humidity),
    }