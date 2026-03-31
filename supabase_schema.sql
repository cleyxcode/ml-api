-- ============================================================
-- SUPABASE SCHEMA — Siram Pintar IoT
-- Jalankan file ini di Supabase → SQL Editor
-- ============================================================


-- ── 1. SENSOR READINGS ───────────────────────────────────────
-- Menyimpan setiap pembacaan sensor dari ESP32 beserta
-- hasil klasifikasi KNN dan status pompa saat itu.
-- Menggantikan: history (deque di memory)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_readings (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Data sensor
    soil_moisture   FLOAT       NOT NULL CHECK (soil_moisture  BETWEEN 0 AND 100),
    temperature     FLOAT       NOT NULL CHECK (temperature    BETWEEN 0 AND 60),
    air_humidity    FLOAT       NOT NULL CHECK (air_humidity   BETWEEN 0 AND 100),

    -- Hasil KNN
    label           TEXT        NOT NULL,   -- 'Kering' | 'Lembab' | 'Basah'
    confidence      FLOAT       NOT NULL,   -- 0–100
    needs_watering  BOOLEAN     NOT NULL DEFAULT FALSE,
    description     TEXT,
    probabilities   JSONB,                  -- { "Kering": 92.5, "Lembab": 5.0, "Basah": 2.5 }

    -- Status sistem saat itu
    pump_status     BOOLEAN     NOT NULL DEFAULT FALSE,
    mode            TEXT        NOT NULL DEFAULT 'auto'  -- 'auto' | 'manual' | 'schedule'
);

-- Index untuk query riwayat terbaru (sering dipakai di GET /history)
CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp
    ON sensor_readings (timestamp DESC);


-- ── 2. SYSTEM STATE ──────────────────────────────────────────
-- Satu baris tunggal — menyimpan status sistem terkini.
-- Diperbarui setiap kali ada data sensor masuk atau kontrol manual.
-- Menggantikan: system_state (dict di memory)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_state (
    id              INT         PRIMARY KEY DEFAULT 1,  -- selalu 1 baris
    pump_status     BOOLEAN     NOT NULL DEFAULT FALSE,
    mode            TEXT        NOT NULL DEFAULT 'auto',
    last_label      TEXT,
    last_updated    TIMESTAMPTZ,

    CONSTRAINT system_state_single_row CHECK (id = 1)
);

-- Insert baris awal (hanya sekali)
INSERT INTO system_state (id, pump_status, mode)
VALUES (1, FALSE, 'auto')
ON CONFLICT (id) DO NOTHING;


-- ── 3. SCHEDULES ─────────────────────────────────────────────
-- Jadwal penyiraman otomatis yang dibuat user lewat Flutter.
-- Menggantikan: schedules (list of dict di memory)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schedules (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name             TEXT        NOT NULL,          -- 'Pagi', 'Sore', dll
    time             TEXT        NOT NULL,          -- format 'HH:MM', mis: '06:00'
    duration_minutes INT         NOT NULL DEFAULT 5 CHECK (duration_minutes BETWEEN 1 AND 60),
    days             TEXT[]      NOT NULL DEFAULT '{"senin","selasa","rabu","kamis","jumat","sabtu","minggu"}',
                                                    -- array hari aktif
    enabled          BOOLEAN     NOT NULL DEFAULT TRUE,
    last_triggered   TIMESTAMPTZ,                   -- kapan terakhir dijalankan
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Auto-update updated_at saat row diubah
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_schedules_updated_at
    BEFORE UPDATE ON schedules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();


-- ── 4. SCHEDULE LOGS ─────────────────────────────────────────
-- Riwayat kapan jadwal penyiraman dijalankan dan berapa lama.
-- Berguna untuk laporan / audit log.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schedule_logs (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_id      UUID        REFERENCES schedules(id) ON DELETE CASCADE,
    triggered_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at     TIMESTAMPTZ,
    duration_minutes INT         NOT NULL
);

-- Index untuk query log per jadwal
CREATE INDEX IF NOT EXISTS idx_schedule_logs_schedule_id
    ON schedule_logs (schedule_id, triggered_at DESC);


-- ============================================================
-- RINGKASAN TABEL
-- ┌─────────────────┬──────────────────────────────────────────┐
-- │ Tabel           │ Fungsi                                   │
-- ├─────────────────┼──────────────────────────────────────────┤
-- │ sensor_readings │ Riwayat data sensor + hasil KNN          │
-- │ system_state    │ Status pompa & mode saat ini (1 baris)   │
-- │ schedules       │ Jadwal penyiraman terjadwal              │
-- │ schedule_logs   │ Log kapan jadwal dijalankan              │
-- └─────────────────┴──────────────────────────────────────────┘
-- ============================================================
