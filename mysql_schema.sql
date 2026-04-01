-- ============================================================
-- MySQL SCHEMA — Siram Pintar IoT
-- Jalankan di Hostinger → phpMyAdmin atau MySQL Remote
-- ============================================================

-- ── 1. SENSOR READINGS ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_readings (
    id              CHAR(36)        PRIMARY KEY DEFAULT (UUID()),
    timestamp       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    soil_moisture   FLOAT           NOT NULL,
    temperature     FLOAT           NOT NULL,
    air_humidity    FLOAT           NOT NULL,
    label           VARCHAR(20)     NOT NULL,
    confidence      FLOAT           NOT NULL,
    needs_watering  TINYINT(1)      NOT NULL DEFAULT 0,
    description     TEXT,
    probabilities   JSON,
    pump_status     TINYINT(1)      NOT NULL DEFAULT 0,
    mode            VARCHAR(20)     NOT NULL DEFAULT 'auto',
    INDEX idx_timestamp (timestamp DESC)
);

-- ── 2. SYSTEM STATE ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_state (
    id              INT             PRIMARY KEY DEFAULT 1,
    pump_status     TINYINT(1)      NOT NULL DEFAULT 0,
    mode            VARCHAR(20)     NOT NULL DEFAULT 'auto',
    last_label      VARCHAR(20),
    last_updated    DATETIME
);

INSERT INTO system_state (id, pump_status, mode)
VALUES (1, 0, 'auto')
ON DUPLICATE KEY UPDATE id = id;

-- ── 3. SCHEDULES ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schedules (
    id               CHAR(36)       PRIMARY KEY DEFAULT (UUID()),
    name             VARCHAR(100)   NOT NULL,
    time             VARCHAR(5)     NOT NULL,
    duration_minutes INT            NOT NULL DEFAULT 5,
    days             JSON           NOT NULL,
    enabled          TINYINT(1)     NOT NULL DEFAULT 1,
    last_triggered   DATETIME,
    created_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- ── 4. SCHEDULE LOGS ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schedule_logs (
    id               CHAR(36)       PRIMARY KEY DEFAULT (UUID()),
    schedule_id      CHAR(36),
    triggered_at     DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at     DATETIME,
    duration_minutes INT            NOT NULL,
    FOREIGN KEY (schedule_id) REFERENCES schedules(id) ON DELETE CASCADE,
    INDEX idx_schedule_id (schedule_id)
);
