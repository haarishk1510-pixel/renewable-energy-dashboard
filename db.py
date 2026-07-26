import sqlite3
import os
import json
import pandas as pd
from datetime import datetime

DB_NAME = "renewable_admin.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT NOT NULL,
        temperature REAL NOT NULL,
        irradiance REAL DEFAULT 800.0,
        hour INTEGER NOT NULL,
        model TEXT NOT NULL,
        prediction REAL NOT NULL,
        status TEXT DEFAULT 'RECORDED'
    )
    """)

    # 2. System Settings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    # 3. System Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,
        component TEXT NOT NULL,
        message TEXT NOT NULL
    )
    """)

    # 4. Telemetry Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        voltage REAL NOT NULL,
        current REAL NOT NULL,
        frequency REAL NOT NULL,
        battery_soc REAL NOT NULL,
        solar_yield_kwh REAL NOT NULL
    )
    """)

    # Seed Default System Settings
    default_settings = {
        "grid_status": "ONLINE",
        "dispatch_mode": "AUTO",
        "active_model": "solar_model.pkl",
        "max_power_limit": "1000",
        "emergency_cutoff": "OFF",
        "auto_tuning": "ENABLED"
    }

    now = datetime.now().isoformat()
    for key, val in default_settings.items():
        cursor.execute("""
        INSERT OR IGNORE INTO system_settings (key, value, updated_at)
        VALUES (?, ?, ?)
        """, (key, val, now))

    conn.commit()
    conn.close()

    # Log DB initialization
    log_event("INFO", "DATABASE", "SQLite Database initialized successfully")
    migrate_csv_history_if_needed()
    seed_telemetry_if_empty()

def log_event(level, component, message):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO system_logs (timestamp, level, component, message)
        VALUES (?, ?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), level, component, message))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging event: {e}")

def get_system_settings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value, updated_at FROM system_settings")
    rows = cursor.fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}

def update_system_setting(key, value):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO system_settings (key, value, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
    """, (key, str(value), now))
    conn.commit()
    conn.close()
    log_event("INFO", "CONTROLLER", f"System setting updated: {key} = {value}")

def insert_prediction(time_str, temp, irradiance, hour, model_name, prediction, status="RECORDED"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (time, temperature, irradiance, hour, model, prediction, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (time_str, temp, irradiance, hour, model_name, prediction, status))
    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()
    log_event("INFO", "PREDICTOR", f"Saved prediction #{prediction_id}: {prediction} kWh ({model_name})")
    return prediction_id

def fetch_predictions(search=None, limit=100, offset=0):
    conn = get_connection()
    cursor = conn.cursor()

    if search:
        query = """
        SELECT * FROM predictions 
        WHERE time LIKE ? OR model LIKE ? OR status LIKE ?
        ORDER BY id DESC LIMIT ? OFFSET ?
        """
        search_pattern = f"%{search}%"
        cursor.execute(query, (search_pattern, search_pattern, search_pattern, limit, offset))
    else:
        query = "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?"
        cursor.execute(query, (limit, offset))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_prediction_count(search=None):
    conn = get_connection()
    cursor = conn.cursor()
    if search:
        query = "SELECT COUNT(*) FROM predictions WHERE time LIKE ? OR model LIKE ? OR status LIKE ?"
        search_pattern = f"%{search}%"
        cursor.execute(query, (search_pattern, search_pattern, search_pattern))
    else:
        query = "SELECT COUNT(*) FROM predictions"
        cursor.execute(query)
    count = cursor.fetchone()[0]
    conn.close()
    return count

def clear_predictions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    log_event("WARNING", "DATABASE", "All prediction records cleared by Admin")

def fetch_logs(limit=50):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM system_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def fetch_telemetry(limit=20):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def insert_telemetry(voltage, current, frequency, battery_soc, solar_yield_kwh):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO telemetry (timestamp, voltage, current, frequency, battery_soc, solar_yield_kwh)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (now_str, voltage, current, frequency, battery_soc, solar_yield_kwh))
    conn.commit()
    conn.close()

def migrate_csv_history_if_needed():
    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        try:
            df = pd.read_csv(history_file)
            conn = get_connection()
            cursor = conn.cursor()

            # Check if predictions table is empty
            cursor.execute("SELECT COUNT(*) FROM predictions")
            if cursor.fetchone()[0] == 0:
                for _, row in df.iterrows():
                    time_val = str(row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    temp_val = float(row.get("temperature", 25.0))
                    hour_val = int(row.get("hour", 12))
                    pred_val = float(row.get("prediction", 0.0))
                    cursor.execute("""
                    INSERT INTO predictions (time, temperature, irradiance, hour, model, prediction, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (time_val, temp_val, 800.0, hour_val, "solar_model.pkl", pred_val, "MIGRATED_CSV"))
                conn.commit()
                log_event("INFO", "MIGRATION", f"Migrated {len(df)} records from CSV to SQLite DB")
            conn.close()
        except Exception as e:
            log_event("ERROR", "MIGRATION", f"Failed to migrate CSV: {e}")

def seed_telemetry_if_empty():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM telemetry")
    if cursor.fetchone()[0] == 0:
        import random
        from datetime import datetime, timedelta
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(24):
            t_str = (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:00:00")
            voltage = round(220.0 + random.uniform(-5.0, 5.0), 2)
            current = round(15.0 + random.uniform(-3.0, 5.0), 2)
            freq = round(50.0 + random.uniform(-0.2, 0.2), 2)
            battery = round(60.0 + (i % 12) * 3.0, 1)
            yield_kwh = round(120.0 + (12 - abs(12 - i)) * 15.5 + random.uniform(-5, 5), 2)
            cursor.execute("""
            INSERT INTO telemetry (timestamp, voltage, current, frequency, battery_soc, solar_yield_kwh)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (t_str, voltage, current, freq, battery, yield_kwh))
        conn.commit()
        log_event("INFO", "TELEMETRY", "Seeded 24h simulated grid telemetry data")
    conn.close()
