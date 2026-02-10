import sqlite3

DB_NAME = "predictions.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        temperature REAL,
        irradiance REAL,
        hour INTEGER,
        model TEXT,
        prediction REAL
    )
    """)

    conn.commit()
    conn.close()

def insert_prediction(time, temp, irradiance, hour, model, prediction):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (time, temperature, irradiance, hour, model, prediction)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (time, temp, irradiance, hour, model, prediction))

    conn.commit()
    conn.close()

def fetch_predictions():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT time, temperature, irradiance, hour, model, prediction FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows

