import os
import sqlite3
import pickle
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, jsonify

# --------------------------------
# Flask app
# --------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "solar_model.pkl")

# --------------------------------
# Database init
# --------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            city TEXT,
            hour INTEGER,
            prediction REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --------------------------------
# Load ML model (SAFE)
# --------------------------------
ml_model = None
ml_model_loaded = False

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            ml_model = pickle.load(f)
        ml_model_loaded = True
except Exception:
    ml_model = None
    ml_model_loaded = False

# --------------------------------
# Fallback prediction
# --------------------------------
def fallback_prediction(hour):
    if hour < 6 or hour > 18:
        return 0.0
    return round(5 * (1 - abs(hour - 12) / 6), 2)

# --------------------------------
# Routes
# --------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    city = request.form.get("city", "Unknown")
    hour = int(request.form.get("hour", 12))

    if ml_model_loaded:
        prediction = float(ml_model.predict(np.array([[hour]]))[0])
    else:
        prediction = fallback_prediction(hour)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO prediction_history (timestamp, city, hour, prediction) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), city, hour, prediction)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "city": city,
        "hour": hour,
        "prediction": round(prediction, 2),
        "ml_used": ml_model_loaded
    })

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT timestamp, city, hour, prediction
        FROM prediction_history
        ORDER BY timestamp DESC
        LIMIT 100
    """)

    rows = c.fetchall()
    conn.close()

    return render_template("history.html", history=rows)

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "ml_model": ml_model_loaded
    })

# --------------------------------
# Entry (for local only)
# --------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

