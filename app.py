import os
import logging
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# -------------------------------------------------
# Basic app + logging
# -------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Paths & constants
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "solar_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "solar_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

# -------------------------------------------------
# Globals
# -------------------------------------------------
solar_df = None
ml_model = None
ml_model_loaded = False

# -------------------------------------------------
# Load dataset (SAFE)
# -------------------------------------------------
try:
    solar_df = pd.read_csv(DATA_PATH)
    logger.info("Solar dataset loaded successfully")
except Exception as e:
    logger.error(f"Dataset load failed: {e}")
    solar_df = None

# -------------------------------------------------
# Load ML model (SAFE + VERSION TOLERANT)
# -------------------------------------------------
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            ml_model = pickle.load(f)
        ml_model_loaded = True
        logger.info("ML model loaded successfully")
    else:
        logger.warning("ML model file not found, running without ML")
except Exception as e:
    logger.error(f"ML model load failed: {e}")
    ml_model = None
    ml_model_loaded = False

# -------------------------------------------------
# Database init
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            hour INTEGER,
            prediction REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------------------------------
# Utility: fallback prediction (NO ML)
# -------------------------------------------------
def fallback_prediction(hour: int) -> float:
    """
    Simple heuristic if ML model is unavailable
    """
    if hour < 6 or hour > 18:
        return 0.0
    peak = 12
    return max(0.0, 100 * (1 - abs(hour - peak) / 6))

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form.get("city", "Unknown")
        hour = int(request.form.get("hour", 12))

        # ML prediction if available
        if ml_model_loaded:
            X = np.array([[hour]])
            prediction = float(ml_model.predict(X)[0])
        else:
            prediction = fallback_prediction(hour)

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO predictions (city, hour, prediction, created_at) VALUES (?, ?, ?, ?)",
            (city, hour, prediction, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify({
            "city": city,
            "hour": hour,
            "prediction": round(prediction, 2),
            "ml_used": ml_model_loaded
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset_loaded": solar_df is not None,
        "ml_model": ml_model_loaded
    })

# -------------------------------------------------
# Railway / Gunicorn entry
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

