import os
import logging
from flask import Flask, render_template, request, jsonify

# -------------------------
# Basic app setup
# -------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------
# Load dataset (safe)
# -------------------------
DATA_LOADED = False
try:
    import pandas as pd
    df = pd.read_csv("solar_data.csv")
    DATA_LOADED = True
    app.logger.info("Solar dataset loaded successfully")
except Exception as e:
    app.logger.warning(f"Dataset not loaded: {e}")

# -------------------------
# Load ML model (SAFE MODE)
# -------------------------
MODEL_AVAILABLE = False
linear_model = None

MODEL_PATH = "models/linear_model.pkl"

try:
    import pickle
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            linear_model = pickle.load(f)
        MODEL_AVAILABLE = True
        app.logger.info("ML model loaded successfully")
    else:
        app.logger.warning("ML model not found, using fallback logic")
except Exception as e:
    app.logger.warning(f"ML model load failed: {e}")

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset_loaded": DATA_LOADED,
        "ml_model": MODEL_AVAILABLE
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hour = float(data.get("hour", 12))

    if MODEL_AVAILABLE:
        prediction = linear_model.predict([[hour]])[0]
        source = "ml_model"
    else:
        # fallback logic
        prediction = round(0.25 * hour + 2.5, 2)
        source = "fallback_logic"

    return jsonify({
        "hour": hour,
        "predicted_energy_kwh": round(float(prediction), 2),
        "source": source
    })

# -------------------------
# IMPORTANT: DO NOT EXIT
# -------------------------
# Gunicorn will handle the server
# DO NOT put app.run() in production

