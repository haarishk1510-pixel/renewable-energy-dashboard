import os
import json
import logging
from flask import Flask, request, jsonify, render_template

# --------------------
# App Configuration
# --------------------
app = Flask(__name__)
PORT = int(os.environ.get("PORT", 8080))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# --------------------
# Optional ML Model
# --------------------
MODEL_AVAILABLE = False
linear_model = None

MODEL_PATH = "models/linear_model.pkl"

try:
    import pickle
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            linear_model = pickle.load(f)
        MODEL_AVAILABLE = True
        logging.info("ML model loaded successfully")
    else:
        logging.warning("Model file not found, running without ML")
except Exception as e:
    logging.error(f"Model load failed: {e}")
    MODEL_AVAILABLE = False

# --------------------
# Routes
# --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "renewable-energy-dashboard",
        "model_loaded": MODEL_AVAILABLE
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        city = data.get("city", "Unknown")
        hour = float(data.get("hour", 12))

        # --------------------
        # Prediction Logic
        # --------------------
        if MODEL_AVAILABLE:
            prediction = linear_model.predict([[hour]])[0]
            source = "ml_model"
        else:
            # Safe fallback formula
            prediction = round(0.25 * hour + 2.5, 2)
            source = "fallback_logic"

        return jsonify({
            "city": city,
            "hour": hour,
            "predicted_energy_kwh": round(float(prediction), 2),
            "unit": "kWh",
            "source": source
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------
# App Runner (Local only)
# --------------------
if __name__ == "__main__":
    logging.info(f"Starting Flask app on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)

