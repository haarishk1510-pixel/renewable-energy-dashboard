from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import logging

# -----------------------------
# App configuration
# -----------------------------
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# -----------------------------
# Logging (Production safe)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Load dataset (safe load)
# -----------------------------
DATA_FILE = "solar_data.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            logger.info("Solar dataset loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    else:
        logger.warning("solar_data.csv not found")
        return None

solar_df = load_data()

# -----------------------------
# Utility: Simple prediction logic
# (No sklearn → deployment safe)
# -----------------------------
def predict_energy(hour: int):
    """
    Simple rule-based prediction
    Safe for cloud deployment
    """
    if hour < 6 or hour > 18:
        return 0.0
    elif 6 <= hour < 9:
        return 1.5
    elif 9 <= hour < 15:
        return 4.5
    elif 15 <= hour <= 18:
        return 2.5
    return 0.0

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "renewable-energy-dashboard"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        city = data.get("city", "Unknown")
        hour = int(data.get("hour", 12))

        prediction = predict_energy(hour)

        response = {
            "city": city,
            "hour": hour,
            "predicted_energy_kwh": prediction,
            "unit": "kWh"
        }

        logger.info(f"Prediction generated: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": "Invalid input",
            "message": str(e)
        }), 400

@app.route("/data-preview")
def data_preview():
    if solar_df is None:
        return jsonify({"error": "Dataset not available"}), 404

    return jsonify(solar_df.head(10).to_dict(orient="records"))

# -----------------------------
# Error Handlers
# -----------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# -----------------------------
# Main entry (IMPORTANT)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port)

