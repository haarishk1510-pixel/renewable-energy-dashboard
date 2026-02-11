import os
import logging
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# -----------------------------
# Load Dataset
# -----------------------------
try:
    dataset = pd.read_csv("solar_data.csv")
    logging.info("Solar dataset loaded successfully")
except Exception as e:
    logging.error(f"Dataset loading failed: {e}")
    dataset = None

# -----------------------------
# Load ML Model
# -----------------------------
model = None
model_path = "models/solar_model.pkl"

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info("ML model loaded successfully")
    except Exception as e:
        logging.warning(f"ML model failed to load: {e}")
else:
    logging.warning("ML model not found, using fallback logic")

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        temperature = float(request.form["temperature"])
        hour = float(request.form["hour"])

        if model:
            prediction = model.predict([[temperature, hour]])[0]
        else:
            # Fallback logic
            prediction = temperature * 0.5 + hour * 0.3

        prediction = round(float(prediction), 2)

        # Save history
        history_file = "prediction_history.csv"

        new_data = pd.DataFrame({
            "timestamp": [pd.Timestamp.now()],
            "temperature": [temperature],
            "hour": [hour],
            "prediction": [prediction]
        })

        if os.path.exists(history_file):
            new_data.to_csv(history_file, mode="a", header=False, index=False)
        else:
            new_data.to_csv(history_file, index=False)

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Error in prediction"

# -----------------------------
# History Route
# -----------------------------
@app.route("/history")
def history():
    try:
        df = pd.read_csv("prediction_history.csv")
        timestamps = df["timestamp"].astype(str).tolist()
        predictions = df["prediction"].tolist()

        return render_template(
            "history.html",
            timestamps=timestamps,
            predictions=predictions
        )
    except:
        return "No prediction history available"

# -----------------------------
# Health Check
# -----------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset_loaded": dataset is not None,
        "ml_model": model is not None
    })

# -----------------------------
# Railway Port Binding
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

