import os
import sqlite3
import requests
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# ------------------------
# Flask App Initialization
# ------------------------
app = Flask(__name__)

# ------------------------
# Environment Variables
# ------------------------
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

# ------------------------
# Load ML Models
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "models", "linear.pkl"), "rb") as f:
    linear_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "random_forest.pkl"), "rb") as f:
    rf_model = pickle.load(f)

# ------------------------
# Database Setup
# ------------------------
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            hour INTEGER,
            temperature REAL,
            irradiance REAL,
            model_used TEXT,
            prediction REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------
# Helper: Fetch Weather Data
# ------------------------
def get_weather(city):
    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={WEATHER_API_KEY}&q={city}"
    )
    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return None, None

    temperature = data["current"]["temp_c"]
    irradiance = data["current"].get("uv", 5)  # fallback UV

    return temperature, irradiance

# ------------------------
# Routes
# ------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    city = request.form.get("city")
    hour = int(request.form.get("hour"))
    model_choice = request.form.get("model")

    temperature, irradiance = get_weather(city)
    if temperature is None:
        return jsonify({"error": "Invalid city name"}), 400

    features = pd.DataFrame([[hour, temperature, irradiance]],
                             columns=["hour", "temperature", "irradiance"])

    if model_choice == "random_forest":
        prediction = rf_model.predict(features)[0]
    else:
        prediction = linear_model.predict(features)[0]

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (city, hour, temperature, irradiance, model_used, prediction)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (city, hour, temperature, irradiance, model_choice, float(prediction)))
    conn.commit()
    conn.close()

    return jsonify({
        "city": city,
        "hour": hour,
        "temperature": temperature,
        "irradiance": irradiance,
        "model": model_choice,
        "prediction": round(float(prediction), 2)
    })

@app.route("/download")
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    csv_path = os.path.join(BASE_DIR, "prediction_history.csv")
    df.to_csv(csv_path, index=False)

    return send_file(csv_path, as_attachment=True)

# ------------------------
# Railway / Gunicorn Entry
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

