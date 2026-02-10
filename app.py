import os
import sqlite3
import csv
import datetime
import joblib
import numpy as np
import requests
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

# ================= CONFIG =================
DB_NAME = "predictions.db"
WEATHER_API_KEY = "9c74a5c6cf5b4aee9ad84441261002"
WEATHER_URL = "http://api.weatherapi.com/v1/current.json"

MODELS = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Linear Regression": joblib.load("models/linear.pkl")
}

# ================= DB INIT =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            city TEXT,
            temperature REAL,
            irradiance REAL,
            hour INTEGER,
            model TEXT,
            prediction REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= WEATHER =================
def get_weather(city):
    params = {
        "key": WEATHER_API_KEY,
        "q": city
    }
    r = requests.get(WEATHER_URL, params=params, timeout=10)
    data = r.json()

    if "error" in data:
        return None, None

    temperature = data["current"]["temp_c"]
    cloud = data["current"]["cloud"]

    return temperature, cloud

# ================= IRRADIANCE =================
def estimate_irradiance(hour, cloud):
    if hour < 6 or hour > 18:
        return 100
    base = 1000
    reduction = (cloud / 100) * 600
    return round(base - reduction, 2)

# ================= HOME =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        city = request.form["city"].strip()
        hour = int(request.form["hour"])
        model_name = request.form["model"]

        temperature, cloud = get_weather(city)

        if temperature is None:
            prediction = "City not found"
        else:
            irradiance = estimate_irradiance(hour, cloud)
            model = MODELS[model_name]

            X = np.array([[temperature, irradiance, hour]])
            prediction = round(float(model.predict(X)[0]), 2)

            time_now = datetime.datetime.now().strftime("%H:%M:%S")

            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions
                (time, city, temperature, irradiance, hour, model, prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                time_now, city, temperature,
                irradiance, hour, model_name, prediction
            ))
            conn.commit()
            conn.close()

    # Fetch history
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    history = c.execute("""
        SELECT time, city, temperature,
               irradiance, hour, model, prediction
        FROM predictions
        ORDER BY id DESC
    """).fetchall()
    conn.close()

    return render_template(
        "index.html",
        prediction=prediction,
        history=history,
        models=MODELS.keys()
    )

# ================= CSV DOWNLOAD (FIXED) =================
@app.route("/download-csv")
def download_csv():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        SELECT time, city, temperature,
               irradiance, hour, model, prediction
        FROM predictions
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()

    file_path = os.path.join(os.getcwd(), "prediction_history.csv")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time", "City", "Temperature",
            "Irradiance", "Hour", "Model", "Prediction"
        ])
        writer.writerows(rows)

    return send_file(file_path, as_attachment=True)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)

