import os
import logging
import pickle
import io
import json
import psutil
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for

import db

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize database tables and seed data
db.init_db()

# Cache loaded models
LOADED_MODELS = {}

def get_ml_model(model_name=None):
    if not model_name:
        settings = db.get_system_settings()
        model_name = settings.get("active_model", "solar_model.pkl")

    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    model_path = os.path.join("models", model_name)
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)
            LOADED_MODELS[model_name] = loaded
            db.log_event("INFO", "MODEL_LOADER", f"Successfully loaded model: {model_name}")
            return loaded
        except Exception as e:
            db.log_event("WARNING", "MODEL_LOADER", f"Failed to load {model_name}: {e}")

    default_path = os.path.join("models", "solar_model.pkl")
    if os.path.exists(default_path):
        try:
            with open(default_path, "rb") as f:
                loaded = pickle.load(f)
            LOADED_MODELS["solar_model.pkl"] = loaded
            return loaded
        except Exception as e:
            pass

    return None

def get_available_models():
    if os.path.exists("models"):
        models = [f for f in os.listdir("models") if f.endswith(".pkl")]
        if models:
            return models
    return ["solar_model.pkl", "linear.pkl"]

# ---------------------------------------------------------
# Dedicated Portal Routes
# ---------------------------------------------------------

@app.route("/")
@app.route("/admin")
def admin_portal():
    settings = db.get_system_settings()
    recent_predictions = db.fetch_predictions(limit=10)
    system_logs = db.fetch_logs(limit=15)
    telemetry = db.fetch_telemetry(limit=24)
    total_count = db.get_prediction_count()

    return render_template(
        "admin.html",
        settings=settings,
        recent_predictions=recent_predictions,
        system_logs=system_logs,
        telemetry=telemetry,
        total_count=total_count,
        available_models=get_available_models()
    )

@app.route("/controller")
def controller_portal():
    settings = db.get_system_settings()
    return render_template(
        "controller.html",
        settings=settings,
        available_models=get_available_models()
    )

@app.route("/predictor")
def predictor_portal():
    settings = db.get_system_settings()
    return render_template(
        "predictor.html",
        settings=settings,
        available_models=get_available_models()
    )

@app.route("/database")
def database_portal():
    predictions = db.fetch_predictions(limit=25)
    total_records = db.get_prediction_count()
    return render_template(
        "database.html",
        predictions=predictions,
        total_records=total_records
    )

@app.route("/logs")
def logs_portal():
    logs = db.fetch_logs(limit=50)
    return render_template("logs.html", logs=logs)

@app.route("/history")
def history_page():
    search = request.args.get("search", "").strip()
    page = int(request.args.get("page", 1))
    per_page = 20
    offset = (page - 1) * per_page

    predictions = db.fetch_predictions(search=search if search else None, limit=per_page, offset=offset)
    total_records = db.get_prediction_count(search=search if search else None)
    total_pages = max(1, (total_records + per_page - 1) // per_page)

    return render_template(
        "history.html",
        predictions=predictions,
        search=search,
        page=page,
        total_pages=total_pages,
        total_records=total_records
    )

# ---------------------------------------------------------
# Prediction Core Route
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            temp = float(data.get("temperature", 25.0))
            hour = float(data.get("hour", 12.0))
            irradiance = float(data.get("irradiance", 800.0))
            selected_model = data.get("model", None)
        else:
            temp = float(request.form.get("temperature", 25.0))
            hour = float(request.form.get("hour", 12.0))
            irradiance = float(request.form.get("irradiance", 800.0))
            selected_model = request.form.get("model", None)

        settings = db.get_system_settings()
        if not selected_model:
            selected_model = settings.get("active_model", "solar_model.pkl")

        if settings.get("emergency_cutoff") == "ON":
            msg = "Prediction blocked: Emergency Cutoff is ACTIVE"
            if request.is_json:
                return jsonify({"status": "error", "message": msg}), 400
            return render_template("predictor.html", error=msg, settings=settings, available_models=get_available_models())

        active_model_obj = get_ml_model(selected_model)
        if active_model_obj:
            try:
                prediction_val = active_model_obj.predict([[temp, irradiance, hour]])[0]
            except Exception:
                try:
                    prediction_val = active_model_obj.predict([[temp, hour]])[0]
                except Exception as ex:
                    logging.warning(f"Model prediction failed: {ex}")
                    active_model_obj = None

        if not active_model_obj:
            solar_efficiency = 0.18
            irradiance_factor = irradiance / 1000.0
            temp_penalty = 1.0 - max(0, (temp - 25) * 0.004)
            time_factor = max(0, sin_sun_hour(hour))
            prediction_val = round(10.0 * irradiance_factor * temp_penalty * time_factor * 50.0, 2)

        prediction_val = round(float(prediction_val), 2)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db.insert_prediction(
            time_str=time_str,
            temp=temp,
            irradiance=irradiance,
            hour=int(hour),
            model_name=selected_model,
            prediction=prediction_val,
            status="RECORDED"
        )

        if request.is_json:
            return jsonify({
                "status": "success",
                "prediction": prediction_val,
                "model_used": selected_model,
                "timestamp": time_str
            })

        return redirect(url_for("predictor_portal"))

    except Exception as e:
        db.log_event("ERROR", "PREDICTOR", f"Prediction execution error: {e}")
        if request.is_json:
            return jsonify({"status": "error", "message": str(e)}), 500
        return f"Prediction error: {e}", 500

def sin_sun_hour(hour):
    import math
    if 6 <= hour <= 18:
        return math.sin((hour - 6) / 12.0 * math.pi)
    return 0.0

# ---------------------------------------------------------
# System Controller APIs
# ---------------------------------------------------------

@app.route("/api/system/status", methods=["GET"])
def get_system_status():
    settings = db.get_system_settings()
    cpu_percent = psutil.cpu_percent() if hasattr(psutil, "cpu_percent") else 12.5
    ram_percent = psutil.virtual_memory().percent if hasattr(psutil, "virtual_memory") else 45.2
    
    telemetry = db.fetch_telemetry(limit=1)
    latest_telemetry = telemetry[0] if telemetry else {
        "voltage": 230.0,
        "current": 18.5,
        "frequency": 50.0,
        "battery_soc": 85.0,
        "solar_yield_kwh": 340.5
    }

    return jsonify({
        "status": "online",
        "settings": settings,
        "system_metrics": {
            "cpu_usage": cpu_percent,
            "ram_usage": ram_percent,
            "total_records": db.get_prediction_count(),
            "latest_telemetry": latest_telemetry
        }
    })

@app.route("/api/system/control", methods=["POST"])
def update_system_control():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON body"}), 400

        allowed_keys = ["grid_status", "dispatch_mode", "active_model", "max_power_limit", "emergency_cutoff", "auto_tuning"]
        updated = {}
        for key in allowed_keys:
            if key in data:
                db.update_system_setting(key, data[key])
                updated[key] = data[key]

        return jsonify({
            "status": "success",
            "message": "System configuration updated successfully",
            "updated_settings": updated
        })
    except Exception as e:
        db.log_event("ERROR", "CONTROLLER", f"System control update failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------------------------------------
# Database Storage Portal APIs
# ---------------------------------------------------------

@app.route("/api/database/records", methods=["GET"])
def get_database_records():
    search = request.args.get("search", "").strip()
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    records = db.fetch_predictions(search=search if search else None, limit=limit, offset=offset)
    total = db.get_prediction_count(search=search if search else None)

    return jsonify({
        "status": "success",
        "total_records": total,
        "returned_records": len(records),
        "records": records
    })

@app.route("/api/database/export", methods=["GET"])
def export_database():
    fmt = request.args.get("format", "csv").lower()
    records = db.fetch_predictions(limit=10000)

    if fmt == "json":
        json_data = json.dumps(records, indent=2)
        return Response(
            json_data,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=solar_database_export.json"}
        )

    df = pd.DataFrame(records)
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_data = output.getvalue()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=solar_database_export.csv"}
    )

@app.route("/api/database/import", methods=["POST"])
def import_database():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"status": "error", "message": "Only CSV files are supported"}), 400

        df = pd.read_csv(file)
        count = 0
        for _, row in df.iterrows():
            time_val = str(row.get("time", row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
            temp_val = float(row.get("temperature", 25.0))
            irr_val = float(row.get("irradiance", 800.0))
            hour_val = int(row.get("hour", 12))
            model_val = str(row.get("model", "IMPORT_CSV"))
            pred_val = float(row.get("prediction", 0.0))

            db.insert_prediction(
                time_str=time_val,
                temp=temp_val,
                irradiance=irr_val,
                hour=hour_val,
                model_name=model_val,
                prediction=pred_val,
                status="IMPORTED"
            )
            count += 1

        db.log_event("INFO", "DATABASE", f"Imported {count} records via Admin Portal CSV Upload")
        return jsonify({"status": "success", "imported_count": count})

    except Exception as e:
        db.log_event("ERROR", "DATABASE", f"Failed to import database file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/database/clear", methods=["POST"])
def clear_database_records():
    db.clear_predictions()
    return jsonify({"status": "success", "message": "Database records cleared successfully"})

@app.route("/api/logs", methods=["GET"])
def get_system_logs():
    limit = int(request.args.get("limit", 50))
    logs = db.fetch_logs(limit=limit)
    return jsonify({"status": "success", "logs": logs})

@app.route("/api/telemetry", methods=["GET"])
def get_telemetry_data():
    telemetry = db.fetch_telemetry(limit=24)
    return jsonify({"status": "success", "telemetry": telemetry[::-1]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
