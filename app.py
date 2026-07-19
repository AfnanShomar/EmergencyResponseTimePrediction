"""
Emergency Response Time Prediction & Dispatch Recommendation - Flask App
=========================================================================
This app loads the models and preprocessing artifacts that were trained
and saved from Test1.ipynb (see the "Save trained models to disk" cell
added to Section 5.9) and serves predictions through a simple web form.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from flask import Flask, render_template, request

app = Flask(__name__)

MODELS_DIR = "saved_models"

# ---------------------------------------------------------------------------
# Load all artifacts once, at startup, instead of on every request
# ---------------------------------------------------------------------------
def load_artifacts():
    artifacts = {}

    artifacts["catboost_models"] = {}
    for q, tag in [(0.1, 10), (0.5, 50), (0.9, 90)]:
        model = CatBoostRegressor()
        model.load_model(os.path.join(MODELS_DIR, f"catboost_q{tag}.cbm"))
        artifacts["catboost_models"][q] = model

    artifacts["num_imputer"] = joblib.load(os.path.join(MODELS_DIR, "num_imputer.joblib"))
    artifacts["cat_imputer"] = joblib.load(os.path.join(MODELS_DIR, "cat_imputer.joblib"))
    artifacts["numeric_cols"] = joblib.load(os.path.join(MODELS_DIR, "numeric_cols.joblib"))
    artifacts["categorical_cols"] = joblib.load(os.path.join(MODELS_DIR, "categorical_cols.joblib"))
    artifacts["cat_features"] = joblib.load(os.path.join(MODELS_DIR, "cat_features.joblib"))
    # CatBoost matches categorical features by column POSITION, not name, so the
    # exact training column order must be reproduced before every prediction.
    artifacts["catboost_feature_columns"] = joblib.load(
        os.path.join(MODELS_DIR, "catboost_feature_columns.joblib")
    )

    return artifacts


ARTIFACTS = load_artifacts()

# Fields collected from the web form (mirrors the columns used in Test1.ipynb,
# minus Response_Time and Label, which are never part of the model input)
FORM_FIELDS = [
    "Incident_Severity", "Incident_Type", "Region_Type", "Traffic_Congestion",
    "Weather_Condition", "Drone_Availability", "Ambulance_Availability",
    "Battery_Life", "Fuel_Level", "Distance_to_Incident", "Drone_Speed",
    "Ambulance_Speed", "Hospital_Capacity", "Number_of_Injuries",
    "Specialist_Availability", "Road_Type", "Emergency_Level", "Air_Traffic",
    "Weather_Impact", "Dispatch_Coordinator", "Payload_Weight", "Timestamp",
]

NUMERIC_FIELDS = {
    "Battery_Life", "Fuel_Level", "Distance_to_Incident", "Drone_Speed",
    "Ambulance_Speed", "Hospital_Capacity", "Number_of_Injuries", "Payload_Weight",
}

# ---------------------------------------------------------------------------
# Preprocessing — mirrors Test1.ipynb, cells 5, 10, 11
# ---------------------------------------------------------------------------
def engineer_features(row: pd.DataFrame) -> pd.DataFrame:
    row = row.copy()
    row["eta_drone"] = row["Distance_to_Incident"] / row["Drone_Speed"] * 60
    row["eta_ambulance"] = row["Distance_to_Incident"] / row["Ambulance_Speed"] * 60
    row.replace([np.inf, -np.inf], np.nan, inplace=True)

    row["Timestamp"] = pd.to_datetime(row["Timestamp"], errors="coerce")
    row["timestamp_year"] = row["Timestamp"].dt.year
    row["timestamp_month"] = row["Timestamp"].dt.month
    row["timestamp_day"] = row["Timestamp"].dt.day
    row["timestamp_dayofweek"] = row["Timestamp"].dt.dayofweek
    row["timestamp_hour"] = row["Timestamp"].dt.hour
    row.drop(columns=["Timestamp"], inplace=True)
    return row


def preprocess_for_catboost(row: pd.DataFrame) -> pd.DataFrame:
    row = engineer_features(row)

    numeric_cols = ARTIFACTS["numeric_cols"]
    categorical_cols = ARTIFACTS["categorical_cols"]

    # Some engineered numeric columns may not exist in numeric_cols if the
    # training run predates them; guard against KeyErrors defensively.
    numeric_cols = [c for c in numeric_cols if c in row.columns]
    categorical_cols = [c for c in categorical_cols if c in row.columns]

    row[numeric_cols] = ARTIFACTS["num_imputer"].transform(row[numeric_cols])
    row[categorical_cols] = ARTIFACTS["cat_imputer"].transform(row[categorical_cols])

    for col in categorical_cols:
        row[col] = row[col].astype(str)

    # Reindex to the exact column order seen during training so CatBoost's
    # positional categorical-feature indices line up correctly.
    row = row[ARTIFACTS["catboost_feature_columns"]]

    return row


# ---------------------------------------------------------------------------
# Dispatch logic — identical rule set to recommend_dispatch() in Test1.ipynb
# ---------------------------------------------------------------------------
def recommend_dispatch(incident_severity, response_q90, drone_availability, ambulance_availability):
    critical_case = incident_severity == "High"
    high_response_risk = response_q90 > 15
    drone_available = drone_availability == "Available"
    ambulance_available = ambulance_availability == "Available"

    if high_response_risk and drone_available and ambulance_available:
        return "Hybrid Dispatch"
    elif critical_case and ambulance_available:
        return "Ambulance Dispatch"
    elif drone_available:
        return "Drone Dispatch"
    else:
        return "Delayed / Manual Review"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, form_data=None)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    record = {field: form.get(field) for field in FORM_FIELDS}
    for field in NUMERIC_FIELDS:
        record[field] = float(record[field])

    row = pd.DataFrame([record])
    processed = preprocess_for_catboost(row)

    q10 = float(ARTIFACTS["catboost_models"][0.1].predict(processed)[0])
    q50 = float(ARTIFACTS["catboost_models"][0.5].predict(processed)[0])
    q90 = float(ARTIFACTS["catboost_models"][0.9].predict(processed)[0])

    dispatch = recommend_dispatch(
        record["Incident_Severity"], q90,
        record["Drone_Availability"], record["Ambulance_Availability"],
    )

    result = {
        "q10": round(q10, 2),
        "q50": round(q50, 2),
        "q90": round(q90, 2),
        "dispatch": dispatch,
    }

    return render_template("index.html", result=result, form_data=form)


if __name__ == "__main__":
    app.run(debug=True)
