"""
Emergency Response Time Prediction & Dispatch Recommendation - Flask App
=========================================================================
Redesigned as a QUICK-ENTRY tool: in a real emergency there is no time to
fill in 20+ fields, so the form only collects the handful of details that
(a) genuinely change from incident to incident and (b) a dispatcher can
supply in a few seconds. Every other feature the model needs is filled in
automatically from realistic defaults (median / most frequent value)
computed from the real training data in Test1.ipynb, and the timestamp is
captured automatically as "now" instead of being typed in.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from catboost import CatBoostRegressor
from flask import Flask, render_template, request

app = Flask(__name__)

MODELS_DIR = "saved_models"


# ---------------------------------------------------------------------------
# Load all artifacts once, at startup
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
    artifacts["catboost_feature_columns"] = joblib.load(
        os.path.join(MODELS_DIR, "catboost_feature_columns.joblib")
    )
    # Realistic fallback values (median / mode from real training data) for
    # every field NOT collected in the quick-entry form.
    artifacts["default_values"] = joblib.load(os.path.join(MODELS_DIR, "default_values.joblib"))

    return artifacts


ARTIFACTS = load_artifacts()

# ---------------------------------------------------------------------------
# QUICK-ENTRY FIELDS — the only information a dispatcher must type in during
# a real, time-critical incident. Everything else is auto-filled.
# ---------------------------------------------------------------------------
QUICK_FIELDS = [
    "Distance_to_Incident",   # locates the incident relative to the dispatch center
    "Incident_Severity",      # drives triage / dispatch urgency
    "Region_Type",            # affects achievable speed and routing
    "Drone_Availability",     # changes minute to minute
    "Ambulance_Availability", # changes minute to minute
]

NUMERIC_QUICK_FIELDS = {"Distance_to_Incident"}


# ---------------------------------------------------------------------------
# Preprocessing — same logic as Test1.ipynb, but the record is assembled by
# starting from realistic defaults and overwriting only the quick-entry
# fields the dispatcher actually provided.
# ---------------------------------------------------------------------------
def build_full_record(quick_input: dict) -> pd.DataFrame:
    record = dict(ARTIFACTS["default_values"])  # start from realistic defaults
    record.update(quick_input)                  # overwrite with what the dispatcher entered
    record["Timestamp"] = datetime.now()         # always "now" — never asked for
    return pd.DataFrame([record])


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

    numeric_cols = [c for c in ARTIFACTS["numeric_cols"] if c in row.columns]
    categorical_cols = [c for c in ARTIFACTS["categorical_cols"] if c in row.columns]

    row[numeric_cols] = ARTIFACTS["num_imputer"].transform(row[numeric_cols])
    row[categorical_cols] = ARTIFACTS["cat_imputer"].transform(row[categorical_cols])

    for col in categorical_cols:
        row[col] = row[col].astype(str)

    # CatBoost matches categorical features by column POSITION, not name.
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

    quick_input = {field: form.get(field) for field in QUICK_FIELDS}
    for field in NUMERIC_QUICK_FIELDS:
        quick_input[field] = float(quick_input[field])

    row = build_full_record(quick_input)
    processed = preprocess_for_catboost(row)

    q10 = float(ARTIFACTS["catboost_models"][0.1].predict(processed)[0])
    q50 = float(ARTIFACTS["catboost_models"][0.5].predict(processed)[0])
    q90 = float(ARTIFACTS["catboost_models"][0.9].predict(processed)[0])

    dispatch = recommend_dispatch(
        quick_input["Incident_Severity"], q90,
        quick_input["Drone_Availability"], quick_input["Ambulance_Availability"],
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
