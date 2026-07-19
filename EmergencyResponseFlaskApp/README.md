# Emergency Response Dispatch Predictor — Flask App

A minimal Flask web interface for the trained CatBoost quantile regression
models from `Test1.ipynb`. It collects a new incident through a web form,
runs it through the same preprocessing logic used in training, and returns
Q10 / Q50 / Q90 response-time predictions plus a dispatch recommendation
(Hybrid / Ambulance / Drone / Delayed-Manual Review).

## 1. Prerequisites

Run the **"Save trained models to disk"** cell that was added to
`Test1.ipynb` (Section 5.9). This creates a `saved_models/` folder containing:

```
saved_models/
├── catboost_q10.cbm
├── catboost_q50.cbm
├── catboost_q90.cbm
├── xgb_quantile_models.joblib
├── qrf_model.joblib
├── num_imputer.joblib
├── cat_imputer.joblib
├── numeric_cols.joblib
├── categorical_cols.joblib
├── encoded_columns.joblib
└── cat_features.joblib
```

Copy that entire `saved_models/` folder into this Flask app's root directory
(next to `app.py`) before running the app.

## 2. Local setup (Anaconda / any Python environment)

```bash
cd EmergencyResponseFlaskApp
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 3. Project structure

```
EmergencyResponseFlaskApp/
├── app.py                 # Flask routes, preprocessing, dispatch logic
├── templates/
│   └── index.html         # Form + result page
├── saved_models/          # Trained model artifacts (from Test1.ipynb)
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. Pushing this app to your existing GitHub repository

This app was generated locally and is not yet inside your GitHub
repository. Since pushing requires your own GitHub credentials, run the
following commands yourself from a terminal (Anaconda Prompt, Git Bash, or
any terminal with `git` installed):

```bash
# 1) Go into your already-cloned project repo
cd EmergencyResponseTimePrediction

# 2) Copy this folder into the repo
#    (adjust the source path to wherever you downloaded EmergencyResponseFlaskApp)
cp -r /path/to/EmergencyResponseFlaskApp ./EmergencyResponseFlaskApp

# 3) Stage, commit, and push
git add EmergencyResponseFlaskApp
git commit -m "Add Flask app for response-time prediction and dispatch recommendation"
git push origin main
```

If you have not cloned the repository locally yet:

```bash
git clone https://github.com/AfnanShomar/EmergencyResponseTimePrediction.git
cd EmergencyResponseTimePrediction
cp -r /path/to/EmergencyResponseFlaskApp ./EmergencyResponseFlaskApp
git add EmergencyResponseFlaskApp
git commit -m "Add Flask app for response-time prediction and dispatch recommendation"
git push origin main
```

You will be prompted for your GitHub username and a **Personal Access
Token** (GitHub no longer accepts account passwords over HTTPS git
operations). You can generate one from:
GitHub → Settings → Developer settings → Personal access tokens.

## 5. Note on trained model files

CatBoost, XGBoost, and Random Forest model files can be a few megabytes
each. If GitHub warns about large files, consider using
[Git LFS](https://git-lfs.com/) for the `saved_models/` folder, or keep the
models out of version control and document the retraining step instead.
