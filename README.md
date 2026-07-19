# Emergency Response Time Prediction

## Overview

Emergency Response Time Prediction is a machine learning project designed to estimate emergency response times using historical emergency incident data. The system combines data preprocessing, feature engineering, and quantile regression techniques to predict lower (Q10), median (Q50), and upper (Q90) response-time estimates. In addition to prediction, the project provides a rule-based dispatch recommendation to support emergency response planning.

A Flask application is included to deploy the trained models and allow users to generate predictions through a web interface.

---

## Project Objectives

- Predict emergency response times using machine learning.
- Compare the performance of multiple quantile regression models.
- Estimate prediction uncertainty using Q10, Q50, and Q90 quantiles.
- Support dispatch decision-making through rule-based recommendations.
- Deploy the trained models as a Flask web application.

---

## Repository Structure

```text
EmergencyResponseTimePrediction/
│
├── Dataset/
├── EmergencyResponseFlaskApp/
├── Test1.ipynb
├── Test1_updated final.ipynb
├── Graduation_Sections(3,4,5).pdf
└── README.md
```

---

## Repository Contents

### Dataset
Contains the emergency incident dataset used for preprocessing, feature engineering, model training, and evaluation.

### Jupyter Notebooks
- **Test1.ipynb** – Initial data exploration and model development.
- **Test1_updated final.ipynb** – Final implementation including preprocessing, feature engineering, model training, model evaluation, and prediction.

### Flask Application
The `EmergencyResponseFlaskApp` directory contains the deployed application, trained models, preprocessing artifacts, and prediction interface.

---

## Machine Learning Models

The project evaluates and compares three quantile regression models:

- CatBoost
- XGBoost
- Quantile Random Forest

Each model predicts:

- **Q10** – Lower response-time estimate
- **Q50** – Median response-time estimate
- **Q90** – Upper response-time estimate

---

## Project Workflow

1. Dataset loading
2. Data inspection
3. Data preprocessing
4. Feature engineering
5. Train-test split
6. Model training
7. Response-time prediction
8. Model evaluation
9. Dispatch recommendation generation

---

## Technologies Used

- Python
- Jupyter Notebook
- Flask
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- Quantile Random Forest
- Joblib
- Git
- GitHub

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/AfnanShomar/EmergencyResponseTimePrediction.git
cd EmergencyResponseTimePrediction
```

### Create a Virtual Environment

Using Conda (recommended):

```bash
conda create -n emergency-response python=3.12
conda activate emergency-response
```

### Install Dependencies

```bash
python -m pip install -r EmergencyResponseFlaskApp/requirements.txt
```

### Run the Application

```bash
cd EmergencyResponseFlaskApp
python app.py
```

The application will be available at:

```
http://127.0.0.1:5000
```

For more information about the Flask application, refer to the `README.md` file inside the **EmergencyResponseFlaskApp** directory.

---

## Future Improvements

- Improve prediction accuracy using additional real-world features.
- Integrate real-time traffic and weather information.
- Deploy the application to a cloud platform.


This project was developed for educational and research purposes.
