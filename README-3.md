# Emergency Service Response Time Prediction

## Project Overview

This project implements a data-driven approach for predicting emergency response time and supporting emergency dispatch decisions. The system uses the Integrated Emergency Response Analytics Dataset (IERAD) from Kaggle to train and compare machine learning models that estimate response time based on emergency, operational, environmental, and resource-related features.

The main goal of the project is to support emergency dispatching by predicting expected response time and recommending a suitable dispatch option, such as ambulance dispatch, drone dispatch, hybrid dispatch, or delayed/manual review.

## Dataset


The dataset used in this project is the Integrated Emergency Response Analytics Dataset (IERAD), obtained from Kaggle.

Due to the large file size, the dataset is not included in this repository.

To run the notebook:

1. Download the dataset from Kaggle.
2. Place the CSV file in the project folder.
3. Make sure the file name matches the name used in the notebook:

Important columns include:

- `Timestamp`
- `Incident_Severity`
- `Incident_Type`
- `Region_Type`
- `Traffic_Congestion`
- `Weather_Condition`
- `Drone_Availability`
- `Ambulance_Availability`
- `Battery_Life`
- `Air_Traffic`
- `Response_Time`
- `Hospital_Capacity`
- `Distance_to_Incident`
- `Number_of_Injuries`
- `Specialist_Availability`
- `Road_Type`
- `Emergency_Level`
- `Drone_Speed`
- `Ambulance_Speed`
- `Payload_Weight`
- `Fuel_Level`
- `Weather_Impact`
- `Dispatch_Coordinator`
- `Label`

The target variable is:

```text
Response_Time
```

## Development Environment

The implementation was developed using:

- Programming language: Python
- Environment: Jupyter Notebook
- Dataset format: CSV

## Libraries Used

The main libraries used in this project are:

- Pandas: data loading, inspection, cleaning, and transformation
- NumPy: numerical operations and handling invalid values
- Scikit-learn: train-test splitting, imputation, and evaluation metrics
- XGBoost: quantile regression model implementation
- CatBoost: categorical-feature-based quantile regression
- Random Forest Quantile Regression: quantile-based Random Forest prediction
- Matplotlib: visualization of prediction results and intervals

## Data Pipeline

The data pipeline includes the following stages:

### 1. Data Loading

The CSV dataset is loaded using Pandas:

```python
df = pd.read_csv("emergency_service_routing_with_timestamps.csv")
```

After loading, the dataset is inspected using commands such as:

```python
df.head()
df.info()
df.isnull().sum()
```

### 2. Data Cleaning

The cleaning process includes:

- Checking dataset structure
- Converting `Response_Time` to numeric format
- Converting `Timestamp` to datetime format
- Checking duplicate records
- Replacing infinite values with missing values

Example:

```python
df["Response_Time"] = pd.to_numeric(df["Response_Time"], errors="coerce")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
```

### 3. Missing Value Handling

Missing values are handled using imputation.

Numerical columns are filled using median imputation:

```python
num_imputer = SimpleImputer(strategy="median")
```

Categorical columns are filled using a constant value:

```python
cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
```

This approach preserves all emergency records and avoids unnecessary data removal.

### 4. Feature Engineering

Two estimated time of arrival features are created:

```python
df["eta_drone"] = df["Distance_to_Incident"] / df["Drone_Speed"] * 60
df["eta_ambulance"] = df["Distance_to_Incident"] / df["Ambulance_Speed"] * 60
```

Time-based features are extracted from the `Timestamp` column:

```python
df["timestamp_year"] = df["Timestamp"].dt.year
df["timestamp_month"] = df["Timestamp"].dt.month
df["timestamp_day"] = df["Timestamp"].dt.day
df["timestamp_dayofweek"] = df["Timestamp"].dt.dayofweek
df["timestamp_hour"] = df["Timestamp"].dt.hour
```

The `Response_Time` column is used as the target variable. The `Label` column is removed from the input features to avoid learning directly from previous dispatch decisions.

### 5. Data Splitting

The dataset is split into training and testing sets:

- Training rows: 294,452
- Testing rows: 73,613

Categorical variables are one-hot encoded for Random Forest and XGBoost. After encoding, both the training and testing sets contain 38 features.

## Models Implemented

Three machine learning models are implemented:

### 1. Random Forest

Random Forest is implemented using Quantile Random Forest. It predicts multiple response time quantiles instead of only a single value.

Predicted quantiles:

- Q10: lower response time estimate
- Q50: median response time estimate
- Q90: high-delay response time estimate

### 2. XGBoost

XGBoost is implemented using quantile regression. Since XGBoost requires numerical input, categorical variables are one-hot encoded before training.

Separate models are trained for Q10, Q50, and Q90 using:

```python
objective="reg:quantileerror"
```

### 3. CatBoost

CatBoost is implemented using quantile regression and handles categorical variables directly. This reduces the need for one-hot encoding and is useful because the dataset contains many categorical features.

The quantile loss function is used:

```python
loss_function=f"Quantile:alpha={q}"
```

## Quantile Prediction

The project uses quantile prediction to represent uncertainty in emergency response time.

The three main quantiles are:

- `response_q10`: lower prediction estimate
- `response_q50`: median prediction estimate
- `response_q90`: upper/high-delay prediction estimate

The Q90 prediction is especially important because it helps identify cases where emergency response may be delayed.

## Dispatch Recommendation

After predicting response time, the system recommends a dispatch action for each emergency case.

The recommendation categories are:

- Hybrid Dispatch
- Drone Dispatch
- Ambulance Dispatch
- Delayed / Manual Review

The final dispatch recommendation distribution in the test set was:

| Recommended Dispatch | Number of Cases |
|---|---:|
| Hybrid Dispatch | 46,463 |
| Delayed / Manual Review | 20,071 |
| Drone Dispatch | 5,122 |
| Ambulance Dispatch | 1,957 |

## Evaluation Metrics

The following metrics are used to evaluate model outputs:

- R² Score
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Pinball Loss for Q10, Q50, and Q90
- Q90 Coverage
- Q10-Q90 Coverage
- Average Interval Width

These metrics evaluate both point prediction accuracy and quantile prediction reliability.

## Project Output

The project produces:

1. Cleaned and preprocessed emergency response data
2. Engineered ETA and timestamp-based features
3. Trained Random Forest, XGBoost, and CatBoost models
4. Quantile response time predictions
5. Dispatch recommendations for emergency cases
6. Model evaluation results

## How to Run the Project

1. Open the Jupyter Notebook file.
2. Place the dataset CSV file in the same working directory as the notebook.
3. Install the required Python libraries.
4. Run the notebook cells in order.
5. Review the generated predictions, evaluation results, and dispatch recommendations.

## Example Installation

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib
```

If using Quantile Random Forest, install the required package used in the notebook environment.

## Notes

This project focuses on model implementation and dispatch decision support. The dataset is used as a proxy because structured local emergency data from Gaza is not available. Future work can improve the system by integrating real-time local data, road accessibility information, hospital capacity updates, and live traffic conditions.
