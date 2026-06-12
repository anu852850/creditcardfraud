# Credit Card Fraud Detection

A machine learning web application that detects fraudulent credit card transactions in real time using XGBoost with Optuna hyperparameter tuning, served via a Gradio interface.

---

## Demo

![App Screenshot](assets/screenshot.png)

Run locally:

```bash
python app.py
```

Opens at `http://127.0.0.1:7862`

---

## Project Overview

This project builds an end-to-end fraud detection pipeline on a real-world credit card transaction dataset. It covers data cleaning, feature engineering, model comparison across multiple algorithms, hyperparameter optimization with Optuna, SHAP-based explainability, and a deployable web UI.

Multiple models were trained and evaluated — with and without SMOTE oversampling — to handle the class imbalance problem common in fraud datasets. The best performing model (XGBoost + Optuna) was selected for deployment.

---

## Features

- Real-time fraud prediction from transaction inputs
- Probability score with risk level (High / Medium / Low / Normal)
- Recommended action per transaction
- Multiple models trained and compared
- SMOTE experimentation for handling class imbalance
- Optuna-tuned XGBoost as final model
- SHAP explainability notebook included
- Clean Gradio UI

---

## Project Structure

```
credit card fraud detection/
│
├── assets/
│   └── screenshot.png
│
├── data/
│   ├── raw/
│   │   └── fraud_data.csv
│   └── processed/
│       ├── X_train_processed.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── xgb_optuna.pkl
│   ├── preprocessor_v2.pkl
│   ├── best_threshold_optuna.pkl
│   ├── best_threshold.pkl
│   ├── random_forest_with_smote.pkl
│   └── random_forest_without_smote.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── featureengineering.ipynb
│   ├── modeltraining.ipynb
│   └── shapexplanability.ipynb
│
├── app.py
└── README.md
```

---

## Tech Stack

| Component | Library |
|---|---|
| Models | XGBoost, Random Forest, Logistic Regression |
| Hyperparameter Tuning | Optuna |
| Class Imbalance Handling | SMOTE (imbalanced-learn) |
| Preprocessing | Scikit-learn (ColumnTransformer) |
| Explainability | SHAP |
| Web UI | Gradio |
| Data Handling | Pandas, NumPy |
| Model Persistence | Joblib, Pickle |

---

## Models Trained

| Model | SMOTE | Tuning | Notes |
|---|---|---|---|
| Logistic Regression | Without SMOTE | Default | Baseline model |
| Logistic Regression | With SMOTE | Default | Improved recall |
| Random Forest | Without SMOTE | Default | Better than LR baseline |
| Random Forest | With SMOTE | Default | Strong recall on minority class |
| XGBoost | With SMOTE | Optuna | Final deployed model |

XGBoost with Optuna tuning and a custom probability threshold gave the best balance of precision and recall and was selected for the Gradio app.

---

## Input Features

| Feature | Description |
|---|---|
| Transaction Category | Type of merchant (shopping, travel, etc.) |
| Transaction Amount | Value in USD |
| Gender | Customer gender |
| State | US state of transaction |
| Customer Lat/Long | Customer location coordinates |
| City Population | Population of customer's city |
| Merchant Lat/Long | Merchant location coordinates |
| Day of Week | 0 = Monday, 6 = Sunday |
| Month | 1-12 |
| Hour of Day | 0-23 |
| Customer Age | Age of cardholder |
| Distance (KM) | Distance between customer and merchant |

Engineered features derived automatically: `weekend`, `night`, `log_amt`, `age_group`, `amount_group`, `city_pop_group`, `distance_group`

---

## How It Works

1. Raw transaction inputs are collected from the Gradio UI
2. Feature engineering runs on the fly (binning, log transform, flags)
3. The saved `preprocessor_v2.pkl` encodes categorical and scales numerical features
4. `xgb_optuna.pkl` predicts fraud probability
5. Probability is compared against `best_threshold_optuna.pkl` (Optuna-tuned)
6. Result is returned with risk level and recommended action

---

## Setup and Installation

```bash
# Clone the repo
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create conda environment
conda create -n fraud_detection python=3.10
conda activate fraud_detection

# Install dependencies
pip install gradio xgboost optuna scikit-learn imbalanced-learn pandas numpy joblib shap
```

Then run:

```bash
python app.py
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `eda.ipynb` | Exploratory data analysis, class imbalance check, distributions |
| `featureengineering.ipynb` | Feature creation, preprocessing pipeline, saving preprocessor |
| `modeltraining.ipynb` | Training all models (LR, RF, XGBoost) with and without SMOTE, Optuna tuning, threshold selection |
| `shapexplanability.ipynb` | SHAP values, feature importance, model interpretability |

---

## Key Design Decisions

- Compared models with and without SMOTE to understand the effect of oversampling on fraud recall
- Used `preprocessor_v2.pkl` with correct ordinal encoding for `distance_group` (`Nearby`, `Medium`, `Far`, `Very Far`)
- Threshold saved from Optuna training — not hardcoded — ensures deployment threshold matches training
- `log_amt` used instead of raw amount to handle skewed transaction value distribution
- Distance between customer and merchant computed as an engineered feature to capture geographic anomalies

---

## Author

Anmakshi
B.Tech — G.B. Pant Institute of Engineering and Technology
Internship Project | 2025-2026

---

## License

This project is for academic and portfolio purposes.