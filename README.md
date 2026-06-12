# Credit Card Fraud Detection

A machine learning web application that detects fraudulent credit card transactions in real time using XGBoost with Optuna hyperparameter tuning and an interactive Gradio interface.

---

## Demo

### Application Interface

![Application Interface](assets/creditcardfraud.png)

### ROC-AUC Curve

![ROC-AUC Curve](assets/auc%20curve.png)

### Confusion Matrix

![Confusion Matrix](assets/confusionmatrix.png)

### Cost vs Threshold Analysis

![Cost vs Threshold](assets/costvsthreshold.png)

Run locally:

```bash
python app.py

## Project Overview

This project implements an end-to-end Credit Card Fraud Detection system using machine learning and advanced feature engineering techniques. The workflow covers data preprocessing, feature engineering, handling class imbalance with SMOTE, model training, hyperparameter optimization using Optuna, model explainability using SHAP, and deployment through a Gradio web application.

Several machine learning models were trained and compared to identify the best-performing solution. After experimentation, XGBoost with Optuna-based hyperparameter tuning achieved the best balance between precision, recall, and fraud detection capability and was selected for deployment.

---

## Features

* Real-time fraud prediction
* Interactive Gradio web application
* Fraud probability estimation
* Risk categorization (Normal, Low, Medium, High)
* Recommended action generation
* Feature engineering pipeline
* SMOTE-based imbalance handling
* Optuna hyperparameter tuning
* SHAP model explainability
* Multiple model comparison and evaluation

---

## Repository Structure

```text
credit-card-fraud-detection/
│
├── assets/
│   ├── creditcardfraud.png
│   ├── auc curve.png
│   ├── confusionmatrix.png
│   └── costvsthreshold.png
│
├── notebooks/
│   ├── eda.ipynb
│   ├── featureengineering.ipynb
│   ├── modeltraining.ipynb
│   └── shapexplanability.ipynb
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

**Note:** Large datasets and trained model artifacts are excluded from the repository because they exceed GitHub's file size limits.

---

## Technology Stack

| Component                   | Library/Tool                                |
| --------------------------- | ------------------------------------------- |
| Machine Learning            | XGBoost, Random Forest, Logistic Regression |
| Hyperparameter Optimization | Optuna                                      |
| Imbalanced Data Handling    | SMOTE                                       |
| Preprocessing               | Scikit-learn                                |
| Explainability              | SHAP                                        |
| Web Application             | Gradio                                      |
| Data Processing             | Pandas, NumPy                               |
| Model Serialization         | Joblib, Pickle                              |

---

## Models Evaluated

| Model               | SMOTE | Hyperparameter Tuning |
| ------------------- | ----- | --------------------- |
| Logistic Regression | No    | No                    |
| Logistic Regression | Yes   | No                    |
| Random Forest       | No    | No                    |
| Random Forest       | Yes   | No                    |
| XGBoost             | Yes   | Optuna                |

The Optuna-tuned XGBoost model achieved the strongest performance and was selected for deployment.

---

## Input Features

* Transaction Category
* Transaction Amount
* Gender
* State
* Customer Latitude
* Customer Longitude
* Merchant Latitude
* Merchant Longitude
* City Population
* Day of Week
* Month
* Hour
* Customer Age
* Distance Between Customer and Merchant

### Engineered Features

* weekend
* night
* log_amt
* age_group
* amount_group
* city_pop_group
* distance_group

---

## Workflow

1. User enters transaction details through the Gradio interface.
2. Feature engineering is applied automatically.
3. Data preprocessing pipeline transforms features.
4. XGBoost predicts fraud probability.
5. Probability is evaluated against the optimized threshold.
6. Risk category and recommendation are generated.
7. Results are displayed to the user.

---

## Notebooks

| Notebook                 | Description                   |
| ------------------------ | ----------------------------- |
| eda.ipynb                | Exploratory Data Analysis     |
| featureengineering.ipynb | Feature Engineering Pipeline  |
| modeltraining.ipynb      | Model Training and Evaluation |
| shapexplanability.ipynb  | SHAP-Based Explainability     |

---

## Key Design Decisions

* Used SMOTE to address severe class imbalance.
* Applied Optuna for automated hyperparameter optimization.
* Implemented threshold optimization instead of relying on the default 0.5 threshold.
* Used logarithmic transformation for transaction amount.
* Created geographic distance-based fraud indicators.
* Included explainability through SHAP values.

---

## Author

**Anmakshi**
B.Tech – G.B. Pant Institute of Engineering and Technology
Internship Project (2025–2026)

---

## License

This project is intended for educational, research, and portfolio purposes.
