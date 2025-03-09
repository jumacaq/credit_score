# Credit Risk Prediction Model

## Overview
This repository contains a machine learning solution developed for a German financial institution to assess credit default probability. The model helps identify high-risk customers and optimize lending decisions through accurate default prediction.

## Problem Statement
The institution needs to reduce credit risk by more accurately assessing the probability of customer defaults. This project addresses this need by developing, evaluating, and implementing appropriate ML models.

## Repository Structure
```
├── CreditRisk.csv           # Original dataset
├── credit_risk_clean.csv    # Preprocessed dataset
├── random_forest_model.pkl  # Serialized Random Forest model
├── scaler.pkl               # Serialized StandardScaler
└── notebooks/               # Jupyter notebooks with analysis
```

## Features
The model analyzes various factors including:
- Customer demographics (age, marital status)
- Financial status (income, expenses, assets, debt)
- Loan characteristics (amount, time)
- Housing situation
- Employment information
- Credit history

## Model Performance
After evaluating multiple algorithms, the ROC-optimized Random Forest classifier was selected as the best-performing model with:

| Metric | Value |
|--------|-------|
| Accuracy | 0.77 |
| F1 (Default Class) | 0.68 |
| Recall (Default) | 0.81 |
| Precision (Default) | 0.58 |
| AUC-ROC | 0.842 |

## Key Technical Components

### Data Preprocessing
- Missing value imputation using median values
- Outlier truncation using quantile-based limits
- Feature engineering (binary debt indicator)
- Categorical variable encoding
- Data normalization via StandardScaler

### Model Development
The project evaluated three primary models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Each model was evaluated with both balanced (SMOTE) and unbalanced datasets, with optimized probability thresholds.

### Credit Scoring System
The model produces a credit score on a scale of 0-1000, where higher scores indicate lower default probability. Score deciles provide stratified risk levels for business decision-making.

## Usage

### Prerequisites
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
```

### Prediction Example
```python
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare customer data
customer_data = pd.DataFrame({
    'seniority': [5.0], 'time': [36.0], 'age': [35.0], 
    'expenses': [800.0], 'income': [3500.0], 'assets': [15000.0], 
    'debt': [1.0], 'amount': [5000.0], 'price': [0.0],
    'home_other': [0.0], 'home_owner': [1.0], 'home_parents': [0.0], 'home_rent': [0.0],
    'marital_married': [1.0], 'marital_other': [0.0], 'marital_single': [0.0],
    'records_no': [1.0], 'records_yes': [0.0],
    'job_fixed': [1.0], 'job_freelance': [0.0], 'job_others': [0.0], 'job_partime': [0.0]
})

# Scale features
scaled_data = scaler.transform(customer_data)

# Generate prediction
default_probability = model.predict_proba(scaled_data)[:, 1][0]
default_prediction = int(default_probability >= 0.28)  # Using optimized threshold
credit_score = (1 - default_probability) * 1000

print(f"Default Probability: {default_probability:.2%}")
print(f"Default Prediction: {'Yes' if default_prediction else 'No'}")
print(f"Credit Score: {credit_score:.0f}/1000")
```

## Implementation Considerations
- The optimal probability threshold is set at 0.28 based on ROC curve analysis
- Score deciles can be used to create risk tiers for business rules
- Model should be periodically retrained as new data becomes available
- Consider implementing model monitoring for drift detection

## Deployment on Streamlit App
- The chosen model ROC-optimized Random Forest classifier is used to deploy on the streamlit app


## Future Improvements
- Feature importance analysis to identify strongest predictors
- Hyperparameter tuning for further model optimization
- Development of explainability tools for customer-facing applications
- Implementation of model versioning and monitoring infrastructure