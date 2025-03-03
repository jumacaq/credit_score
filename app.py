import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("🏦 Predictor de puntaje crediticio")

st.write("""
### Ingresa tus datos para obtener tu puntaje crediticio estimado!
Este modelo predice tu **puntaje crediticio** en función de información financiera y personal.
""")

# Continuous input fields
seniority = st.number_input("Antigüedad laboral", min_value=0, max_value=50, value=5)
time = st.number_input("Tiempo del préstamo (meses)", min_value=6, max_value=72, value=60)
age = st.number_input("Edad", min_value=18, max_value=70, value=30)
expenses = st.number_input("Gastos ($)", min_value=0.0, max_value=100000.0, value=500.0)
income = st.number_input("Ingresos ($)", min_value=0.0, max_value=100000.0, value=2000.0)
assets = st.number_input("Ahorros ($)", min_value=0.0, max_value=10000000.0, value=5000.0)
amount = st.number_input("Monto del préstamo ($)", min_value=0.0, max_value=1000000.0, value=10000.0)
price = st.number_input("Valor total del préstamo ($)", min_value=0.0, max_value=1000000.0, value=15000.0)

# Categorical dropdowns
home = st.selectbox("Tipo de vivienda", ["other", "owner", "parents", "rent"])
marital = st.selectbox("Estado civil", ["married", "other", "single"])
records = st.selectbox("Historial crediticio", ["no", "yes"])
job = st.selectbox("Tipo de empleo", ["fixed", "freelance", "others", "partime"])
debt = st.selectbox("Tiene deuda actualmente?", ["no", "yes"])  # Binary variable

# Mapping categorical inputs to one-hot encoding
home_mapping = {'other': [1, 0, 0, 0], 'owner': [0, 1, 0, 0], 'parents': [0, 0, 1, 0], 'rent': [0, 0, 0, 1]}
marital_mapping = {'married': [1, 0, 0], 'other': [0, 1, 0], 'single': [0, 0, 1]}
records_mapping = {'no': [1, 0], 'yes': [0, 1]}
job_mapping = {'fixed': [1, 0, 0, 0], 'freelance': [0, 1, 0, 0], 'others': [0, 0, 1, 0], 'partime': [0, 0, 0, 1]}
debt_mapping = {'no': 0, 'yes': 1}  # Binary mapping

# Convert categorical choices into numerical format
home_features = home_mapping[home]
marital_features = marital_mapping[marital]
records_features = records_mapping[records]
job_features = job_mapping[job]
debt_value = debt_mapping[debt]  # Convert debt to 0 or 1

# Prepare data for prediction
columns = ['seniority', 'time', 'age', 'expenses', 'income', 'assets', 'debt', 'amount', 'price',
           'home_other', 'home_owner', 'home_parents', 'home_rent',
           'marital_married', 'marital_other', 'marital_single',
           'records_no', 'records_yes',
           'job_fixed', 'job_freelance', 'job_others', 'job_partime']

# Create DataFrame with user data and one-hot encoded categorical features
user_data = pd.DataFrame([[seniority, time, age, expenses, income, assets, debt_value, amount, price] +
                          home_features + marital_features + records_features + job_features],
                         columns=columns)

# Scale the input data
user_data_scaled = scaler.transform(user_data)


# Predict the credit score
if st.button("📊 Calculando puntaje crediticio"):
    prob = model.predict_proba(user_data_scaled)[0, 1]
    optimal_threshold = 0.28
    prediction = int(prob >= optimal_threshold)
    score = 1000 * (1 - prob)
    st.success(f"🎯 Puntaje crediticio estimado: {score:.0f}")
