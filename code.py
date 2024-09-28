import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('adaboost_model.pkl')

# Define the input features
st.title("Boston Housing Price Prediction")
CRIM = st.number_input("CRIM (per capita crime rate by town)")
NOX = st.number_input("NOX (nitric oxides concentration)")
RM = st.number_input("RM (average number of rooms per dwelling)")
AGE = st.number_input("AGE (proportion of owner-occupied units built prior to 1940)")
DIS = st.number_input("DIS (weighted distances to five Boston employment centers)")
TAX = st.number_input("TAX (full-value property tax rate per $10,000)")
PTRATIO = st.number_input("PTRATIO (pupil-teacher ratio by town)")
B = st.number_input("B (1000(Bk - 0.63)^2 where Bk is the proportion of Black residents)")
LSTAT = st.number_input("LSTAT (percentage of lower status of the population)")

# Prepare the input data for prediction
input_data = np.array([[CRIM, NOX, RM, AGE, DIS, TAX, PTRATIO, B, LSTAT]])

# Scale the input data
scaler = joblib.load('scaler.pkl')  # Save the scaler during training and load here
input_data_scaled = scaler.transform(input_data)

# Predict the median value of homes
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Median Value: ${prediction[0] * 1000:.2f}")
