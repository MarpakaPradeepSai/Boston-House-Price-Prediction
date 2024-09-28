# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model_filename = 'best_adaboost_model.pkl'
loaded_model = joblib.load(model_filename)

# Define the feature names
feature_names = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Streamlit app
st.title("Boston Housing Price Prediction")

with st.form("input_form"):
    st.write("Enter the following features:")
    
    # Create input fields for each feature
    crim = st.number_input('CRIM (Per capita crime rate)')
    nox = st.number_input('NOX (Nitrogen oxide level)')
    rm = st.number_input('RM (Average number of rooms)')
    age = st.number_input('AGE (Proportion of owner-occupied units built prior to 1940)')
    dis = st.number_input('DIS (Distance to employment centers)')
    tax = st.number_input('TAX (Full-value property tax rate)')
    ptratio = st.number_input('PTRATIO (Pupil/Teacher ratio)')
    b = st.number_input('B (Proportion of blacks)')
    lstat = st.number_input('LSTAT (Lower status of the population)')

    # Submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[crim, nox, rm, age, dis, tax, ptratio, b, lstat]], columns=feature_names)
    
    # Scale the input data (assuming StandardScaler was used)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    
    # Make prediction
    prediction = loaded_model.predict(input_scaled)
    
    # Display the prediction
    st.write(f"Predicted Median House Price: ${prediction[0]:.2f}")
