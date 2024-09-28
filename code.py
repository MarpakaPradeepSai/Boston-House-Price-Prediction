import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('adaboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input features
st.title("Boston Housing Price Prediction")

# Create columns for the first set of inputs (2 per row)
col1, col2 = st.columns(2)

with col1:
    CRIM = st.number_input("CRIM (per capita crime rate by town)", min_value=0.0)
    NOX = st.number_input("NOX (nitric oxides concentration)", min_value=0.0)
    RM = st.number_input("RM (average number of rooms per dwelling)", min_value=0.0)

with col2:
    AGE = st.number_input("AGE (owner-occupied units built before 1940)", min_value=0.0)
    DIS = st.number_input("DIS (distances to Boston employment centers)", min_value=0.0)
    TAX = st.number_input("TAX (full-value property tax rate per $10,000)", min_value=0)

# Create a new row for the next set of inputs
col3, col4 = st.columns(2)

with col3:
    PTRATIO = st.number_input("PTRATIO (pupil-teacher ratio by town)", min_value=0.0)
    LSTAT = st.number_input("LSTAT (percentage of lower status of the population)", min_value=0.0)

with col4:
    B = st.number_input("B (1000(Bk - 0.63)²; Bk = proportion of Black residents)", min_value=0.0)

# Prepare the input data for prediction
input_data = np.array([[CRIM, NOX, RM, AGE, DIS, TAX, PTRATIO, B, LSTAT]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Add custom CSS to change button color without hover or active effect
st.markdown("""
    <style>
    .stButton > button {
        background-color: #007bff; /* Bootstrap primary blue */
        color: white !important; /* Text color */
        border: none;
        transition: none; /* Remove all transitions */
    }
    .stButton > button:focus,
    .stButton > button:active,
    .stButton > button:hover {
        outline: none; /* Remove focus outline */
        background-color: #007bff !important; /* Keep blue color on focus and active */
        color: white !important; /* Keep text color */
    }
    </style>
    """, unsafe_allow_html=True)

# Predict button
if st.button('Predict 🔍'):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Median Value: ${prediction[0] * 1000:.2f}")
