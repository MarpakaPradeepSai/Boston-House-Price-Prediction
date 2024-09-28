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
    CRIM = st.text_input("CRIM (per capita crime rate by town)", placeholder="Enter value")
    NOX = st.text_input("NOX (nitric oxides concentration)", placeholder="Enter value")
    RM = st.text_input("RM (average number of rooms per dwelling)", placeholder="Enter value")

with col2:
    AGE = st.text_input("AGE (owner-occupied units built before 1940)", placeholder="Enter value")
    DIS = st.text_input("DIS (distances to Boston employment centers)", placeholder="Enter value")
    TAX = st.text_input("TAX (full-value property tax rate per $10,000)", placeholder="Enter value")

# Create a new row for the next set of inputs
col3, col4 = st.columns(2)

with col3:
    PTRATIO = st.text_input("PTRATIO (pupil-teacher ratio by town)", placeholder="Enter value")
    LSTAT = st.text_input("LSTAT (percentage of lower status of the population)", placeholder="Enter value")

with col4:
    B = st.text_input("B (1000(Bk - 0.63)²; Bk = proportion of Black residents)", placeholder="Enter value")

# Prepare the input data for prediction
try:
    input_data = np.array([[float(CRIM), float(NOX), float(RM), float(AGE), float(DIS), 
                             float(TAX), float(PTRATIO), float(B), float(LSTAT)]])
except ValueError:
    input_data = None  # Set to None if conversion fails

# Scale the input data if it's valid
if input_data is not None:
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
    if input_data is not None:
        prediction = model.predict(input_data_scaled)
        median_value = prediction[0]  # No multiplication by 1000
        
        # Display the result in a styled box
        st.markdown(f"""
            <div style="background-color: green; padding: 20px; text-align: center; border-radius: 10px;">
                <h3 style="color: white;"><medium>Predicted Median Value</medium></h3>
                <p style="font-size: 24px; color: white;">${median_value:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Please enter valid numeric values for all fields.")
