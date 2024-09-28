import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the trained model
model_filename = 'best_adaboost_model.pkl'
model = joblib.load(model_filename)

# Define the feature names based on your best features
feature_names = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Function to make predictions
def predict(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the input data
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)  # Ensure you use the same scaler used in training

    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit app
st.title("Boston Housing Price Prediction")
st.write("Enter the features below to predict the median value of homes:")

# Input fields for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0, format="%.2f")

# Button to predict
if st.button("Predict"):
    prediction = predict(input_data)
    st.write(f"The predicted median value of homes is: ${prediction:.2f}")
