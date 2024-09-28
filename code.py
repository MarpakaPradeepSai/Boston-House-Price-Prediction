import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_filename = 'best_adaboost_model.pkl'
model = joblib.load(model_filename)

# Feature names based on the best features
feature_names = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Function to predict MEDV
def predict_medv(input_data):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the input data
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0]

# Streamlit app layout
st.title("Boston Housing Price Prediction")
st.write("Enter the features to predict the MEDV (Median value of owner-occupied homes in $1000s):")

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    # Predict and display the result
    medv_prediction = predict_medv(input_data)
    st.write(f"The predicted MEDV value is: ${medv_prediction * 1000:.2f}")

# Run the app
if __name__ == '__main__':
    st.run()
