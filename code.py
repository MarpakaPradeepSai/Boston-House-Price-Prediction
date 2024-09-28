import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load('best_adaboost_model.pkl')

# Streamlit app title
st.title("House Price Prediction App")

# Sidebar inputs
st.sidebar.header("Input Features")

# Function to take user input
def user_input_features():
    CRIM = st.sidebar.number_input('CRIM', min_value=0.0, max_value=100.0, value=0.00632)
    NOX = st.sidebar.number_input('NOX', min_value=0.0, max_value=1.0, value=0.538)
    RM = st.sidebar.number_input('RM', min_value=0.0, max_value=10.0, value=6.575)
    AGE = st.sidebar.number_input('AGE', min_value=0.0, max_value=100.0, value=65.2)
    DIS = st.sidebar.number_input('DIS', min_value=0.0, max_value=12.0, value=4.0900)
    TAX = st.sidebar.number_input('TAX', min_value=0.0, max_value=1000.0, value=296.0)
    PTRATIO = st.sidebar.number_input('PTRATIO', min_value=0.0, max_value=30.0, value=15.3)
    B = st.sidebar.number_input('B', min_value=0.0, max_value=500.0, value=396.90)
    LSTAT = st.sidebar.number_input('LSTAT', min_value=0.0, max_value=40.0, value=4.98)

    # Create a DataFrame with the input data
    data = {
        'CRIM': CRIM,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'B': B,
        'LSTAT': LSTAT
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.write("### Input Features")
st.write(input_df)

# Load the scaler and apply it to the input
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_df)

# Predict the house price using the model
prediction = model.predict(scaled_input)

# Display the prediction
st.write("### Predicted House Price")
st.write(f"House price prediction: **${prediction[0]:,.2f}**")

# Running the streamlit app
if __name__ == "__main__":
    st.write("Ready to Predict!")
