import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model
model_filename = 'best_adaboost_model.pkl'
model = joblib.load(model_filename)

# List of features based on your dataset
features = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Title of the app
st.title("Boston Housing Price Prediction")

# Input fields for each feature
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=0.0)

# Button to predict
if st.button("Predict"):
    # Prepare input data for the model
    input_df = pd.DataFrame([input_data])
    
    # Scale the input data using the same scaler used during training
    scaler = StandardScaler()
    # Fit and transform the scaler on the training data (or save and load it if you want)
    # Here we assume it was fit earlier; you might want to save the scaler similarly to the model.
    # scaler.fit(X_train_scaled) # Uncomment this line if you have access to X_train_scaled

    input_scaled = scaler.fit_transform(input_df)  # For demo, you might want to load the scaler instead

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display the result
    st.write(f"Predicted MEDV (House Price): ${prediction[0]:,.2f}")
