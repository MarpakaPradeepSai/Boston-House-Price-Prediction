import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load('best_adaboost_model.pkl')

# Define the feature names (best features from your analysis)
feature_names = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Create a function to make predictions
def predict_house_price(features):
    # Convert the features to a numpy array
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return prediction[0]

# Streamlit app
def main():
    st.title('House Price Predictor')
    st.write('Enter the following features to predict the median house price:')

    # Create input fields for each feature
    feature_values = []
    for feature in feature_names:
        value = st.number_input(f'Enter {feature}:', step=0.01, format="%.2f")
        feature_values.append(value)

    if st.button('Predict Price'):
        prediction = predict_house_price(feature_values)
        st.success(f'The predicted median house price is: ${prediction:.2f}')

if __name__ == '__main__':
    main()
