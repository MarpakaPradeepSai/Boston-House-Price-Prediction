import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('best_adaboost_model.joblib')
scaler = joblib.load('scaler.joblib')

# Create a function to make predictions
def predict(features):
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)

# Streamlit app layout
st.title("Boston Housing Price Prediction")
st.write("Enter the features to predict the MEDV (Median value of owner-occupied homes in $1000s):")

# User input for features
CRIM = st.number_input("CRIM (Per capita crime rate by town)", min_value=0.0)
NOX = st.number_input("NOX (Nitric oxides concentration)", min_value=0.0)
RM = st.number_input("RM (Average number of rooms per dwelling)", min_value=0.0)
AGE = st.number_input("AGE (Proportion of owner-occupied units built prior to 1940)", min_value=0.0)
DIS = st.number_input("DIS (Weighted distances to five Boston employment centres)", min_value=0.0)
TAX = st.number_input("TAX (Full-value property tax rate per $10,000)", min_value=0.0)
PTRATIO = st.number_input("PTRATIO (Pupil-teacher ratio by town)", min_value=0.0)
B = st.number_input("B (1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town)", min_value=0.0)
LSTAT = st.number_input("LSTAT (% lower status of the population)", min_value=0.0)
CHAS = 0  # Default value for unused features
INDUS = 0  # Default value for unused features
RAD = 0  # Default value for unused features
ZN = 0  # Default value for unused features

# Collect features into a DataFrame
features = pd.DataFrame({
    'CRIM': [CRIM],
    'NOX': [NOX],
    'RM': [RM],
    'AGE': [AGE],
    'DIS': [DIS],
    'TAX': [TAX],
    'PTRATIO': [PTRATIO],
    'B': [B],
    'LSTAT': [LSTAT],
    'CHAS': [CHAS],
    'INDUS': [INDUS],
    'RAD': [RAD],
    'ZN': [ZN]
})

# Predict and display the result
if st.button("Predict"):
    try:
        prediction = predict(features)
        st.write(f"The predicted MEDV value is: ${prediction[0] * 1000:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
