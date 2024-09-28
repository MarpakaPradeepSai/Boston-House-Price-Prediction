import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_filename = 'best_adaboost_model.pkl'
model = joblib.load(model_filename)

# Define the feature names based on the best features you selected
feature_names = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Streamlit app title
st.title("Boston Housing Price Prediction")

# Create input fields for each feature in rows of three
cols = st.columns(3)  # Create three columns
feature_values = {}

for i, feature in enumerate(feature_names):
    if i % 3 == 0:  # Every three features, create a new row
        with cols[i % 3]:
            if feature == 'CRIM':
                feature_values[feature] = st.number_input(f"Enter {feature} (per capita crime rate by town):", min_value=0.0)
            elif feature == 'NOX':
                feature_values[feature] = st.number_input(f"Enter {feature} (nitric oxides concentration):", min_value=0.0)
            elif feature == 'RM':
                feature_values[feature] = st.number_input(f"Enter {feature} (average number of rooms per dwelling):", min_value=0.0)
            elif feature == 'AGE':
                feature_values[feature] = st.number_input(f"Enter {feature} (proportion of owner-occupied units built prior to 1940):", min_value=0.0)
            elif feature == 'DIS':
                feature_values[feature] = st.number_input(f"Enter {feature} (weighted distances to five Boston employment centers):", min_value=0.0)
            elif feature == 'TAX':
                feature_values[feature] = st.number_input(f"Enter {feature} (full-value property tax rate per $10,000):", min_value=0)
            elif feature == 'PTRATIO':
                feature_values[feature] = st.number_input(f"Enter {feature} (pupil-teacher ratio by town):", min_value=0.0)
            elif feature == 'B':
                feature_values[feature] = st.number_input(f"Enter {feature} (1000(Bk - 0.63)^2 where Bk is the proportion of Black residents):", min_value=0.0)
            elif feature == 'LSTAT':
                feature_values[feature] = st.number_input(f"Enter {feature} (% lower status of the population):", min_value=0.0)

# When the user clicks the button, make a prediction
if st.button("Predict"):
    # Prepare the input data for the model
    input_data = np.array([[feature_values[feature] for feature in feature_names]])
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f"The predicted median value of homes is: ${prediction[0] * 1000:.2f}")

# Run the app with `streamlit run app.py` in the terminal
