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

# Create input fields for each feature in sections of three
feature_values = {}
num_features = len(feature_names)

# Create columns for three sections
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:  # Distribute features across columns
        feature_values[feature] = st.number_input(
            f"Enter {feature}:",
            min_value=0.0,
            value=None,  # Start with an empty field
            format="%.6f"  # Keep the float format for input display
        )

# When the user clicks the button, make a prediction
if st.button("Predict"):
    # Check for empty inputs
    if any(value is None for value in feature_values.values()):
        st.error("Please fill in all fields before making a prediction.")
    else:
        # Prepare the input data for the model
        input_data = np.array([[feature_values[feature] for feature in feature_names]])
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the prediction
        st.write(f"The predicted median value of homes is: ${prediction[0] * 1000:.2f}")

# Run the app with `streamlit run app.py` in the terminal
