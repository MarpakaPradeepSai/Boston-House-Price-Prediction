import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib to save the scaler

# Prepare features and target variable
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale all features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler to a file

# Create an AdaBoost Regressor model
base_model = DecisionTreeRegressor()  # Base estimator for AdaBoost
model = AdaBoostRegressor(estimator=base_model, random_state=42)

# Store RMSE for different numbers of features
rmse_results = []

# Iterate over a range of features to select
for n_features in range(1, X.shape[1] + 1):
    selector = RFE(model, n_features_to_select=n_features)
    selector = selector.fit(X_train_scaled, y_train)
    
    # Get selected features
    selected_features = X.columns[selector.support_]

    # Fit the model with selected features
    model.fit(X_train_scaled[:, selector.support_], y_train)
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_test_scaled[:, selector.support_])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Store the result
    rmse_results.append((n_features, selected_features.tolist(), rmse))

# Find the best number of features based on RMSE
best_n_features, best_features, best_rmse = min(rmse_results, key=lambda x: x[2])

# Results
print("RMSE for different feature sets:")
for n_features, features, rmse in rmse_results:
    print(f"Number of Features: {n_features}, Features: {features}, RMSE: {rmse:.2f}")

print("\nBest Number of Features:", best_n_features)
print("Best Features:", best_features)
print("Best RMSE:", best_rmse)
