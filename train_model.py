# train_model.py (Corrected Version 3)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import numpy as np

print("Starting the model training process...")

# --- 1. Data Loading ---
try:
    # Load the datasets
    weather_df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
    generation_df = pd.read_csv('Plant_1_Generation_Data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Make sure 'Plant_1_Weather_Sensor_Data.csv' and 'Plant_1_Generation_Data.csv' are in the same folder.")
    exit()

# --- 2. Data Preparation ---

# Convert DATE_TIME to datetime objects
# *** THIS SECTION IS NOW CORRECTED ***
# Apply the correct format to each file.
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
generation_df['DATE_TIME'] = pd.to_datetime(generation_df['DATE_TIME'], format='%d-%m-%Y %H:%M')


# Merge the two dataframes on DATE_TIME
df = pd.merge(generation_df, weather_df, on='DATE_TIME', how='inner')

# --- 3. Feature Engineering ---
# Extract useful features from DATE_TIME
df['Month'] = df['DATE_TIME'].dt.month
df['Hour'] = df['DATE_TIME'].dt.hour
df['Minute'] = df['DATE_TIME'].dt.minute

# It's known that solar generation is 0 at night (low irradiation)
# We can also drop rows where IRRADIATION is 0 and AC_POWER is 0
df = df[df['IRRADIATION'] > 0.0]
df = df[df['AC_POWER'] > 0.0]

# --- 4. Data Cleaning ---
# Check for missing values
print(f"Missing values before cleaning:\n{df.isnull().sum()}")
# Drop any remaining rows with missing values (simple approach)
df = df.dropna()

# --- 5. Feature & Target Selection ---
# Define our features (X) and target (y)
# We predict AC_POWER based on weather and time
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'Hour', 'Month']
target = 'AC_POWER'

X = df[features]
y = df[target]

print(f"Using features: {features}")
print(f"Predicting target: {target}")

# --- 6. Model Training ---

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
# n_estimators=100 means it will build 100 "trees"
# random_state=42 ensures you get the same results every time you run
# n_jobs=-1 uses all your computer's processors to speed up training
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("Training the model... This might take a minute.")
# Train the model
model.fit(X_train, y_train)
print("Model trained successfully.")

# --- 7. Model Evaluation ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"  R-squared (R²): {r2:.2f}")
print(f"  Mean Absolute Error (MAE): {mae:.2f} kWh")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} kWh")
print("---------------------------------")
print(f"\nYour model is, on average, {mae:.2f} kWh off from the actual value.")
print(f"The R² score of {r2:.2f} means the model explains {r2*100:.1f}% of the variability in power generation, which is very good!")


# --- 8. Save the Model ---
# We save the trained model to a file named 'solar_model.pkl'
# We will load this file in our web app
with open('solar_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# We also save the list of feature columns
# This is CRITICAL to make sure our app uses the features in the same order
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\nModel and feature columns saved as 'solar_model.pkl' and 'model_columns.pkl'")
print("Training process complete!")