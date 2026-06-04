import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib  # This is used to save our trained model to a file

print("⏳ Step 1: Loading Moogsoft historical data...")
df = pd.read_csv("moogsoft_historical_data.csv")

# Convert the text timestamps into actual Python DateTime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("📊 Step 2: Extracting time clues (Feature Engineering)...")
# Extract numeric features the AI can understand
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday is 0, Sunday is 6

# Create "Lag" features (What happened 1 and 2 hours ago?)
df['lag_1'] = df['alert_count'].shift(1)
df['lag_2'] = df['alert_count'].shift(2)

# Drop empty rows caused by shifting/lagging data
df = df.dropna()

# Define what we want to use to predict (X) and what we want to predict (y)
X = df[['hour', 'day_of_week', 'lag_1', 'lag_2']]  # Our clues
y = df['alert_count']                              # The target answer

print("🤖 Step 3: Training the Machine Learning Model...")
# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on our data
model.fit(X, y)

print("💾 Step 4: Saving the model for our dashboard...")
# Save the trained model to a file so our web app can use it later
joblib.dump(model, "moogsoft_forecaster_model.pkl")

print("✅ SUCCESS: 'moogsoft_forecaster_model.pkl' has been created and saved!")