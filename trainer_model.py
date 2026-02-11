import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("solar_data.csv")

# Adjust column names if needed
# Example columns: hour, temperature, radiation, power
X = df[["hour", "temperature", "radiation"]]
y = df["power"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model (pickle protocol 4 = safest)
with open("models/solar_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)

print("✅ ML model trained and saved as models/solar_model.pkl")

