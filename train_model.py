import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Load dataset
data = pd.read_csv("solar_data.csv")

X = data[["temperature", "irradiance", "hour"]]
y = data["solar_power"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("Model trained successfully")
print("MAE:", round(mae, 2))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl saved")

