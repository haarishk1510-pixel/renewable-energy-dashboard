import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("solar_data.csv")

# ✅ Correct column names
X = df[["temperature", "irradiance", "hour"]]
y = df["solar_power"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/linear.pkl")

print("✅ Linear model saved successfully")

