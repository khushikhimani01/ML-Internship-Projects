import pandas as pd
import pickle

# Load model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example single transaction (replace with input values)
sample = {
    "CUSTOMER_ID": 12345,
    "TERMINAL_ID": 99999,
    "TX_AMOUNT": 120.50,
    "TX_TIME_SECONDS": 45231,
    "TX_TIME_DAYS": 15,
    "TX_FRAUD_SCENARIO": 0
}

df = pd.DataFrame([sample])

prediction = model.predict(df)[0]
label = "FRAUD" if prediction == 1 else "LEGIT"

print("Prediction:", label)
