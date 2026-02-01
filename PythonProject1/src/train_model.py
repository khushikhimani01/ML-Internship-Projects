import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ---------------- LOAD & MERGE DATA ---------------- #
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))

dfs = [pd.read_pickle(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("Full dataset shape:", df.shape)

# ---------------- PREPROCESSING ---------------- #
# Drop columns we don’t need for ML
df = df.drop(["TRANSACTION_ID", "TX_DATETIME"], axis=1)

# Separate features & target
X = df.drop("TX_FRAUD", axis=1)
y = df["TX_FRAUD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------- TRAIN MODEL ---------------- #
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"  # helps with imbalance
)
model.fit(X_train, y_train)

# ---------------- EVALUATE ---------------- #
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE MODEL ---------------- #
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as fraud_model.pkl")
