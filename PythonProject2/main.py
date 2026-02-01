import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Remove rows with no price (target cannot be null)
df = df.dropna(subset=["price"])

# Select features (X) and target (y)
X = df.drop(columns=["price", "name", "description"])  # removing text-heavy columns
y = df["price"]

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Pipelines for preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print("MAE (Mean Absolute Error):", mae)
print("R² Score:", r2)

# Example prediction using first row
example = X.iloc[[0]]
true_price = y.iloc[0]
predicted_price = model.predict(example)[0]

print("\nExample Prediction:")
print("True Price:", true_price)
print("Predicted Price:", predicted_price)

def predict_vehicle_price(make, model_name, year, engine, cylinders, fuel, mileage, transmission, body, doors, exterior_color, interior_color, drivetrain):
    input_data = {
        "make": [make],
        "model": [model_name],
        "year": [year],
        "engine": [engine],
        "cylinders": [cylinders],
        "fuel": [fuel],
        "mileage": [mileage],
        "transmission": [transmission],
        "trim": [None],
        "body": [body],
        "doors": [doors],
        "exterior_color": [exterior_color],
        "interior_color": [interior_color],
        "drivetrain": [drivetrain]
    }

    df_input = pd.DataFrame(input_data)

    return model.predict(df_input)[0]



result = predict_vehicle_price(
    make="Jeep",
    model_name="Wagoneer",
    year=2024,
    engine=3.0,
    cylinders=6,
    fuel="Gasoline",
    mileage=12000,
    transmission="Automatic",
    body="SUV",
    doors=4,
    exterior_color="Black",
    interior_color="Black",
    drivetrain="Four-wheel Drive"
)

print("\nCustom Vehicle Predicted Price:", result)

import joblib
joblib.dump(model, "vehicle_price_model.pkl")
