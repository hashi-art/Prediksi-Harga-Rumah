"""
Contoh script training menggunakan California Housing dataset.
Simpan pipeline (preprocessor + model) ke models/model.joblib
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import os
import pandas as pd

def main():
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=["MedHouseVal"])
    y = data.frame["MedHouseVal"]

    # Feature names used by API/UI
    feature_names = list(X.columns)

    # Simple pipeline: imputer -> scaler -> model
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "meta": {"name": "california_rf", "version": "0.1"}
    }

    joblib.dump(artifact, "models/model.joblib")
    # Also save feature_names separately for convenience (UI)
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print("Model saved to models/model.joblib")
    # Quick evaluation
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Eval on test: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

if _name_ == "_main_":
    main()