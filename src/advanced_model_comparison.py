
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

TRAIN_FILE = "data/processed/advanced_train_data.csv"
TEST_FILE = "data/processed/advanced_test_data.csv"

MODEL_DIR = "models/advanced"
OUTPUT_DIR = "outputs/advanced"

def train_and_compare():

    print("\n===== ADVANCED MODEL COMPARISON =====")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    target = "passengers"

    features = [
        "route", "bus_type", "capacity", "distance_km",
        "year", "month", "day", "day_of_week",
        "week_of_year", "is_weekend", "quarter",
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_mean_7"
    ]

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    categorical_features = ["route", "bus_type"]

    numerical_features = [
        col for col in features
        if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            ),
            (
                "numerical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numerical_features
            )
        ]
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            random_state=42
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
    }

    results = []
    predictions = pd.DataFrame({
        "date": test_df["date"],
        "actual_passengers": y_test
    })

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })

        predictions[name + "_prediction"] = y_pred

        joblib.dump(
            pipeline,
            f"{MODEL_DIR}/{name.lower().replace(' ', '_')}.pkl"
        )

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE")

    results_df.to_csv(
        f"{OUTPUT_DIR}/model_comparison.csv",
        index=False
    )

    predictions.to_csv(
        f"{OUTPUT_DIR}/predictions.csv",
        index=False
    )

    print("\n===== FINAL MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    print("\nBest model by MAE:")
    print(results_df.iloc[0]["Model"])

    print("\nAll models and results saved successfully!")

if __name__ == "__main__":
    train_and_compare()
