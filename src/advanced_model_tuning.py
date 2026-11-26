
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# -------------------------------
# CONFIGURATION
# -------------------------------

TRAIN_FILE = "data/processed/advanced_train_data.csv"
TEST_FILE = "data/processed/advanced_test_data.csv"

OUTPUT_DIR = "outputs/advanced"
MODEL_DIR = "models/advanced/tuned"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "route", "bus_type", "capacity", "distance_km",
    "year", "month", "day", "day_of_week",
    "week_of_year", "is_weekend", "quarter",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_7"
]

CATEGORICAL_FEATURES = ["route", "bus_type"]

NUMERIC_FEATURES = [
    col for col in FEATURES
    if col not in CATEGORICAL_FEATURES
]

# -------------------------------
# LOAD DATA
# -------------------------------

print("\n===== ADVANCED MODEL TUNING =====")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

train_df = train_df.sort_values("date").reset_index(drop=True)
test_df = test_df.sort_values("date").reset_index(drop=True)

if train_df["date"].max() >= test_df["date"].min():
    raise ValueError("Training and testing dates overlap!")

X_train = train_df[FEATURES]
y_train = train_df["passengers"]

X_test = test_df[FEATURES]
y_test = test_df["passengers"]

# -------------------------------
# PREPROCESSING
# -------------------------------

def create_pipeline(model):

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# -------------------------------
# MODELS AND SEARCH SPACES
# -------------------------------

models = {
    "Gradient_Boosting": {
        "pipeline": create_pipeline(
            GradientBoostingRegressor(random_state=42)
        ),
        "parameters": {
            "model__n_estimators": [100, 150, 200, 300],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.7, 0.8, 0.9, 1.0]
        }
    },

    "XGBoost": {
        "pipeline": create_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
        ),
        "parameters": {
            "model__n_estimators": [100, 150, 200, 300],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_lambda": [1, 2, 5, 10]
        }
    }
}

# -------------------------------
# TIME SERIES CROSS VALIDATION
# -------------------------------

cv = TimeSeriesSplit(n_splits=3)

evaluation_results = []
best_models = {}

for model_name, config in models.items():

    print(f"\nTuning {model_name}...")

    search = RandomizedSearchCV(
        estimator=config["pipeline"],
        param_distributions=config["parameters"],
        n_iter=15,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Final untouched test evaluation
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} Results")
    print(f"Best CV MAE: {-search.best_score_:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    print("Best Parameters:", search.best_params_)

    model_path = f"{MODEL_DIR}/{model_name.lower()}.pkl"
    joblib.dump(best_model, model_path)

    best_models[model_name] = best_model

    evaluation_results.append({
        "Model": model_name,
        "CV_MAE": -search.best_score_,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2,
        "Best_Parameters": str(search.best_params_)
    })

# -------------------------------
# SAVE RESULTS
# -------------------------------

results_df = pd.DataFrame(evaluation_results)

results_df.to_csv(
    f"{OUTPUT_DIR}/tuned_model_comparison.csv",
    index=False
)

print("\n===== TUNING SUMMARY =====")
print(
    results_df[
        ["Model", "CV_MAE", "Test_MAE", "Test_RMSE", "Test_R2"]
    ].to_string(index=False)
)

best_row = results_df.loc[results_df["Test_MAE"].idxmin()]

print("\nBest Test Model:", best_row["Model"])
print("Tuning completed successfully!")
