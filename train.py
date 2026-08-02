"""
train.py
--------
Car Selling Price Prediction - Training Pipeline
Dataset: cardekho.csv

Steps:
1. Load data
2. Preprocess (nulls, duplicates, feature cleanup)
3. Encode categorical columns (LabelEncoder)
4. Scale numeric features (StandardScaler)
5. Train + tune top 3 regression models (RandomForest, GradientBoosting, XGBoost)
6. Evaluate for overfitting (train vs test metrics) + plots
7. Save best model + scaler + encoders for the Streamlit app
"""

import warnings
warnings.filterwarnings("ignore")

import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
DATA_PATH = "cardekho.csv"
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------

# --- 2a. Drop duplicates ---
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Dropped {before - df.shape[0]} duplicate rows")

# --- 2b. Clean max_power / mileage / engine (they carry units as strings in
# the raw CarDekho dump; here they already arrive numeric-like strings, but
# we coerce defensively in case of stray text e.g. "bhp") ---
def to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
    )

df["max_power"] = to_numeric(df["max_power"])
df["engine"] = to_numeric(df["engine"])
df["mileage(km/ltr/kg)"] = to_numeric(df["mileage(km/ltr/kg)"])

# --- 2c. Fill null values (numeric -> median, keeps outliers from skewing it) ---
num_cols_with_na = ["mileage(km/ltr/kg)", "engine", "max_power", "seats"]
for col in num_cols_with_na:
    df[col] = df[col].fillna(df[col].median())

print("Nulls after imputation:\n", df.isnull().sum())

# --- 2d. Feature engineering / drop unnecessary features ---
# 'name' has 2000+ unique values (exact trims), which is too high-cardinality
# to encode directly and mostly duplicates info already in year/engine/power.
# We extract just the brand (first word) which is genuinely useful signal,
# then drop the raw 'name' column.
df["brand"] = df["name"].str.split().str[0]
df.drop(columns=["name"], inplace=True)

# Optional but common transform: convert year -> car age (more stable signal)
CURRENT_YEAR = 2026
df["car_age"] = CURRENT_YEAR - df["year"]
df.drop(columns=["year"], inplace=True)

# --- 2e. Drop obvious outliers in target (helps both training and plots) ---
before = df.shape[0]
q_low, q_high = df["selling_price"].quantile([0.01, 0.99])
df = df[(df["selling_price"] >= q_low) & (df["selling_price"] <= q_high)]
print(f"Dropped {before - df.shape[0]} extreme price outliers (1st/99th pct clip)")

# ---------------------------------------------------------------------------
# 3. ENCODE CATEGORICAL COLUMNS (LabelEncoder, saved per-column for the app)
# ---------------------------------------------------------------------------
categorical_cols = ["fuel", "seller_type", "transmission", "owner", "brand"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "label_encoders.pkl")
print("Saved label_encoders.pkl")

# ---------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT + SCALING
# ---------------------------------------------------------------------------
X = df.drop(columns=["selling_price"])
y = df["selling_price"]

FEATURE_ORDER = X.columns.tolist()
joblib.dump(FEATURE_ORDER, "feature_order.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("Saved scaler.pkl")

# ---------------------------------------------------------------------------
# 5. TOP 3 REGRESSION MODELS + HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------
model_configs = {
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [200, 300, 400],
            "max_depth": [8, 12, 16, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [150, 250, 350],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 0.9, 1.0],
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(random_state=42, objective="reg:squarederror"),
        "params": {
            "n_estimators": [200, 300, 400],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 6],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        },
    },
}

results = {}
fitted_models = {}

for name, cfg in model_configs.items():
    print(f"\nTuning {name} ...")
    search = RandomizedSearchCV(
        estimator=cfg["estimator"],
        param_distributions=cfg["params"],
        n_iter=10,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train_scaled, y_train)
    best_model = search.best_estimator_
    fitted_models[name] = best_model

    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    results[name] = {
        "best_params": search.best_params_,
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }

    print(f"{name} best params: {search.best_params_}")
    print(
        f"{name} -> Train R2: {results[name]['train_r2']:.4f} | "
        f"Test R2: {results[name]['test_r2']:.4f} | "
        f"Gap: {results[name]['train_r2'] - results[name]['test_r2']:.4f}"
    )

# ---------------------------------------------------------------------------
# 6. PICK BEST MODEL (highest test R2, penalize large train/test gap = overfit)
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(
    {
        name: {
            "Train R2": r["train_r2"],
            "Test R2": r["test_r2"],
            "Overfit Gap (Train-Test R2)": r["train_r2"] - r["test_r2"],
            "Train MAE": r["train_mae"],
            "Test MAE": r["test_mae"],
            "Train RMSE": r["train_rmse"],
            "Test RMSE": r["test_rmse"],
        }
        for name, r in results.items()
    }
).T
print("\n=== Model Comparison ===")
print(summary_df.round(4))
summary_df.to_csv("model_comparison.csv")

best_model_name = summary_df["Test R2"].idxmax()
best_model = fitted_models[best_model_name]
print(f"\nBest model selected: {best_model_name}")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_model_name, "best_model_name.pkl")
print("Saved best_model.pkl")

# Save ALL three tuned models + their metrics so the app can offer a choice
joblib.dump(fitted_models, "all_models.pkl")
metrics_for_app = {
    name: {
        "train_r2": float(r["train_r2"]),
        "test_r2": float(r["test_r2"]),
        "train_mae": float(r["train_mae"]),
        "test_mae": float(r["test_mae"]),
        "train_rmse": float(r["train_rmse"]),
        "test_rmse": float(r["test_rmse"]),
    }
    for name, r in results.items()
}
joblib.dump(metrics_for_app, "model_metrics.pkl")
print("Saved all_models.pkl, model_metrics.pkl")

# ---------------------------------------------------------------------------
# 7. VISUALIZATIONS (overfitting checks)
# ---------------------------------------------------------------------------

# 7a. Train vs Test R2 bar chart across all 3 models (overfit at a glance)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results))
width = 0.35
ax.bar(x - width / 2, summary_df["Train R2"], width, label="Train R2")
ax.bar(x + width / 2, summary_df["Test R2"], width, label="Test R2")
ax.set_xticks(x)
ax.set_xticklabels(summary_df.index)
ax.set_ylabel("R2 Score")
ax.set_title("Train vs Test R2 by Model (large gap = overfitting)")
ax.legend()
plt.tight_layout()
plt.savefig("plot_train_vs_test_r2.png", dpi=150)
plt.close()

# 7b. Actual vs Predicted scatter (train + test) for each model
fig, axes = plt.subplots(len(results), 2, figsize=(12, 5 * len(results)))
for i, (name, r) in enumerate(results.items()):
    axes[i, 0].scatter(y_train, r["y_train_pred"], alpha=0.3, s=10)
    axes[i, 0].plot(
        [y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--"
    )
    axes[i, 0].set_title(f"{name} - Train: Actual vs Predicted")
    axes[i, 0].set_xlabel("Actual")
    axes[i, 0].set_ylabel("Predicted")

    axes[i, 1].scatter(y_test, r["y_test_pred"], alpha=0.3, s=10, color="orange")
    axes[i, 1].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--"
    )
    axes[i, 1].set_title(f"{name} - Test: Actual vs Predicted")
    axes[i, 1].set_xlabel("Actual")
    axes[i, 1].set_ylabel("Predicted")
plt.tight_layout()
plt.savefig("plot_actual_vs_predicted.png", dpi=150)
plt.close()

# 7c. Residual plots (test set) - random scatter around 0 = good fit,
# funnel/pattern shapes = model struggling
fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
if len(results) == 1:
    axes = [axes]
for ax, (name, r) in zip(axes, results.items()):
    residuals = y_test - r["y_test_pred"]
    ax.scatter(r["y_test_pred"], residuals, alpha=0.3, s=10)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"{name} - Residuals (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig("plot_residuals.png", dpi=150)
plt.close()

# 7d. Feature importance for the best model (if tree-based)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURE_ORDER)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax)
    ax.set_title(f"Feature Importance - {best_model_name}")
    plt.tight_layout()
    plt.savefig("plot_feature_importance.png", dpi=150)
    plt.close()

# 7e. Individual per-model plot (train + test Actual vs Predicted side by side)
# -- used on the Streamlit model-selection page so each model has its own image
for name, r in results.items():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_train, r["y_train_pred"], alpha=0.3, s=10)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
    axes[0].set_title(f"{name} - Train")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")

    axes[1].scatter(y_test, r["y_test_pred"], alpha=0.3, s=10, color="orange")
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[1].set_title(f"{name} - Test")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"plot_{name}.png", dpi=150)
    plt.close()

print("\nSaved plots: plot_train_vs_test_r2.png, plot_actual_vs_predicted.png, "
      "plot_residuals.png, plot_feature_importance.png, plot_<ModelName>.png (x3)")
print("\nTraining complete. Files ready for the Streamlit app:")
print(" - best_model.pkl, best_model_name.pkl, all_models.pkl, model_metrics.pkl,")
print(" - scaler.pkl, label_encoders.pkl, feature_order.pkl")