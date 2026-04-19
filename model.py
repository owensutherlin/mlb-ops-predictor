# model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import RANDOM_SEED

np.random.seed(RANDOM_SEED)

FEATURE_COLS = [
    "OPS_mean", "OBP_mean", "SLG_mean", "AVG_mean",
    "K_pct_mean", "BB_pct_mean",
    "exit_velocity_avg_mean", "hard_hit_percent_mean", "Age_mean",
    "OPS_delta", "OBP_delta", "SLG_delta", "AVG_delta",
    "K_pct_delta", "BB_pct_delta",
    "exit_velocity_avg_delta", "hard_hit_percent_delta", "Age_delta",
]

def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["target_OPS"].values
    return X, y

def build_linear_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])

def build_random_forest() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
        )),
    ])

def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Works for both linear regression (coefficients) and random forest."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
    else:
        importance = model.feature_importances_
    return pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importance,
    }).sort_values("importance", ascending=False)