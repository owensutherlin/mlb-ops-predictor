# evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def evaluate(y_true, y_pred, label: str = "Model") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"{'='*40}")
    return {"label": label, "MAE": mae, "RMSE": rmse}

def print_predictions(names, y_true, y_pred, n=15):
    """Print top N predictions vs actuals sorted by predicted OPS."""
    df = pd.DataFrame({
        "Name":         names,
        "Predicted_OPS": np.round(y_pred, 3),
        "Actual_OPS":    np.round(y_true, 3),
        "Error":         np.round(np.abs(y_pred - y_true), 3),
    }).sort_values("Predicted_OPS", ascending=False)
    print(f"\nTop {n} Predictions vs Actuals:")
    print(df.head(n).to_string(index=False))

def print_2025_predictions(names, y_pred, n=20):
    """Print 2025 predictions (no actuals yet)."""
    df = pd.DataFrame({
        "Name":          names,
        "Predicted_OPS": np.round(y_pred, 3),
    }).sort_values("Predicted_OPS", ascending=False)
    print(f"\n2025 OPS Predictions (Top {n}):")
    print(df.head(n).to_string(index=False))