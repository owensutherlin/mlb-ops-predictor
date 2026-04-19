# main.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import RANDOM_SEED
from features import build_windows
from model import build_linear_model, build_random_forest, get_X_y, get_feature_importance, FEATURE_COLS
from evaluate import evaluate

np.random.seed(RANDOM_SEED)

TRAIN_INPUTS = [2015, 2016, 2017]
TRAIN_TARGET = 2018
os.makedirs("outputs", exist_ok=True)

def save_csv(train, y_pred_lr, y_pred_rf, lr_metrics, rf_metrics, lr_imp, rf_imp):
    # Predictions table
    preds_df = pd.DataFrame({
        "Name":             train["name"].values,
        "Actual_OPS_2018":  np.round(train["target_OPS"].values, 3),
        "LR_Predicted_OPS": np.round(y_pred_lr, 3),
        "LR_Error":         np.round(np.abs(y_pred_lr - train["target_OPS"].values), 3),
        "RF_Predicted_OPS": np.round(y_pred_rf, 3),
        "RF_Error":         np.round(np.abs(y_pred_rf - train["target_OPS"].values), 3),
    }).sort_values("Actual_OPS_2018", ascending=False)
    preds_df.to_csv("outputs/predictions.csv", index=False)

    # Metrics
    pd.DataFrame([lr_metrics, rf_metrics]).to_csv("outputs/metrics.csv", index=False)

    # Feature importances
    lr_imp.to_csv("outputs/lr_feature_importance.csv", index=False)
    rf_imp.to_csv("outputs/rf_feature_importance.csv", index=False)

    print("  CSVs saved to outputs/")
    return preds_df

def plot_actual_vs_predicted(preds_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Actual vs Predicted OPS (2018)", fontsize=15, fontweight="bold")

    for ax, pred_col, label, color in zip(
        axes,
        ["LR_Predicted_OPS", "RF_Predicted_OPS"],
        ["Linear Regression", "Random Forest"],
        ["steelblue", "darkorange"]
    ):
        x = preds_df["Actual_OPS_2018"]
        y = preds_df[pred_col]
        ax.scatter(x, y, alpha=0.6, color=color, edgecolors="white", s=60)

        # Perfect prediction line
        lims = [min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")

        ax.set_xlabel("Actual OPS", fontsize=12)
        ax.set_ylabel("Predicted OPS", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("  Saved: actual_vs_predicted.png")

def plot_error_distribution(preds_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prediction Error Distribution (2018)", fontsize=15, fontweight="bold")

    for ax, col, label, color in zip(
        axes,
        ["LR_Error", "RF_Error"],
        ["Linear Regression", "Random Forest"],
        ["steelblue", "darkorange"]
    ):
        errors = preds_df[col]
        ax.hist(errors, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(errors.mean(), color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean error: {errors.mean():.3f}")
        ax.set_xlabel("Absolute Error", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/error_distribution.png", dpi=150)
    plt.close()
    print("  Saved: error_distribution.png")

def plot_feature_importance(lr_imp, rf_imp):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature Importance", fontsize=15, fontweight="bold")

    for ax, imp, label, color in zip(
        axes,
        [lr_imp.head(10), rf_imp.head(10)],
        ["Linear Regression (|coefficient|)", "Random Forest (impurity)"],
        ["steelblue", "darkorange"]
    ):
        ax.barh(imp["feature"][::-1], imp["importance"][::-1],
                color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("  Saved: feature_importance.png")

def plot_top_worst(preds_df, n=10):
    """Bar chart of best and worst predicted players for both models."""
    for model_label, err_col, color_good, color_bad, filename in [
        ("Linear Regression", "LR_Error", "seagreen",   "crimson",    "best_worst_lr.png"),
        ("Random Forest",     "RF_Error", "dodgerblue",  "darkorange", "best_worst_rf.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"{model_label}: Best & Worst Predictions (2018)",
                     fontsize=14, fontweight="bold")

        best  = preds_df.nsmallest(n, err_col)
        worst = preds_df.nlargest(n, err_col)

        for ax, subset, label, color in zip(
            axes,
            [best, worst],
            [f"Top {n} Most Accurate", f"Top {n} Least Accurate"],
            [color_good, color_bad]
        ):
            ax.barh(subset["Name"][::-1], subset[err_col][::-1],
                    color=color, edgecolor="white", alpha=0.85)
            ax.set_xlabel("Absolute Error", fontsize=12)
            ax.set_title(label, fontsize=13)
            ax.tick_params(axis="y", labelsize=9)

        plt.tight_layout()
        plt.savefig(f"outputs/{filename}", dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

def plot_ops_distribution(preds_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("OPS Distribution: Actual vs Predicted (2018)", fontsize=14, fontweight="bold")

    ax.hist(preds_df["Actual_OPS_2018"],  bins=25, alpha=0.6, color="gray",
            edgecolor="white", label="Actual OPS")
    ax.hist(preds_df["LR_Predicted_OPS"], bins=25, alpha=0.6, color="steelblue",
            edgecolor="white", label="LR Predicted")
    ax.hist(preds_df["RF_Predicted_OPS"], bins=25, alpha=0.6, color="darkorange",
            edgecolor="white", label="RF Predicted")

    ax.set_xlabel("OPS", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/ops_distribution.png", dpi=150)
    plt.close()
    print("  Saved: ops_distribution.png")

def plot_residuals(preds_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residuals vs Actual OPS (2018)", fontsize=14, fontweight="bold")

    for ax, pred_col, label, color in zip(
        axes,
        ["LR_Predicted_OPS", "RF_Predicted_OPS"],
        ["Linear Regression", "Random Forest"],
        ["steelblue", "darkorange"]
    ):
        residuals = preds_df[pred_col] - preds_df["Actual_OPS_2018"]
        ax.scatter(preds_df["Actual_OPS_2018"], residuals,
                   alpha=0.6, color=color, edgecolors="white", s=60)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Actual OPS", fontsize=12)
        ax.set_ylabel("Residual (Predicted − Actual)", fontsize=12)
        ax.set_title(label, fontsize=13)

    plt.tight_layout()
    plt.savefig("outputs/residuals.png", dpi=150)
    plt.close()
    print("  Saved: residuals.png")

def plot_age_vs_error(train, preds_df):
    ages = train[["name", "Age_mean"]].copy()
    ages = ages.rename(columns={"name": "Name"})  # match preds_df column
    merged = preds_df.merge(ages, on="Name", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Player Age vs Prediction Error (2018)", fontsize=14, fontweight="bold")

    for ax, err_col, label, color in zip(
        axes,
        ["LR_Error", "RF_Error"],
        ["Linear Regression", "Random Forest"],
        ["steelblue", "darkorange"]
    ):
        ax.scatter(merged["Age_mean"], merged[err_col],
                   alpha=0.6, color=color, edgecolors="white", s=60)
        # Trend line
        z = np.polyfit(merged["Age_mean"].dropna(), 
                       merged.loc[merged["Age_mean"].notna(), err_col], 1)
        p = np.poly1d(z)
        xs = np.linspace(merged["Age_mean"].min(), merged["Age_mean"].max(), 100)
        ax.plot(xs, p(xs), "k--", linewidth=1.2, label="Trend")
        ax.set_xlabel("Average Age (2015-17)", fontsize=12)
        ax.set_ylabel("Absolute Error", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/age_vs_error.png", dpi=150)
    plt.close()
    print("  Saved: age_vs_error.png")

def plot_all_players(preds_df):
    for label, key, color_good, color_bad in [
        ("Linear Regression", "LR",  "seagreen", "crimson"),
        ("Random Forest",     "RF",  "dodgerblue", "darkorange"),
    ]:
        err_col  = f"{key}_Error"
        pred_col = f"{key}_Predicted_OPS"
        df = preds_df.sort_values("Actual_OPS_2018", ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(20, len(df) * 0.28))
        fig.suptitle(f"{label}: All Player Predictions vs Actual OPS (2018)",
                     fontsize=14, fontweight="bold")

        y_pos = np.arange(len(df))

        # Draw connecting line
        for i, row in df.iterrows():
            actual = row["Actual_OPS_2018"]
            pred   = row[pred_col]
            color  = color_good if abs(actual - pred) <= 0.05 else color_bad
            ax.plot([actual, pred], [i, i], color=color, alpha=0.6, linewidth=1.2)

        # Actual OPS dots + labels
        ax.scatter(df["Actual_OPS_2018"], y_pos,
                   color="black", zorder=5, s=25, label="Actual OPS")
        for i, val in enumerate(df["Actual_OPS_2018"]):
            ax.text(val, i + 0.18, f"{val:.3f}", ha="center",
                    fontsize=5.5, color="black")

        # Predicted OPS dots + labels
        dot_colors = [color_good if e <= 0.05 else color_bad for e in df[err_col]]
        ax.scatter(df[pred_col], y_pos,
                   color=dot_colors, zorder=5, s=25, alpha=0.8, label="Predicted OPS")
        for i, (val, c) in enumerate(zip(df[pred_col], dot_colors)):
            ax.text(val, i - 0.28, f"{val:.3f}", ha="center",
                    fontsize=5.5, color=c)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Name"], fontsize=7)
        ax.set_xlabel("OPS", fontsize=12)
        ax.axvline(df["Actual_OPS_2018"].mean(), color="gray",
                   linestyle="--", linewidth=1, alpha=0.5, label="League avg")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        filename = f"outputs/all_players_{key.lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: all_players_{key.lower()}.png")

def main():
    print("=" * 50)
    print("  MLB OPS Predictor")
    print("=" * 50)

    print("\n[1] Loading data...")
    df = pd.read_csv("data/all_seasons.csv")

    print("\n[2] Building features (2015-2017 → 2018)...")
    train = build_windows(df, TRAIN_INPUTS, TRAIN_TARGET)
    print(f"  Samples: {len(train)}")
    X, y = get_X_y(train)

    print("\n[3] Training Linear Regression...")
    lr = build_linear_model()
    lr.fit(X, y)
    lr_preds   = lr.predict(X)
    lr_metrics = evaluate(y, lr_preds, "Linear Regression (2015-17 → 2018)")
    lr_imp     = get_feature_importance(lr)

    print("\n[4] Training Random Forest...")
    rf = build_random_forest()
    rf.fit(X, y)
    rf_preds   = rf.predict(X)
    rf_metrics = evaluate(y, rf_preds, "Random Forest (2015-17 → 2018)")
    rf_imp     = get_feature_importance(rf)

    print("\n[5] Saving CSVs...")
    preds_df = save_csv(train, lr_preds, rf_preds, lr_metrics, rf_metrics, lr_imp, rf_imp)

    print("\n[6] Generating plots...")
    plot_actual_vs_predicted(preds_df)
    plot_error_distribution(preds_df)
    plot_feature_importance(lr_imp, rf_imp)
    plot_top_worst(preds_df)
    plot_ops_distribution(preds_df)
    plot_residuals(preds_df)
    plot_age_vs_error(train, preds_df)
    plot_all_players(preds_df)

    print("\n" + "="*50)
    print("  FINAL RESULTS SUMMARY")
    print("="*50)
    for m in [lr_metrics, rf_metrics]:
        print(f"  {m['label']}")
        print(f"    MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")
    print("\n  All outputs saved to outputs/")

if __name__ == "__main__":
    main()