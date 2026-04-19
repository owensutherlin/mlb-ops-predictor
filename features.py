# features.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from config import MIN_PA, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and drop rows missing key stats."""
    df = df.copy()
    df = df.rename(columns={
        "last_name, first_name": "name",
        "year":                  "Season",
        "on_base_plus_slg":      "OPS",
        "on_base_percent":       "OBP",
        "slg_percent":           "SLG",
        "batting_avg":           "AVG",
        "k_percent":             "K_pct",
        "bb_percent":            "BB_pct",
        "player_age":            "Age",
    })
    # Drop the duplicate Season column Savant adds
    if df.columns.tolist().count("Season") > 1:
        df = df.loc[:, ~df.columns.duplicated()]

    df = df[df["pa"] >= MIN_PA].copy()
    df = df.dropna(subset=["OPS", "OBP", "SLG", "AVG", "exit_velocity_avg", "hard_hit_percent"])
    return df


def build_windows(df: pd.DataFrame, input_years: list, target_year: int) -> pd.DataFrame:
    """
    Build one row per player using input_years as features
    and target_year OPS as the label.
    Each feature is averaged across the 3 input years,
    plus year-over-year deltas (year3 minus year1).
    """
    df = clean_data(df)

    feature_cols = ["OPS", "OBP", "SLG", "AVG", "K_pct", "BB_pct",
                    "exit_velocity_avg", "hard_hit_percent", "Age"]

    all_years = sorted(input_years) + [target_year]
    df_filtered = df[df["Season"].isin(all_years)].copy()

    records = []
    for player_id, group in df_filtered.groupby("player_id"):
        group = group.sort_values("Season")

        # Player must have all 3 input years + target year
        seasons_present = set(group["Season"].tolist())
        if not all(y in seasons_present for y in all_years):
            continue

        input_rows = group[group["Season"].isin(input_years)].sort_values("Season")
        target_row = group[group["Season"] == target_year].iloc[0]

        row = {
            "player_id": player_id,
            "name":      group["name"].iloc[0],
            "target_OPS": target_row["OPS"],
        }

        # Mean of each feature across the 3 input seasons
        for col in feature_cols:
            vals = input_rows[col].values
            row[f"{col}_mean"] = np.mean(vals)

        # Delta: last input year minus first input year
        for col in feature_cols:
            vals = input_rows[col].values
            row[f"{col}_delta"] = float(vals[-1]) - float(vals[0])

        records.append(row)

    result = pd.DataFrame(records)
    print(f"  Built {len(result)} player windows for {input_years} → {target_year}")
    return result


def build_training_data(df: pd.DataFrame, train_start: int, train_end: int) -> pd.DataFrame:
    """
    Slide a 3-year window through training seasons.
    e.g. [2015,2016,2017]→2018, [2016,2017,2018]→2019, ...
    """
    frames = []
    for target in range(train_start + 3, train_end + 1):
        input_years = [target - 3, target - 2, target - 1]
        window = build_windows(df, input_years, target)
        frames.append(window)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv("data/all_seasons.csv")
    from config import TRAIN_START, TRAIN_END, PRED_YEARS, TARGET_YEAR

    print("\n[Features] Building training windows...")
    train = build_training_data(df, TRAIN_START, TRAIN_END)
    print(f"  Training set shape: {train.shape}")

    print("\n[Features] Building prediction window...")
    pred = build_windows(df, PRED_YEARS, TARGET_YEAR)
    print(f"  Prediction set shape: {pred.shape}")

    print("\nSample training row:")
    print(train.iloc[0])