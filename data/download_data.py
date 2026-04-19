# data/download_data.py
import pandas as pd
from pybaseball import batting_stats, statcast_batter_exitvelo_barrels, cache
from config import MIN_PA, RANDOM_SEED

cache.enable()  # Caches API calls so we don't re-pull every run

def get_traditional_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """Pull traditional + FanGraphs rate stats for a range of seasons."""
    print(f"  Fetching traditional stats {start_year}–{end_year}...")
    frames = []
    for year in range(start_year, end_year + 1):
        df = batting_stats(year, qual=MIN_PA)
        df["Season"] = year
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    # Keep only what we need
    cols = ["Name", "IDfg", "Season", "PA", "AVG", "OBP", "SLG", "OPS",
            "BB%", "K%", "Age"]
    data = data[[c for c in cols if c in data.columns]]
    data = data.rename(columns={"BB%": "BB_pct", "K%": "K_pct", "IDfg": "player_id"})
    return data


def get_statcast_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """Pull Statcast exit velo / hard-hit data for a range of seasons."""
    print(f"  Fetching Statcast stats {start_year}–{end_year}...")
    frames = []
    for year in range(start_year, end_year + 1):
        df = statcast_batter_exitvelo_barrels(year, minBBE=50)
        df["Season"] = year
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    cols = ["player_id", "Season", "exit_velocity_avg", "hard_hit_percent"]
    data = data[[c for c in cols if c in data.columns]]
    return data


def load_all_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Merge traditional and Statcast data into one DataFrame."""
    print(f"\n[Data] Loading seasons {start_year}–{end_year}...")
    trad = get_traditional_stats(start_year, end_year)
    stat = get_statcast_stats(start_year, end_year)

    # Statcast uses int player_id, FanGraphs uses int too — align types
    trad["player_id"] = trad["player_id"].astype(int)
    stat["player_id"] = stat["player_id"].astype(int)

    merged = pd.merge(trad, stat, on=["player_id", "Season"], how="left")
    print(f"  Merged shape: {merged.shape}")
    return merged


if __name__ == "__main__":
    df = load_all_data(2015, 2025)
    df.to_csv("data/all_seasons.csv", index=False)
    print("\nSaved to data/all_seasons.csv")
    print(df.head())