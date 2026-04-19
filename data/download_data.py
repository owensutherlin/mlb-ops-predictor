# data/download_data.py
import pandas as pd
import requests
import os
from config import MIN_PA

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def build_savant_url(year: int) -> str:
    """Build Baseball Savant CSV export URL for a given season."""
    return (
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={year}&type=batter&filter=&sort=4&sortDir=desc"
        f"&min={MIN_PA}&selections=player_age,ab,pa,hit,single,double,triple,"
        f"home_run,strikeout,walk,k_percent,bb_percent,batting_avg,"
        f"slg_percent,on_base_percent,on_base_plus_slg,"
        f"exit_velocity_avg,hard_hit_percent"
        f"&chart=false&x=exit_velocity_avg&y=exit_velocity_avg"
        f"&r=no&chartType=beeswarm&csv=true"
    )

def get_season(year: int) -> pd.DataFrame:
    """Download one season from Baseball Savant."""
    url = build_savant_url(year)
    print(f"    Downloading {year}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df["Season"] = year
    return df

def load_all_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Download and concatenate all seasons."""
    print(f"\n[Data] Loading seasons {start_year}–{end_year}...")
    frames = []
    for year in range(start_year, end_year + 1):
        df = get_season(year)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"\n  Raw columns: {list(data.columns)}")
    print(f"  Shape: {data.shape}")
    return data

if __name__ == "__main__":
    df = load_all_data(2015, 2025)
    df.to_csv(os.path.join(DATA_DIR, "all_seasons.csv"), index=False)
    print("\nSaved to data/all_seasons.csv")
    print(df.head())