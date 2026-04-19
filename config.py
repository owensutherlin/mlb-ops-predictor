# config.py
import numpy as np

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Season windows
TRAIN_START = 2015   # First season of training data
TRAIN_END   = 2021   # Last season used in training windows
PRED_YEARS  = [2022, 2023, 2024]  # Three-year window to predict 2025
TARGET_YEAR = 2025   # Season we are predicting

# Data filtering
MIN_PA = 250  # Minimum plate appearances to qualify

# Features to use from traditional stats
TRADITIONAL_FEATURES = [
    "AVG", "OBP", "SLG", "BB_pct", "K_pct"
]

# Features to use from Statcast
STATCAST_FEATURES = [
    "exit_velocity_avg", "hard_hit_percent"
]

# Age + delta features are added dynamically in features.py