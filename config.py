"""
config.py -- single place for paths, constants and experiment settings.

Every other module imports from here, so a number such as a threshold or the
random seed is never typed in two places.

Items marked  [BASELINE]  are copied from the ORIGINAL notebook (main.ipynb, cell 6).
Items marked  [PROPOSED]  are our own choices and must be frozen before test evaluation.
"""
from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_PATH = ROOT_DIR / "ml" / "model.pkl"
SCALER_PATH = ROOT_DIR / "ml" / "scaler.pkl"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "processed_trips.csv"

# ------------------------------------------------------------------ reproducibility
RANDOM_SEED = 42  # the original paper also uses 42

# ------------------------------------------------------------------ raw data (one row = one COMPLETE trip)
RAW_COLUMNS = [
    "vehicle_id", "timestamp_lend", "lng_lend", "lat_lend",
    "timestamp_returned", "lng_returned", "lat_returned",
    "current_range_meters_lend", "pedelec_battery_lend",
    "pedelec_battery_returned", "current_range_meters_returned",
]

# ------------------------------------------------------------------ model interface
# Order matters: the pickled XGBoost model was trained with exactly this column order.
MODEL_FEATURES = [
    "lat_lend", "lng_lend", "time_diff", "battery_diff", "range_diff",
    "distance", "mean_speed_distance_based", "mean_speed_range_based",
]
# The scaler was fitted on these 6 columns only (lat/lng are NOT scaled).
SCALED_FEATURES = [
    "time_diff", "battery_diff", "range_diff", "distance",
    "mean_speed_distance_based", "mean_speed_range_based",
]

# ------------------------------------------------------------------ [BASELINE] candidate filtering
MAX_TRIP_DURATION_S = 3 * 60 * 60   # trips longer than 3 h are dropped while reading the CSV
MAX_TIME_DIFF_S = 1800              # R2: return within 30 min of rental
MAX_BATTERY_DIFF = 10               # R4: battery drop of at most 10 percentage points
MAX_SPEED_KMH = 20                  # R5
MAX_DISTANCE_M = 3000               # R6
# R7 "non-moving" rule. NOTE: the notebook compares time_diff <= 6 in SECONDS,
# the paper says six MINUTES. We reproduce the notebook first (Phase 3), then decide.
NON_MOVING_MAX_TIME = 6
NON_MOVING_MAX_BATTERY_DIFF = 1
NON_MOVING_MAX_DISTANCE_M = 1000

# ------------------------------------------------------------------ distances
OSRM_ENABLED = False                # lightweight mode: geopy geodesic distance only
GEO_TOLERANCES_M = [100, 250, 500, 1000]

# ------------------------------------------------------------------ [PROPOSED] data split (by day)
# The repo has only 7 daily files (2024-03-01 .. 2024-03-07). We split by whole days
# so that no trip appears in two sets. Freeze these in Phase 2, before looking at results.
VAL_DATES = ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"]
TEST_DATES = ["2024-03-05", "2024-03-06", "2024-03-07"]

# ------------------------------------------------------------------ [PROPOSED] coverage-aware selection
# A_MIN is the minimum validation accuracy a filtering configuration must reach.
# It must be set from the VALIDATION baseline in Phase 7 and never changed after seeing test results.
# Minimum acceptable candidate coverage on validation data
C_MIN = 80.0


def ensure_dirs() -> None:
    """Create output folders if they do not exist yet."""
    for d in (DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
