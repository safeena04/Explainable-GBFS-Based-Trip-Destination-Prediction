import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_PATH,
    SCALER_PATH,
    MODEL_FEATURES,
    SCALED_FEATURES,
)

# Load model and scaler
print("Loading model and scaler...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load candidate data
candidate_path = PROJECT_ROOT / "data" / "processed" / "baseline_candidates.csv"
candidates = pd.read_csv(candidate_path)

print("\n========== DATA INFORMATION ==========")
print("Candidate records:", len(candidates))

# Rename candidate dataset columns to match model features
candidates = candidates.rename(columns={
    "lat_rental": "lat_lend",
    "lng_rental": "lng_lend",
    "speed": "mean_speed_distance_based",
    "speed_range_based": "mean_speed_range_based"
})

# Prepare five real candidate records
sample = candidates[MODEL_FEATURES].head(5).copy()

# Convert model input features to floating-point values
sample = sample.astype(float)

# Scale only the six selected features
sample.loc[:, SCALED_FEATURES] = scaler.transform(
    sample[SCALED_FEATURES]
)

# Predict destination coordinates
predictions = model.predict(sample)

print("\n========== VALIDATION RESULTS ==========")
print("Input shape:", sample.shape)
print("Prediction shape:", np.asarray(predictions).shape)

for i, prediction in enumerate(predictions):
    print(f"\nCandidate {i + 1}:")
    print("Predicted latitude:", prediction[0])
    print("Predicted longitude:", prediction[1])

# Check whether predictions are valid coordinates
predictions = np.asarray(predictions)

valid = (
    np.isfinite(predictions).all()
    and np.all((predictions[:, 0] >= -90) & (predictions[:, 0] <= 90))
    and np.all((predictions[:, 1] >= -180) & (predictions[:, 1] <= 180))
)

print("\n========== FINAL CHECK ==========")
print("Are predictions finite and within coordinate ranges?", valid)