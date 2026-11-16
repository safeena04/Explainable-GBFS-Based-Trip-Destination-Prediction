import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    GEO_TOLERANCES_M,
    METRICS_DIR,
    PREDICTIONS_DIR,
    VAL_DATES,
    TEST_DATES
)

# Load predictions
prediction_path = PREDICTIONS_DIR / "baseline_predictions.csv"
predictions = pd.read_csv(prediction_path)

print("Loaded predictions:", len(predictions))

# Identify correct destination using vehicle ID matching
predictions["is_correct_destination"] = (
    predictions["vehicle_id_rental"]
    == predictions["vehicle_id_return"]
)

# Evaluate geographical prediction accuracy
print("\n========== BASELINE PREDICTION EVALUATION ==========")

results = []

for tolerance in GEO_TOLERANCES_M:

    correct = (
        predictions["is_correct_destination"]
        & (predictions["prediction_distance_m"] <= tolerance)
    )

    accuracy = correct.sum() / len(predictions) * 100

    results.append({
        "tolerance_m": tolerance,
        "correct_predictions": int(correct.sum()),
        "total_predictions": len(predictions),
        "accuracy_percent": accuracy
    })

    print(
        f"\nTolerance: {tolerance} metres"
        f"\nCorrect predictions: {correct.sum()}"
        f"\nAccuracy: {accuracy:.2f}%"
    )

# Save metrics
metrics_df = pd.DataFrame(results)

output_path = METRICS_DIR / "baseline_prediction_accuracy.csv"
metrics_df.to_csv(output_path, index=False)

print("\nMetrics saved to:", output_path)