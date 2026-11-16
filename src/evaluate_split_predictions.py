import sys
from pathlib import Path

import pandas as pd

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
predictions = pd.read_csv(
    PREDICTIONS_DIR / "baseline_predictions.csv"
)

# Extract rental date
predictions["date"] = predictions["spoofed_id_rental"].str[:10]

# Identify correct destination
predictions["is_correct_destination"] = (
    predictions["vehicle_id_rental"]
    == predictions["vehicle_id_return"]
)

all_results = []

for split_name, dates in [
    ("Validation", VAL_DATES),
    ("Test", TEST_DATES)
]:

    split_data = predictions[
        predictions["date"].isin(dates)
    ].copy()

    print(f"\n========== {split_name.upper()} RESULTS ==========")
    print("Rentals with candidates:", len(split_data))

    for tolerance in GEO_TOLERANCES_M:

        correct = (
            split_data["is_correct_destination"]
            & (split_data["prediction_distance_m"] <= tolerance)
        )

        accuracy = (
            correct.sum() / len(split_data) * 100
            if len(split_data) > 0 else 0
        )

        print(
            f"{tolerance}m accuracy: {accuracy:.2f}% "
            f"({correct.sum()}/{len(split_data)})"
        )

        all_results.append({
            "split": split_name,
            "tolerance_m": tolerance,
            "correct_predictions": int(correct.sum()),
            "total_predictions": len(split_data),
            "accuracy_percent": accuracy
        })

# Save results
results_df = pd.DataFrame(all_results)

output_path = METRICS_DIR / "baseline_split_accuracy.csv"
results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)