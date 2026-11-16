import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    VAL_DATES,
    TEST_DATES,
    GEO_TOLERANCES_M,
    METRICS_DIR,
)

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

coverage = pd.read_csv(
    METRICS_DIR / "baseline_rental_coverage.csv"
)

predictions = pd.read_csv(
    PROJECT_ROOT / "results" / "predictions"
    / "baseline_predictions.csv"
)

print("Coverage records:", len(coverage))
print("Prediction records:", len(predictions))

# --------------------------------------------------
# 2. Rename and merge datasets
# --------------------------------------------------

predictions = predictions.rename(
    columns={"spoofed_id_rental": "rental_id"}
)

merged = coverage.merge(
    predictions[
        [
            "rental_id",
            "vehicle_id_rental",
            "vehicle_id_return",
            "prediction_distance_m",
        ]
    ],
    on="rental_id",
    how="left",
    validate="one_to_one"
)

# Check whether a prediction was generated
merged["prediction_available"] = (
    merged["prediction_distance_m"].notna()
)

# Check whether the selected destination belongs
# to the same vehicle as the rental
merged["selected_destination_correct"] = (
    merged["vehicle_id_rental"]
    == merged["vehicle_id_return"]
)

print("\nMerged records:", len(merged))

# --------------------------------------------------
# 3. Evaluate validation and test separately
# --------------------------------------------------

all_results = []

splits = {
    "Validation": VAL_DATES,
    "Test": TEST_DATES,
}

for split_name, dates in splits.items():

    data = merged[merged["date"].isin(dates)].copy()

    total_rentals = len(data)

    available_rentals = int(
        data["has_candidate"].sum()
    )

    correct_retained = int(
        data["correct_destination_retained"].sum()
    )

    availability = (
        available_rentals / total_rentals * 100
    )

    coverage_percent = (
        correct_retained / total_rentals * 100
    )

    avg_candidates = data["candidate_count"].mean()

    print(f"\n========== {split_name.upper()} ==========")

    print("Total rentals:", total_rentals)
    print("Rentals with candidates:", available_rentals)
    print("Rentals without candidates:",
          total_rentals - available_rentals)

    print(f"Candidate availability: {availability:.2f}%")
    print(f"Candidate coverage: {coverage_percent:.2f}%")
    print(f"Average candidates per rental: {avg_candidates:.2f}")

    # Prediction accuracy at each geographical tolerance
    for tolerance in GEO_TOLERANCES_M:

        correct_predictions = (
            data["prediction_available"]
            & data["selected_destination_correct"]
            & (data["prediction_distance_m"] <= tolerance)
        )

        correct_count = int(correct_predictions.sum())

        conditional_accuracy = (
            correct_count / available_rentals * 100
            if available_rentals else 0
        )

        end_to_end_accuracy = (
            correct_count / total_rentals * 100
            if total_rentals else 0
        )

        print(f"\nTolerance: {tolerance} metres")
        print(f"Correct predictions: {correct_count}")
        print(
            f"Conditional accuracy: "
            f"{conditional_accuracy:.2f}%"
        )
        print(
            f"End-to-end accuracy: "
            f"{end_to_end_accuracy:.2f}%"
        )

        all_results.append({
            "split": split_name,
            "tolerance_m": tolerance,
            "total_rentals": total_rentals,
            "rentals_with_candidates": available_rentals,
            "candidate_availability_percent": availability,
            "candidate_coverage_percent": coverage_percent,
            "average_candidates": avg_candidates,
            "correct_predictions": correct_count,
            "conditional_accuracy_percent": conditional_accuracy,
            "end_to_end_accuracy_percent": end_to_end_accuracy,
        })

# --------------------------------------------------
# 4. Save results
# --------------------------------------------------

results = pd.DataFrame(all_results)

output_path = METRICS_DIR / "complete_baseline_evaluation.csv"

results.to_csv(output_path, index=False)

print("\n========================================")
print("Complete baseline evaluation saved!")
print(output_path)