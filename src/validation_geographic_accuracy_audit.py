import sys
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PREDICTIONS_DIR,
    METRICS_DIR,
    GEO_TOLERANCES_M,
    VAL_DATES,
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

CONFIGURATIONS = {
    "Baseline": "validation_baseline_predictions.csv",
    "E1": "validation_e1_predictions.csv",
    "E2": "validation_e2_predictions.csv",
    "E3": "validation_e3_predictions.csv",
    "E4": "validation_e4_predictions.csv",
    "E5": "validation_e5_predictions.csv",
    "E6": "validation_e6_predictions.csv",
}

REQUIRED_COLUMNS = [
    "spoofed_id_rental",
    "lat_return",
    "lng_return",
    "lat_returned_true",
    "lng_returned_true",
    "predicted_latitude",
    "predicted_longitude",
]

# --------------------------------------------------
# 2. LOAD COMPLETE RENTAL UNIVERSE
# --------------------------------------------------

coverage_path = METRICS_DIR / "baseline_rental_coverage.csv"

coverage = pd.read_csv(coverage_path)

coverage["date"] = coverage["date"].astype(str)

validation_rentals = coverage[
    coverage["date"].isin(VAL_DATES)
].copy()

total_rentals = len(validation_rentals)

print("Total validation rentals:", total_rentals)

# --------------------------------------------------
# 3. DISTANCE CALCULATION
# --------------------------------------------------

def calculate_distance(lat1, lng1, lat2, lng2):

    coordinates = [lat1, lng1, lat2, lng2]

    if not all(pd.notna(value) for value in coordinates):
        return float("nan")

    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return float("nan")

    if not (-180 <= lng1 <= 180 and -180 <= lng2 <= 180):
        return float("nan")

    return geodesic(
        (lat1, lng1),
        (lat2, lng2)
    ).meters


# --------------------------------------------------
# 4. EVALUATE EACH CONFIGURATION
# --------------------------------------------------

all_results = []
all_details = []

for config_name, filename in CONFIGURATIONS.items():

    print("\n========================================")
    print("Evaluating:", config_name)
    print("========================================")

    file_path = PREDICTIONS_DIR / filename

    predictions = pd.read_csv(file_path)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{config_name} is missing columns: {missing_columns}"
        )

    if predictions["spoofed_id_rental"].duplicated().any():
        raise ValueError(
            f"Duplicate rental IDs found in {filename}"
        )

    # Verify that all predictions belong to validation rentals.
    valid_ids = set(
        validation_rentals["rental_id"].astype(str)
    )

    prediction_ids = set(
        predictions["spoofed_id_rental"].astype(str)
    )

    unexpected_ids = prediction_ids - valid_ids

    if unexpected_ids:
        raise ValueError(
            f"{config_name} contains rentals outside validation dates."
        )

    available_rentals = len(predictions)

    # --------------------------------------------------
    # Calculate actual geographical errors
    # --------------------------------------------------

    predictions["selected_candidate_error_m"] = predictions.apply(
        lambda row: calculate_distance(
            row["lat_return"],
            row["lng_return"],
            row["lat_returned_true"],
            row["lng_returned_true"]
        ),
        axis=1
    )

    predictions["model_coordinate_error_m"] = predictions.apply(
        lambda row: calculate_distance(
            row["predicted_latitude"],
            row["predicted_longitude"],
            row["lat_returned_true"],
            row["lng_returned_true"]
        ),
        axis=1
    )

    predictions["configuration"] = config_name

    # Save individual error details.
    all_details.append(predictions)

    print("Available predictions:", available_rentals)
    print(
        "Candidate availability:",
        f"{available_rentals / total_rentals * 100:.2f}%"
    )

    # --------------------------------------------------
    # Evaluate at each geographical tolerance
    # --------------------------------------------------

    for tolerance in GEO_TOLERANCES_M:

        # Primary metric: selected candidate vs actual destination.
        selected_correct = (
            predictions["selected_candidate_error_m"]
            <= tolerance
        )

        selected_correct_count = int(
            selected_correct.sum()
        )

        selected_conditional = (
            selected_correct_count / available_rentals * 100
            if available_rentals else 0
        )

        selected_end_to_end = (
            selected_correct_count / total_rentals * 100
        )

        # Diagnostic metric: model coordinates vs actual destination.
        model_correct = (
            predictions["model_coordinate_error_m"]
            <= tolerance
        )

        model_correct_count = int(model_correct.sum())

        model_conditional = (
            model_correct_count / available_rentals * 100
            if available_rentals else 0
        )

        model_end_to_end = (
            model_correct_count / total_rentals * 100
        )

        all_results.append({

            "configuration": config_name,
            "tolerance_m": tolerance,

            "total_rentals": total_rentals,
            "available_predictions": available_rentals,

            "candidate_availability_percent":
                available_rentals / total_rentals * 100,

            "selected_candidate_correct_count":
                selected_correct_count,

            "selected_candidate_conditional_accuracy_percent":
                selected_conditional,

            "selected_candidate_end_to_end_accuracy_percent":
                selected_end_to_end,

            "model_coordinate_correct_count":
                model_correct_count,

            "model_coordinate_conditional_accuracy_percent":
                model_conditional,

            "model_coordinate_end_to_end_accuracy_percent":
                model_end_to_end,
        })

        print(f"\nTolerance: {tolerance} metres")

        print(
            "Selected candidate conditional accuracy:",
            f"{selected_conditional:.2f}%"
        )

        print(
            "Selected candidate end-to-end accuracy:",
            f"{selected_end_to_end:.2f}%"
        )

        print(
            "Model coordinate end-to-end accuracy:",
            f"{model_end_to_end:.2f}%"
        )


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

details_df = pd.concat(
    all_details,
    ignore_index=True
)

summary_path = (
    METRICS_DIR / "validation_geographic_accuracy.csv"
)

details_path = (
    PREDICTIONS_DIR / "validation_geographic_errors_all.csv"
)

results_df.to_csv(summary_path, index=False)

details_df.to_csv(details_path, index=False)

print("\n========================================")
print("VALIDATION GEOGRAPHIC AUDIT COMPLETED")
print("========================================")

print("\nSummary:")
print(
    results_df[
        [
            "configuration",
            "tolerance_m",
            "selected_candidate_end_to_end_accuracy_percent",
            "model_coordinate_end_to_end_accuracy_percent",
        ]
    ].to_string(index=False)
)

print("\nSaved summary:", summary_path)
print("Saved detailed predictions:", details_path)
