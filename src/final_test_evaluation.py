import sys
from pathlib import Path

import joblib
import pandas as pd
from geopy.distance import geodesic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_PATH,
    SCALER_PATH,
    MODEL_FEATURES,
    SCALED_FEATURES,
    TEST_DATES,
    GEO_TOLERANCES_M,
    METRICS_DIR,
    PREDICTIONS_DIR,
    MAX_BATTERY_DIFF,
    MAX_SPEED_KMH,
)

# --------------------------------------------------
# 1. FROZEN FINAL CONFIGURATION: E11
# --------------------------------------------------

CONFIG = {
    "name": "E11",
    "max_distance": 5000,
    "max_time": 1800
}

COVERAGE_TOLERANCE_M = 1000

print("\n======================================")
print("FINAL TEST EVALUATION: E11")
print("======================================")

# --------------------------------------------------
# 2. LOAD MODEL AND SCALER
# --------------------------------------------------

print("\nLoading model and scaler...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# 3. LOAD EXPANDED TEST CANDIDATES
# --------------------------------------------------

candidates = pd.read_csv(
    PROJECT_ROOT / "data" / "processed"
    / "expanded_candidates.csv"
)

rentals = pd.read_csv(
    METRICS_DIR / "baseline_rental_coverage.csv"
)

test_candidates = candidates[
    candidates["date"].isin(TEST_DATES)
].copy()

test_rentals = rentals[
    rentals["date"].isin(TEST_DATES)
].copy()

total_rentals = len(test_rentals)

print("\n========== TEST DATA ==========")
print("Total test rentals:", total_rentals)
print("Expanded candidate pairs:", len(test_candidates))

if total_rentals == 0:
    raise ValueError("No test rentals found.")

# --------------------------------------------------
# 4. APPLY E11 FILTERING
# --------------------------------------------------

filtered = test_candidates[
    (test_candidates["distance"] <= CONFIG["max_distance"])
    & (test_candidates["time_diff"] <= CONFIG["max_time"])
    & (test_candidates["battery_diff"] <= MAX_BATTERY_DIFF)
    & (test_candidates["speed"] <= MAX_SPEED_KMH)
].copy()

print("\n========== E11 FILTERING ==========")
print("E11 candidate pairs:", len(filtered))

if filtered.empty:
    raise ValueError("No candidates remain after E11 filtering.")

# --------------------------------------------------
# 5. CANDIDATE AVAILABILITY
# --------------------------------------------------

counts = filtered.groupby("spoofed_id_rental").size()

candidate_counts = (
    test_rentals["rental_id"]
    .map(counts)
    .fillna(0)
)

available = int((candidate_counts > 0).sum())

availability = available / total_rentals * 100

average_candidates = candidate_counts.mean()

# --------------------------------------------------
# 6. GEOGRAPHICAL CANDIDATE COVERAGE
# --------------------------------------------------

print("\nCalculating geographical candidate coverage...")

filtered["candidate_true_distance_m"] = filtered.apply(
    lambda row: geodesic(
        (row["lat_return"], row["lng_return"]),
        (row["lat_returned_true"], row["lng_returned_true"])
    ).meters,
    axis=1
)

covered_rental_ids = (
    filtered[
        filtered["candidate_true_distance_m"]
        <= COVERAGE_TOLERANCE_M
    ]["spoofed_id_rental"]
    .unique()
)

coverage = len(covered_rental_ids) / total_rentals * 100

print("\n========== CANDIDATE METRICS ==========")
print(f"Availability: {availability:.2f}%")
print(f"Geographical coverage: {coverage:.2f}%")
print(f"Average candidates: {average_candidates:.2f}")

# --------------------------------------------------
# 7. PREPARE MODEL FEATURES
# --------------------------------------------------

renamed = filtered.rename(columns={
    "lat_rental": "lat_lend",
    "lng_rental": "lng_lend",
    "speed": "mean_speed_distance_based",
    "speed_range_based": "mean_speed_range_based"
})

features = renamed[MODEL_FEATURES].copy().astype(float)

features.loc[:, SCALED_FEATURES] = scaler.transform(
    features[SCALED_FEATURES]
)

# --------------------------------------------------
# 8. PREDICT DESTINATION COORDINATES
# --------------------------------------------------

print("\nPredicting destination coordinates...")

predictions = model.predict(features)

if predictions.ndim != 2 or predictions.shape[1] != 2:
    raise ValueError(
        f"Unexpected model prediction shape: {predictions.shape}"
    )

filtered["predicted_latitude"] = predictions[:, 0]
filtered["predicted_longitude"] = predictions[:, 1]

# --------------------------------------------------
# 9. SELECT BEST CANDIDATE
# --------------------------------------------------

print("Selecting best candidate destinations...")

filtered["prediction_distance_m"] = filtered.apply(
    lambda row: geodesic(
        (
            row["predicted_latitude"],
            row["predicted_longitude"]
        ),
        (
            row["lat_return"],
            row["lng_return"]
        )
    ).meters,
    axis=1
)

indices = (
    filtered.groupby("spoofed_id_rental")[
        "prediction_distance_m"
    ].idxmin()
)

selected = filtered.loc[indices].copy()

# --------------------------------------------------
# 10. CALCULATE ACTUAL GEOGRAPHICAL ERROR
# --------------------------------------------------

selected["selected_candidate_error_m"] = selected.apply(
    lambda row: geodesic(
        (row["lat_return"], row["lng_return"]),
        (row["lat_returned_true"], row["lng_returned_true"])
    ).meters,
    axis=1
)

# Diagnostic only; not used as ground truth.
selected["vehicle_id_match_proxy"] = (
    selected["vehicle_id_rental"]
    == selected["vehicle_id_return"]
)

# --------------------------------------------------
# 11. CALCULATE GEOGRAPHICAL ACCURACY
# --------------------------------------------------

print("\n========== FINAL E11 TEST ACCURACY ==========")

results = []

for tolerance in GEO_TOLERANCES_M:

    correct = int(
        (
            selected["selected_candidate_error_m"]
            <= tolerance
        ).sum()
    )

    conditional = (
        correct / available * 100
        if available else 0
    )

    end_to_end = correct / total_rentals * 100

    print(
        f"{tolerance}m: "
        f"Conditional = {conditional:.2f}%, "
        f"End-to-end = {end_to_end:.2f}%"
    )

    results.append({
        "configuration": "E11",
        "max_distance_m": CONFIG["max_distance"],
        "max_time_s": CONFIG["max_time"],
        "total_rentals": total_rentals,
        "available_rentals": available,
        "candidate_pairs": len(filtered),
        "candidate_availability_percent": availability,
        "candidate_coverage_1000m_percent": coverage,
        "average_candidates": average_candidates,
        "tolerance_m": tolerance,
        "correct_predictions": correct,
        "conditional_accuracy_percent": conditional,
        "end_to_end_accuracy_percent": end_to_end
    })

# --------------------------------------------------
# 12. SAVE RESULTS
# --------------------------------------------------

prediction_path = (
    PREDICTIONS_DIR / "test_e11_predictions.csv"
)

metrics_path = (
    METRICS_DIR / "test_e11_metrics.csv"
)

selected.to_csv(
    prediction_path,
    index=False
)

pd.DataFrame(results).to_csv(
    metrics_path,
    index=False
)

print("\n======================================")
print("E11 TEST EVALUATION COMPLETED")
print("======================================")

print("\nPredictions saved:")
print(prediction_path)

print("\nMetrics saved:")
print(metrics_path)