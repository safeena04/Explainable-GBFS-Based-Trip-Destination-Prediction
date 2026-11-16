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
    VAL_DATES,
    GEO_TOLERANCES_M,
    METRICS_DIR,
    PREDICTIONS_DIR,
    MAX_BATTERY_DIFF,
    MAX_SPEED_KMH,
)

# --------------------------------------------------
# 1. NEW EXPERIMENT CONFIGURATIONS
# --------------------------------------------------

EXPERIMENT_CONFIGS = {
    "E7": {"max_distance": 3000, "max_time": 1350},
    "E8": {"max_distance": 3000, "max_time": 1500},
    "E9": {"max_distance": 3000, "max_time": 1650},
}

# --------------------------------------------------
# 2. LOAD MODEL AND VALIDATION DATA
# --------------------------------------------------

print("Loading model and scaler...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

candidates = pd.read_csv(
    PROJECT_ROOT / "data" / "processed"
    / "baseline_candidates.csv"
)

rental_coverage = pd.read_csv(
    METRICS_DIR / "baseline_rental_coverage.csv"
)

validation_candidates = candidates[
    candidates["date"].astype(str).isin(VAL_DATES)
].copy()

validation_rentals = rental_coverage[
    rental_coverage["date"].astype(str).isin(VAL_DATES)
].copy()

total_rentals = len(validation_rentals)

print("Validation rentals:", total_rentals)
print("Validation candidate pairs:", len(validation_candidates))

# --------------------------------------------------
# 3. PREPARE MODEL FEATURES
# --------------------------------------------------

def prepare_features(data):

    renamed = data.rename(columns={
        "lat_rental": "lat_lend",
        "lng_rental": "lng_lend",
        "speed": "mean_speed_distance_based",
        "speed_range_based": "mean_speed_range_based"
    })

    features = renamed[MODEL_FEATURES].copy()
    features = features.astype(float)

    features.loc[:, SCALED_FEATURES] = scaler.transform(
        features[SCALED_FEATURES]
    )

    return features


# --------------------------------------------------
# 4. FILTER CANDIDATES
# --------------------------------------------------

def filter_candidates(data, config):

    return data[
        (data["distance"] <= config["max_distance"])
        & (data["time_diff"] <= config["max_time"])
        & (data["battery_diff"] <= MAX_BATTERY_DIFF)
        & (data["speed"] <= MAX_SPEED_KMH)
    ].copy()


# --------------------------------------------------
# 5. PREDICT DESTINATION COORDINATES
# --------------------------------------------------

def predict_destinations(data):

    if data.empty:
        return data.copy()

    features = prepare_features(data)

    predictions = model.predict(features)

    results = data.copy()

    results["predicted_latitude"] = predictions[:, 0]
    results["predicted_longitude"] = predictions[:, 1]

    return results


# --------------------------------------------------
# 6. CALCULATE MODEL-TO-CANDIDATE DISTANCE
# --------------------------------------------------

def calculate_selection_distances(data):

    if data.empty:
        return data.copy()

    distances = []

    for _, row in data.iterrows():

        predicted = (
            row["predicted_latitude"],
            row["predicted_longitude"]
        )

        candidate_return = (
            row["lat_return"],
            row["lng_return"]
        )

        distances.append(
            geodesic(predicted, candidate_return).meters
        )

    results = data.copy()
    results["prediction_distance_m"] = distances

    return results


# --------------------------------------------------
# 7. SELECT BEST CANDIDATE
# --------------------------------------------------

def select_best_destinations(data):

    if data.empty:
        return data.copy()

    indices = (
        data.groupby("spoofed_id_rental")[
            "prediction_distance_m"
        ].idxmin()
    )

    return data.loc[indices].copy()


# --------------------------------------------------
# 8. CALCULATE ACTUAL GEOGRAPHICAL ERROR
# --------------------------------------------------

def calculate_actual_geographic_error(data):

    if data.empty:
        return data.copy()

    errors = []

    for _, row in data.iterrows():

        selected_location = (
            row["lat_return"],
            row["lng_return"]
        )

        actual_location = (
            row["lat_returned_true"],
            row["lng_returned_true"]
        )

        errors.append(
            geodesic(
                selected_location,
                actual_location
            ).meters
        )

    results = data.copy()
    results["selected_candidate_error_m"] = errors

    return results


# --------------------------------------------------
# 9. EVALUATE ONE EXPERIMENT
# --------------------------------------------------

def evaluate_experiment(name, config):

    print("\n========================================")
    print("Running:", name)
    print("========================================")

    filtered = filter_candidates(
        validation_candidates,
        config
    )

    # Candidate availability
    candidate_counts = (
        filtered.groupby("spoofed_id_rental").size()
    )

    rental_ids = validation_rentals["rental_id"].astype(str)

    candidate_counts = (
        rental_ids.map(candidate_counts)
        .fillna(0)
    )

    available_rentals = int(
        (candidate_counts > 0).sum()
    )

    # Candidate coverage using vehicle-ID matching proxy
    if filtered.empty:
        correct_rentals = 0
    else:
        correct_mask = (
            filtered["vehicle_id_rental"]
            == filtered["vehicle_id_return"]
        )

        correct_rentals = int(
            filtered.loc[
                correct_mask, "spoofed_id_rental"
            ].nunique()
        )

    availability = (
        available_rentals / total_rentals * 100
    )

    coverage = (
        correct_rentals / total_rentals * 100
    )

    average_candidates = candidate_counts.mean()

    # Predict and select
    predicted = predict_destinations(filtered)

    predicted = calculate_selection_distances(predicted)

    selected = select_best_destinations(predicted)

    # Calculate actual geographical error
    selected = calculate_actual_geographic_error(selected)

    # Save predictions separately
    prediction_path = (
        PREDICTIONS_DIR
        / f"validation_{name.lower()}_predictions.csv"
    )

    selected.to_csv(prediction_path, index=False)

    print("Candidate pairs:", len(filtered))
    print("Available rentals:", available_rentals)
    print(f"Availability: {availability:.2f}%")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Average candidates: {average_candidates:.2f}")

    # Evaluate geographical accuracy
    results = []

    for tolerance in GEO_TOLERANCES_M:

        if selected.empty:
            correct_count = 0
        else:
            correct_count = int(
                (
                    selected["selected_candidate_error_m"]
                    <= tolerance
                ).sum()
            )

        conditional_accuracy = (
            correct_count / available_rentals * 100
            if available_rentals else 0
        )

        end_to_end_accuracy = (
            correct_count / total_rentals * 100
        )

        print(
            f"{tolerance}m geographical accuracy: "
            f"{conditional_accuracy:.2f}% conditional, "
            f"{end_to_end_accuracy:.2f}% end-to-end"
        )

        results.append({

            "configuration": name,
            "max_distance_m": config["max_distance"],
            "max_time_s": config["max_time"],

            "total_rentals": total_rentals,
            "candidate_pairs": len(filtered),

            "available_rentals": available_rentals,
            "candidate_availability_percent": availability,

            "correct_rentals_retained": correct_rentals,
            "candidate_coverage_percent": coverage,

            "average_candidates": average_candidates,

            "tolerance_m": tolerance,
            "correct_predictions": correct_count,

            "conditional_accuracy_percent":
                conditional_accuracy,

            "end_to_end_accuracy_percent":
                end_to_end_accuracy
        })

    return results


# --------------------------------------------------
# 10. RUN E7, E8 AND E9
# --------------------------------------------------

all_results = []

for name, config in EXPERIMENT_CONFIGS.items():

    results = evaluate_experiment(name, config)

    all_results.extend(results)


# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

output_path = (
    METRICS_DIR / "extended_validation_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\n========================================")
print("EXTENDED VALIDATION EXPERIMENTS COMPLETED")
print("========================================")

print(
    results_df[
        [
            "configuration",
            "candidate_pairs",
            "candidate_availability_percent",
            "candidate_coverage_percent",
            "average_candidates",
            "tolerance_m",
            "end_to_end_accuracy_percent"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:", output_path)
