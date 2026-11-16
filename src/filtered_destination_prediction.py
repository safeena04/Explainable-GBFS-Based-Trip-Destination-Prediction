import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
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
# 1. Experiment configurations
# --------------------------------------------------

EXPERIMENT_CONFIGS = {
    "E1": {"max_distance": 1000, "max_time": 600},
    "E2": {"max_distance": 1000, "max_time": 1200},
    "E3": {"max_distance": 2000, "max_time": 600},
    "E4": {"max_distance": 2000, "max_time": 1200},
    "E5": {"max_distance": 3000, "max_time": 600},
    "E6": {"max_distance": 3000, "max_time": 1200},
    "Baseline": {"max_distance": 3000, "max_time": 1800},
}


# --------------------------------------------------
# 2. Load model, scaler and data
# --------------------------------------------------

print("Loading model and scaler...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Loading candidate datasets...")

candidates = pd.read_csv(
    PROJECT_ROOT / "data" / "processed"
    / "baseline_candidates.csv"
)

rental_coverage = pd.read_csv(
    METRICS_DIR / "baseline_rental_coverage.csv"
)

# Use validation dates only
validation_candidates = candidates[
    candidates["date"].isin(VAL_DATES)
].copy()

validation_rentals = rental_coverage[
    rental_coverage["date"].isin(VAL_DATES)
].copy()

print("Validation rentals:", len(validation_rentals))
print("Validation candidate pairs:", len(validation_candidates))


# --------------------------------------------------
# 3. Prepare model features
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
# 4. Apply candidate filtering
# --------------------------------------------------

def filter_candidates(data, config):

    filtered = data[
        (data["distance"] <= config["max_distance"])
        & (data["time_diff"] <= config["max_time"])
        & (data["battery_diff"] <= MAX_BATTERY_DIFF)
        & (data["speed"] <= MAX_SPEED_KMH)
    ].copy()

    return filtered


# --------------------------------------------------
# 5. Predict destination coordinates
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
# 6. Calculate prediction distances
# --------------------------------------------------

def calculate_distances(data):

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

        distance = geodesic(
            predicted,
            candidate_return
        ).meters

        distances.append(distance)

    data = data.copy()
    data["prediction_distance_m"] = distances

    return data


# --------------------------------------------------
# 7. Select the nearest candidate per rental
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
# 8. Evaluate one experiment
# --------------------------------------------------

def evaluate_experiment(name, config):

    print(f"\nRunning experiment: {name}")

    filtered = filter_candidates(
        validation_candidates,
        config
    )

    print("Remaining candidate pairs:", len(filtered))

    # Candidate counts for every rental, including zero
    candidate_counts = (
        filtered.groupby("spoofed_id_rental")
        .size()
    )

    all_rental_ids = validation_rentals["rental_id"]

    candidate_counts = (
        all_rental_ids.map(candidate_counts)
        .fillna(0)
    )

    total_rentals = len(validation_rentals)

    available_rentals = int(
        (candidate_counts > 0).sum()
    )

    # Correct candidate retained
    correct_candidate_count = int(
        (
            filtered["vehicle_id_rental"]
            == filtered["vehicle_id_return"]
        ).sum()
    )

    correct_rentals = int(
        (
            filtered["vehicle_id_rental"]
            == filtered["vehicle_id_return"]
        ).groupby(
            filtered["spoofed_id_rental"]
        ).any().sum()
    ) if not filtered.empty else 0

    availability = (
        available_rentals / total_rentals * 100
    )

    coverage = (
        correct_rentals / total_rentals * 100
    )

    average_candidates = candidate_counts.mean()

    # Generate predictions
    predicted = predict_destinations(filtered)

    # Calculate distance to each candidate
    predicted = calculate_distances(predicted)

    # Select best candidate for each rental
    selected = select_best_destinations(predicted)

    if not selected.empty:

        selected["is_correct_destination"] = (
            selected["vehicle_id_rental"]
            == selected["vehicle_id_return"]
        )

    print("Rentals with predictions:", len(selected))
    print(f"Candidate availability: {availability:.2f}%")
    print(f"Candidate coverage: {coverage:.2f}%")
    print(f"Average candidates: {average_candidates:.2f}")

    results = []

    for tolerance in GEO_TOLERANCES_M:

        if selected.empty:
            correct_predictions = 0
        else:
            correct_mask = (
                selected["is_correct_destination"]
                & (
                    selected["prediction_distance_m"]
                    <= tolerance
                )
            )

            correct_predictions = int(correct_mask.sum())

        conditional_accuracy = (
            correct_predictions / available_rentals * 100
            if available_rentals else 0
        )

        end_to_end_accuracy = (
            correct_predictions / total_rentals * 100
            if total_rentals else 0
        )

        print(
            f"{tolerance}m accuracy: "
            f"{conditional_accuracy:.2f}% conditional, "
            f"{end_to_end_accuracy:.2f}% end-to-end"
        )

        results.append({
            "experiment": name,
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
            "correct_predictions": correct_predictions,
            "conditional_accuracy_percent": conditional_accuracy,
            "end_to_end_accuracy_percent": end_to_end_accuracy,
        })

    # Save selected predictions for this configuration
    output_file = (
        PREDICTIONS_DIR /
        f"validation_{name.lower()}_predictions.csv"
    )

    selected.to_csv(output_file, index=False)

    return results


# --------------------------------------------------
# 9. Run all validation experiments
# --------------------------------------------------

all_results = []

for name, config in EXPERIMENT_CONFIGS.items():

    experiment_results = evaluate_experiment(
        name,
        config
    )

    all_results.extend(experiment_results)


# --------------------------------------------------
# 10. Save experiment results
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

output_path = (
    METRICS_DIR /
    "validation_filtered_prediction_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\n==========================================")
print("ALL VALIDATION EXPERIMENTS COMPLETED")
print("Results saved to:", output_path)
print("==========================================")